from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, AsyncGenerator, TypedDict

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - Kaggle users install google-genai from requirements.txt.
    genai = None
    genai_types = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is listed in requirements.txt.
    load_dotenv = None

try:
    from google.adk.agents import Agent, BaseAgent, SequentialAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event, EventActions
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
except ImportError:  # pragma: no cover - keeps CLI/imports usable before ADK is installed.
    Agent = None
    BaseAgent = object
    SequentialAgent = None
    InvocationContext = Any
    Event = None
    EventActions = None
    Runner = None
    InMemorySessionService = None

from .card_input_normalizer import standardize_card_inputs
from .graphrag_tool import GraphRAGTool
from .tarot_memory import DEFAULT_SESSION_ID, ConversationMemory, FileConversationMemoryStore


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
APP_NAME = "app"
USER_ID = "tarot_user"
SKILLS_DIR = Path(__file__).resolve().parent / "skills"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_project_env() -> None:
    """Load project-local .env values without requiring python-dotenv."""
    env_path = PROJECT_ROOT / ".env"
    if load_dotenv is not None:
        load_dotenv(env_path)
        load_dotenv()
        return
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_project_env()


class TarotRunState(TypedDict, total=False):
    """Typed state object shared by the ADK workflow and local fallback runner."""

    user_query: str
    input_cards: list[str]
    session_id: str
    reset_session: bool
    memory: ConversationMemory
    memory_context: dict[str, Any]
    topic: str
    is_followup: bool
    all_cards: list[str]
    retrieval_cards: list[str]
    supplemental_cards: list[str]
    turn_mode: str
    graph_knowledge: dict[str, Any]
    reader_payload: dict[str, Any]
    astro_context: dict[str, Any] | None
    skills_enabled: bool
    discovered_skills: list[dict[str, str]]
    draft_response: str
    safety_analysis: dict[str, Any]
    final_response: str
    result: dict[str, Any]


class GeminiService:
    """Small Google Gemini client wrapper used by deterministic workflow nodes."""

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("GOOGLE_ADK_MODEL") or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.client = self._build_client()

    @property
    def available(self) -> bool:
        return self.client is not None

    def generate_text(self, system_instruction: str, payload: dict[str, Any], temperature: float = 0.35) -> str:
        if not self.client or genai_types is None:
            raise RuntimeError("Google Gemini client is unavailable.")
        prompt = f"{system_instruction}\n\nINPUT_JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=temperature),
        )
        return str(getattr(response, "text", "") or "").strip()

    def generate_json(self, system_instruction: str, payload: dict[str, Any], temperature: float = 0.0) -> dict[str, Any]:
        return parse_json_object(self.generate_text(system_instruction, payload, temperature=temperature))

    def search_daily_astro(self, sign: str, topic: str, query: str) -> dict[str, Any]:
        if not self.client or genai_types is None:
            raise RuntimeError("Google Gemini client is unavailable for daily astrology lookup.")
        search_tool = genai_types.Tool(google_search=genai_types.GoogleSearch())
        prompt = (
            "Search current public web context for today's daily horoscope. "
            "Return JSON with keys content and sources. Keep wording reflective, not deterministic.\n\n"
            + json.dumps({"sign": sign, "topic": topic, "query": query}, ensure_ascii=False)
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(tools=[search_tool], temperature=0.2),
        )
        parsed = parse_json_object(str(getattr(response, "text", "") or ""))
        content = str(parsed.get("content") or getattr(response, "text", "") or "").strip()
        if not content:
            raise RuntimeError("Daily astrology lookup returned no usable content.")
        return {"content": content, "sources": parsed.get("sources", []), "retrieval_method": "gemini_google_search", "search_model": self.model}

    def _build_client(self):
        if genai is None:
            return None
        try:
            if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true":
                project = os.getenv("GOOGLE_CLOUD_PROJECT")
                location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
                return genai.Client(vertexai=True, project=project, location=location) if project else None
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            return genai.Client(api_key=api_key) if api_key else None
        except Exception:
            return None


class TarotWorkflowServices:
    """Dependency holder reused by ADK nodes and the compatibility class."""

    def __init__(self, model: str | None = None, max_memory_turns: int = 8, memory_root: str | Path | None = None):
        self.gemini = GeminiService(model=model)
        self.graph_tool = GraphRAGTool()
        self.memory_store = FileConversationMemoryStore(max_turns=max_memory_turns, root=Path(memory_root) if memory_root else None)
        self.current_memory: ConversationMemory | None = None

    def prepare_context(self, state: TarotRunState) -> dict[str, Any]:
        memory = self.memory_store.load(state["session_id"], reset_session=state.get("reset_session", False))
        self.current_memory = memory
        card_resolution = resolve_turn_cards(state.get("input_cards", []), memory)
        topic = classify_topic_with_memory(state["user_query"], memory)
        prepared: dict[str, Any] = {
            "memory": memory,
            "memory_context": memory.as_context(query=state["user_query"]),
            "topic": topic,
            "is_followup": bool(memory.turns),
            "skills_enabled": bool(state.get("skills_enabled", False)),
            "discovered_skills": discover_skill_metadata() if state.get("skills_enabled", False) else [],
            **card_resolution,
        }
        prepared.update(self.retrieve_or_use_memory({**state, **prepared}))
        prepared["astro_context"] = self.read_daily_astro_if_available(state["user_query"], topic)
        return prepared

    def retrieve_or_use_memory(self, state: TarotRunState) -> dict[str, Any]:
        if state["turn_mode"] == "followup_memory":
            graph_knowledge = empty_memory_graph_result(state["all_cards"], state["topic"])
            reader_payload = build_memory_followup_payload(state["user_query"], state["memory_context"], state["topic"])
        else:
            graph_knowledge = self.graph_tool.retrieve(state["retrieval_cards"], topic=state["topic"])
            reader_payload = build_reader_payload(state["user_query"], graph_knowledge, state["topic"])
        reader_payload.update(
            {
                "conversation_memory": state["memory_context"],
                "is_followup": state["is_followup"],
                "turn_mode": state["turn_mode"],
                "all_cards": state["all_cards"],
                "retrieval_cards": state["retrieval_cards"],
                "supplemental_cards": state["supplemental_cards"],
            }
        )
        return {"graph_knowledge": graph_knowledge, "reader_payload": reader_payload}

    def read_daily_astro_if_available(self, user_query: str, topic: str) -> dict[str, Any] | None:
        extraction = extract_user_sign(user_query)
        if not extraction["user_sign"]:
            return None
        astro_topic = "love" if topic == "romance" else "money" if topic == "finance" else topic
        try:
            retrieved = self.gemini.search_daily_astro(extraction["user_sign"], astro_topic, user_query)
            summary = self.gemini.generate_text(
                "Summarize the daily astrology web context in 2-3 reflective, non-deterministic sentences.",
                {
                    "sign": extraction["user_sign"],
                    "topic": astro_topic,
                    "query": user_query,
                    "retrieved_daily_astro": shorten(retrieved.get("content", ""), 2500),
                    "sources": retrieved.get("sources", []),
                },
                temperature=0.2,
            )
            return {
                "available": True,
                "sign": extraction["user_sign"],
                "topic": astro_topic,
                "summary": summary,
                "source": "gemini_daily_astro_tool",
                "retrieval_method": retrieved.get("retrieval_method"),
                "search_model": retrieved.get("search_model"),
                "sources": retrieved.get("sources", []),
                "retrieved_excerpt": shorten(retrieved.get("content", ""), 900),
                "extraction": extraction,
            }
        except Exception as exc:
            return {
                "available": False,
                "sign": extraction["user_sign"],
                "topic": astro_topic,
                "summary": "",
                "source": "daily_astro_unavailable",
                "error": f"Google daily astrology lookup failed: {exc}",
                "extraction": extraction,
            }

    def generate_reading_locally(self, state: TarotRunState) -> str:
        if not self.gemini.available:
            return "Google Gemini is not configured or unavailable, so no substitute reading was generated. Please configure GOOGLE_API_KEY or Vertex AI environment variables and try again."
        payload = {
            "user_query": state["user_query"],
            **state["reader_payload"],
            "daily_astro_context": state.get("astro_context"),
            "available_skills": state.get("discovered_skills", []),
            "skill_loading_rule": "Skills are auto-discovered from app/skills and may be requested by the LLM when relevant.",
        }
        try:
            return self.gemini.generate_text(reader_system_prompt(bool(state["reader_payload"].get("is_followup")), bool(state.get("skills_enabled"))), payload)
        except Exception as exc:
            return f"Google Gemini reading call failed, so no substitute reading was generated. Error: {exc}"

    def check_safety_locally(self, state: TarotRunState) -> dict[str, Any]:
        if not self.gemini.available:
            return {"available": False, "is_safe": True, "safety_reason": "Safety check was not run because Google Gemini is unavailable.", "source": "safety_unavailable"}
        try:
            result = self.gemini.generate_json(
                "Return ONLY JSON with keys is_safe and safety_reason. Unsafe means deterministic medical, death, legal, financial, relationship-fact, or harmful claims.",
                {"user_query": state["user_query"], "draft_reading": state["draft_response"], "topic": state["topic"], "astro_context": state.get("astro_context")},
            )
            return {"available": True, "is_safe": bool(result.get("is_safe", True)), "safety_reason": str(result.get("safety_reason", "")), "source": "gemini"}
        except Exception as exc:
            return {"available": False, "is_safe": True, "safety_reason": f"Safety check failed: {exc}", "source": "safety_failed"}

    def finalize(self, state: TarotRunState) -> dict[str, Any]:
        final_response = apply_safety_note(state["draft_response"], state["safety_analysis"])
        memory = state.get("memory") or self.current_memory
        if memory is None:
            memory = self.memory_store.load(state["session_id"])
        memory.add_turn(
            user_query=state["user_query"],
            cards=state["all_cards"],
            topic=state["topic"],
            final_response=final_response,
            astro_context=state.get("astro_context"),
        )
        graph_knowledge = state["graph_knowledge"]
        return {
            "user_query": state["user_query"],
            "drawn_cards": state["all_cards"],
            "retrieval_cards": state["retrieval_cards"],
            "supplemental_cards": state["supplemental_cards"],
            "turn_mode": state["turn_mode"],
            "session_id": state["session_id"],
            "topic": state["topic"],
            "is_followup": state["is_followup"],
            "memory": memory.as_context(query=state["user_query"], exclude_last=True),
            "reader_payload": state["reader_payload"],
            "astro_context": state.get("astro_context"),
            "astro_retrieval_sources": astro_retrieval_sources(state.get("astro_context")),
            "safety_analysis": state["safety_analysis"],
            "element_summary": graph_knowledge["element_analysis"],
            "graph_chains": graph_knowledge["graph_chains"],
            "retrieval_meta": graph_knowledge["retrieval_meta"],
            "safety_triggered": not state["safety_analysis"].get("is_safe", True),
            "final_response": final_response,
            "runtime": {
                "framework": "adk_2_workflow",
                "model": self.gemini.model if self.gemini.available else None,
                "gemini_available": self.gemini.available,
                "skills_enabled": state.get("skills_enabled", False),
                "skill_loading": "adk_skill_toolset_dynamic" if state.get("skills_enabled") else "disabled",
                "discovered_skills": state.get("discovered_skills", []),
            },
        }


class ContextPrepAgent(BaseAgent):
    """ADK workflow node: prepares deterministic context and writes it to session.state."""

    def __init__(self, services: TarotWorkflowServices, name: str = "context_prep"):
        if Agent is not None:
            super().__init__(name=name)
        object.__setattr__(self, "services", services)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Any, None]:
        updates = self.services.prepare_context(ctx.session.state)
        updates.pop("memory", None)
        yield _state_event(self.name, updates)


class FinalizeAgent(BaseAgent):
    """ADK workflow node: applies safety, persists memory, and emits the final answer."""

    def __init__(self, services: TarotWorkflowServices, name: str = "finalize"):
        if Agent is not None:
            super().__init__(name=name)
        object.__setattr__(self, "services", services)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Any, None]:
        state = ctx.session.state
        state["safety_analysis"] = parse_json_object(str(state.get("safety_result", ""))) or {"available": True, "is_safe": True, "safety_reason": "", "source": "adk_safety_agent"}
        state["draft_response"] = str(state.get("draft_response", "")).strip()
        result = self.services.finalize(state)
        updates = {"final_response": result["final_response"], "result": result}
        yield _state_event(self.name, updates, text=result["final_response"])


def create_reader_agent(model: str | None = None, use_skills: bool = True):
    """Create the LLM reading node with automated ADK skill discovery."""
    if Agent is None:
        return None
    tools = build_dynamic_skill_tools() if use_skills else []
    return Agent(
        name="reader",
        model=model or os.getenv("GOOGLE_ADK_MODEL", DEFAULT_GEMINI_MODEL),
        description="Generates the integrated tarot reading from prepared workflow state.",
        instruction=reader_system_prompt(is_followup=False, skills_enabled=use_skills)
        + "\n\nUse state values {reader_payload}, {astro_context}, and {discovered_skills}. "
        + "When skills are enabled, inspect the automatically loaded skill tools only if the turn needs that guidance. "
        + "Return the user-facing tarot reading only.",
        tools=tools,
        output_key="draft_response",
    )


def create_safety_agent(model: str | None = None):
    """Create the LLM safety node that classifies the draft before persistence."""
    if Agent is None:
        return None
    return Agent(
        name="safety",
        model=model or os.getenv("GOOGLE_ADK_MODEL", DEFAULT_GEMINI_MODEL),
        description="Classifies tarot output for high-risk deterministic claims.",
        instruction=(
            "Return ONLY JSON with keys is_safe and safety_reason. "
            "Unsafe means deterministic medical, death, legal, financial, relationship-fact, or harmful claims. "
            "Evaluate this draft: {draft_response}"
        ),
        output_key="safety_result",
        include_contents="none",
    )


def create_tarot_pipeline(model: str | None = None, max_memory_turns: int = 8, memory_root: str | Path | None = None, use_skills: bool = True):
    """Build the ADK 2.0 SequentialAgent workflow."""
    if SequentialAgent is None:
        return None
    services = TarotWorkflowServices(model=model, max_memory_turns=max_memory_turns, memory_root=memory_root)
    return SequentialAgent(
        name="tarot_pipeline",
        sub_agents=[
            ContextPrepAgent(services),
            create_reader_agent(model=model, use_skills=use_skills),
            create_safety_agent(model=model),
            FinalizeAgent(services),
        ],
    )


def run_tarot_pipeline(user_query: str, cards_csv: str, session_id: str = DEFAULT_SESSION_ID, reset_session: bool = False, use_skills: bool = True) -> dict[str, Any]:
    """Run the ADK workflow. Falls back to the same nodes locally when ADK is absent."""
    cards = [item.strip() for item in cards_csv.split(",") if item.strip()]
    if Runner is not None and InMemorySessionService is not None and SequentialAgent is not None:
        try:
            return asyncio.run(_run_adk_pipeline(user_query, cards, session_id, reset_session, use_skills))
        except RuntimeError:
            # If called inside an existing event loop, use the deterministic path.
            pass
    services = TarotWorkflowServices()
    return run_local_workflow(services, user_query, cards, session_id, reset_session, use_skills)


async def _run_adk_pipeline(user_query: str, cards: list[str], session_id: str, reset_session: bool, use_skills: bool) -> dict[str, Any]:
    session_service = InMemorySessionService()
    pipeline = create_tarot_pipeline(use_skills=use_skills)
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
        state={
            "user_query": user_query,
            "input_cards": standardize_card_inputs(cards),
            "session_id": session_id,
            "reset_session": reset_session,
            "skills_enabled": use_skills,
        },
    )
    runner = Runner(agent=pipeline, app_name=APP_NAME, session_service=session_service)
    if genai_types is None:
        raise RuntimeError("google-genai is unavailable.")
    message = genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=user_query)])
    async for _event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=message):
        pass
    session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    return session.state.get("result", {"final_response": session.state.get("final_response", ""), "runtime": {"framework": "adk_2_workflow"}})


def run_local_workflow(services: TarotWorkflowServices, user_query: str, cards: list[str], session_id: str, reset_session: bool, use_skills: bool) -> dict[str, Any]:
    """Non-ADK execution path used for CLI smoke tests before google-adk is installed."""
    state: TarotRunState = {
        "user_query": user_query,
        "input_cards": standardize_card_inputs(cards),
        "session_id": session_id,
        "reset_session": reset_session,
        "skills_enabled": use_skills,
    }
    state.update(services.prepare_context(state))
    state["draft_response"] = services.generate_reading_locally(state)
    state["safety_analysis"] = services.check_safety_locally(state)
    state["result"] = services.finalize(state)
    return state["result"]


class GoogleTarotMainAgent:
    """Backward-compatible facade; implementation now delegates to the workflow nodes."""

    def __init__(
        self,
        model: str | None = None,
        max_memory_turns: int = 8,
        memory_root: str | Path | None = None,
        use_skills: bool = False,
        skill_path: str | Path | None = None,
    ):
        self.use_skills = use_skills
        self.skill_path = skill_path
        self.services = TarotWorkflowServices(model=model, max_memory_turns=max_memory_turns, memory_root=memory_root)

    def run_pipeline(
        self,
        user_query: str,
        cards: list[str] | None = None,
        session_id: str = DEFAULT_SESSION_ID,
        reset_session: bool = False,
        use_skills: bool | None = None,
        skill_instructions: str | None = None,
    ) -> dict[str, Any]:
        if skill_instructions:
            raise ValueError("Manual one-turn skill instructions were removed. Add a SKILL.md directory under app/skills instead.")
        enabled = self.use_skills if use_skills is None else use_skills
        return run_local_workflow(self.services, user_query, cards or [], session_id, reset_session, enabled)

    def get_memory_context(self, session_id: str = DEFAULT_SESSION_ID) -> dict[str, Any]:
        return self.services.memory_store.load(session_id).as_context()

    def reset_memory(self, session_id: str = DEFAULT_SESSION_ID) -> None:
        self.services.memory_store.delete(session_id)


def run_tarot_reading(user_query: str, cards_csv: str, session_id: str) -> dict[str, Any]:
    """Compatibility tool name retained for older prompts and notebooks."""
    return run_tarot_pipeline(user_query=user_query, cards_csv=cards_csv, session_id=session_id, use_skills=True)


def get_tarot_memory(session_id: str) -> dict[str, Any]:
    return GoogleTarotMainAgent().get_memory_context(session_id)


def reset_tarot_memory(session_id: str) -> dict[str, str]:
    GoogleTarotMainAgent().reset_memory(session_id)
    return {"status": "success", "message": f"Session {session_id!r} was reset."}


def build_dynamic_skill_tools() -> list[Any]:
    """Load Agent Skill directories automatically; fall back to an on-demand skill lookup tool."""
    native_toolsets = build_native_skill_toolsets()
    if native_toolsets:
        return native_toolsets

    def load_tarot_skills() -> dict[str, Any]:
        """Return all automatically discovered local tarot skills."""
        return {"skills": [load_skill_document(path) for path in iter_skill_dirs()]}

    return [load_tarot_skills]


def build_native_skill_toolsets() -> list[Any]:
    """Best-effort bridge to ADK SkillToolset/load_skill_from_dir across SDK versions."""
    imports = [
        ("google.adk.tools.skill_toolset", "SkillToolset", "load_skill_from_dir"),
        ("google.adk.tools.skills", "SkillToolset", "load_skill_from_dir"),
        ("google.adk.tools.agent_skill", "SkillToolset", "load_skill_from_dir"),
    ]
    for module_name, class_name, loader_name in imports:
        try:
            module = __import__(module_name, fromlist=[class_name, loader_name])
            skill_toolset = getattr(module, class_name)
            load_skill_from_dir = getattr(module, loader_name)
            loaded_skills = [load_skill_from_dir(str(path)) for path in iter_skill_dirs()]
            if not loaded_skills:
                return []
            for factory in (
                lambda: [skill_toolset(skills=loaded_skills)],
                lambda: [skill_toolset(loaded_skills)],
                lambda: [skill_toolset(skill=skill) for skill in loaded_skills],
            ):
                try:
                    return factory()
                except TypeError:
                    continue
        except Exception:
            continue
    return []


def iter_skill_dirs() -> list[Path]:
    if not SKILLS_DIR.exists():
        return []
    return sorted(path for path in SKILLS_DIR.iterdir() if path.is_dir() and (path / "SKILL.md").exists())


def discover_skill_metadata() -> list[dict[str, str]]:
    return [{"name": load_skill_document(path).get("name", path.name), "path": str(path / "SKILL.md"), "description": load_skill_document(path).get("description", "")} for path in iter_skill_dirs()]


def load_skill_document(skill_dir: Path) -> dict[str, str]:
    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8", errors="ignore")
    metadata: dict[str, str] = {"name": skill_dir.name, "description": "", "instructions": text, "path": str(skill_dir / "SKILL.md")}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip().strip('"')
            metadata["instructions"] = parts[2].strip()
    return metadata


def resolve_turn_cards(cards: list[str], memory: ConversationMemory) -> dict[str, Any]:
    cleaned = [str(card).strip() for card in cards if str(card).strip()]
    if not memory.turns:
        if cleaned:
            return {"all_cards": cleaned, "retrieval_cards": cleaned, "supplemental_cards": [], "turn_mode": "initial"}
        raise ValueError("No cards provided. First turn must include drawn cards; follow-ups can omit them.")
    previous_cards = list(memory.last_cards)
    if not cleaned:
        return {"all_cards": previous_cards, "retrieval_cards": [], "supplemental_cards": [], "turn_mode": "followup_memory"}
    if previous_cards and cleaned[: len(previous_cards)] == previous_cards:
        supplemental_cards = cleaned[len(previous_cards) :]
        all_cards = cleaned
    else:
        supplemental_cards = cleaned
        all_cards = previous_cards + supplemental_cards
    if not supplemental_cards:
        return {"all_cards": all_cards or previous_cards, "retrieval_cards": [], "supplemental_cards": [], "turn_mode": "followup_memory"}
    return {"all_cards": all_cards, "retrieval_cards": supplemental_cards, "supplemental_cards": supplemental_cards, "turn_mode": "followup_supplement"}


def classify_topic(query: str) -> str:
    text = query.lower()
    topic_keywords = {
        "romance": {"love", "relationship", "partner", "crush", "marriage", "romance"},
        "career": {"career", "job", "work", "business", "promotion"},
        "finance": {"money", "finance", "income", "invest", "investment", "wealth"},
        "health": {"health", "illness", "sick", "body"},
    }
    for topic, keywords in topic_keywords.items():
        if any(keyword in text for keyword in keywords):
            return topic
    return "general"


def classify_topic_with_memory(query: str, memory: ConversationMemory) -> str:
    topic = classify_topic(query)
    return memory.last_topic if topic == "general" and memory.turns else topic


def extract_user_sign(query: str) -> dict[str, Any]:
    signs = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
    sign_by_token = {sign.lower(): sign for sign in signs}
    tokens = tokenize(query)
    mentions = [(sign_by_token[token], idx) for idx, token in enumerate(tokens) if token in sign_by_token]
    seen: list[str] = []
    for sign, _ in mentions:
        if sign not in seen:
            seen.append(sign)
    if not mentions:
        return {"user_sign": None, "mentioned_signs": [], "other_signs": []}
    if len(mentions) == 1:
        return {"user_sign": mentions[0][0], "mentioned_signs": seen, "other_signs": []}
    user_cues = {"i", "im", "i'm", "me", "my", "mine", "myself", "am", "as", "my sign"}
    other_cues = {"crush", "partner", "boyfriend", "girlfriend", "spouse", "he", "she", "they", "his", "her", "their"}
    scored = []
    for sign, idx in mentions:
        window = tokens[max(0, idx - 5) : idx] + tokens[idx + 1 : idx + 4]
        user_score = sum(1 for cue in user_cues if cue in window)
        other_score = sum(1 for cue in other_cues if cue in window)
        scored.append((user_score - other_score, user_score, -idx, sign))
    scored.sort(reverse=True)
    user_sign = scored[0][3] if scored[0][0] > 0 or scored[0][1] > 0 else mentions[0][0]
    return {"user_sign": user_sign, "mentioned_signs": seen, "other_signs": [sign for sign in seen if sign != user_sign]}


def build_reader_payload(user_query: str, graph_knowledge: dict[str, Any], topic: str) -> dict[str, Any]:
    card_readings = []
    for item in graph_knowledge["card_meanings"]:
        card_readings.append(
            {
                "position": item.get("position"),
                "card": item.get("card"),
                "meaning": shorten(item.get("meaning", ""), 500),
                "element": item.get("element"),
                "themes": item.get("themes", []),
                "status": item.get("status", "ok"),
                "message": item.get("message", ""),
                "community_reports": item.get("community_reports", [])[:1],
            }
        )
    return {
        "request_summary": shorten(user_query, 180),
        "topic": topic,
        "card_readings": card_readings,
        "element_distribution": graph_knowledge["element_analysis"],
        "astro_highlights": [
            {"card": link.get("card"), "target": link.get("target"), "description": shorten(link.get("description", ""), 240)}
            for link in graph_knowledge["astro_associations"][:5]
        ],
        "graph_chains": graph_knowledge["graph_chains"][:8],
        "retrieval_meta": graph_knowledge.get("retrieval_meta", {}),
    }


def build_memory_followup_payload(user_query: str, memory_context: dict[str, Any], topic: str) -> dict[str, Any]:
    recent_history = memory_context.get("recent_history", [])
    previous_answer = recent_history[-1].get("assistant", "") if recent_history else ""
    return {
        "request_summary": shorten(user_query, 180),
        "topic": topic,
        "card_readings": [],
        "element_distribution": {"counts": {}, "dominant": None, "missing": [], "interpretation": "No new cards were drawn for this follow-up."},
        "astro_highlights": [],
        "graph_chains": [],
        "previous_answer": previous_answer,
        "retrieval_meta": {"status": "skipped", "source": "memory_only_followup", "topic": topic},
    }


def empty_memory_graph_result(cards: list[str], topic: str) -> dict[str, Any]:
    return {
        "card_meanings": [],
        "element_analysis": {"counts": {}, "dominant": None, "missing": [], "interpretation": "No new GraphRAG retrieval was run because this follow-up has no extra cards."},
        "astro_associations": [],
        "graph_chains": [],
        "retrieval_meta": {"status": "skipped", "source": "memory_only_followup", "topic": topic, "conversation_cards": cards},
    }


def reader_system_prompt(is_followup: bool, skills_enabled: bool) -> str:
    if is_followup:
        prompt = (
            "You are a Tarot Reader answering a follow-up question. Use tarot as symbolic reflection, not factual prediction. "
            "If turn_mode is followup_memory, continue from previous_answer and answer directly. Keep under 400 words."
        )
    else:
        prompt = """
You are a Tarot Reader. Integrate retrieved GraphRAG evidence into a coherent tarot reading.

Reading style:
- Warm, grounded, psychologically useful.
- Synthesize cards, element balance, astrology links, community summaries, graph chains, memory, and daily astrology when present.
- If a retrieval item is missing or a tool reports failure, acknowledge uncertainty briefly instead of inventing evidence.
- For health, legal, relationship-fact, or financial decisions, avoid deterministic claims.

Output content:
1. A direct answer to the user's question.
2. A concise integrated reading of the retrieved evidence.
3. One practical reflection/action step.

Do not output JSON. Do not include raw retrieval metadata. Keep it under 550 words.
""".strip()
    if skills_enabled:
        prompt += "\n\nSkill guidance is loaded automatically from app/skills through ADK skill tools. Use it only when relevant."
    return prompt


def apply_safety_note(draft_response: str, safety_result: dict[str, Any]) -> str:
    if safety_result.get("is_safe", True):
        return draft_response
    reason = safety_result.get("safety_reason") or "Sensitive topic detected."
    return f"Safety note: {reason} Tarot can support reflection, but it should not replace qualified medical, legal, financial, or crisis support.\n\n{draft_response}"


def astro_retrieval_sources(astro_context: dict[str, Any] | None) -> list[dict[str, str]]:
    if not astro_context:
        return []
    sources = astro_context.get("sources")
    return sources if isinstance(sources, list) else []


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\u4e00-\u9fff']+", " ", text.lower()).split()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def shorten(text: str, limit: int) -> str:
    cleaned = " ".join(str(text).split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3].rstrip() + "..."


def _state_event(author: str, updates: dict[str, Any], text: str | None = None):
    if Event is None or EventActions is None:
        return None
    kwargs: dict[str, Any] = {"author": author, "actions": EventActions(state_delta=updates)}
    if text and genai_types is not None:
        kwargs["content"] = genai_types.Content(role="model", parts=[genai_types.Part.from_text(text=text)])
    return Event(**kwargs)


if Agent is not None:
    root_agent = Agent(
        name="google_tarot_reader",
        model=os.getenv("GOOGLE_ADK_MODEL", DEFAULT_GEMINI_MODEL),
        description="Google ADK tarot reader with a SequentialAgent workflow, GraphRAG retrieval, memory, dynamic skills, astrology context, and safety checks.",
        instruction=(
            "You are the ADK wrapper for the Google Tarot Reader. "
            "For tarot questions, call run_tarot_pipeline with the user's question, comma-separated cards, and session id. "
            "Use get_tarot_memory only when the user asks to inspect memory, and reset_tarot_memory only when asked to reset."
        ),
        tools=[run_tarot_pipeline, run_tarot_reading, get_tarot_memory, reset_tarot_memory],
    )
else:
    root_agent = None
