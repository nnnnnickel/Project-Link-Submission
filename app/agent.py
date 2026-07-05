from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, TypedDict

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - Kaggle users install google-genai from requirements.txt.
    genai = None
    genai_types = None

try:
    from google.adk.agents import Agent
except ImportError:  # pragma: no cover - allows CLI smoke tests without ADK installed.
    Agent = None

from .card_input_normalizer import standardize_card_inputs
from .graphrag_tool import GraphRAGTool
from .skill_registry import SkillRegistry, SkillSelector
from .tarot_memory import DEFAULT_SESSION_ID, ConversationMemory, FileConversationMemoryStore


# Default Gemini model used by both the ADK wrapper and the direct pipeline.
# Users can override it with GOOGLE_ADK_MODEL in Kaggle secrets or a local .env file.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class TarotRunState(TypedDict, total=False):
    """Typed state object passed through the deterministic Google tarot pipeline.

    This mirrors a Kaggle notebook's cell-by-cell data flow: every stage adds
    a small, inspectable group of fields instead of hiding the full process in
    one large function.
    """

    # Original user question, for example: "How is my love life today? I am Gemini."
    user_query: str

    # Cards supplied in this invocation after normalization, preserving reversed markers.
    input_cards: list[str]

    # Stable session id used by the JSON memory store.
    session_id: str

    # When true, previous JSON memory for this session is deleted before the run.
    reset_session: bool

    # ConversationMemory object loaded from disk for the current session.
    memory: ConversationMemory

    # JSON-serializable view of recent and relevant previous turns.
    memory_context: dict[str, Any]

    # Lightweight topic label used only to choose retrieval/astro context.
    topic: str

    # True when the memory store already has at least one previous turn.
    is_followup: bool

    # Full card spread considered by the current answer.
    all_cards: list[str]

    # Cards that should be looked up in GraphRAG during this run.
    retrieval_cards: list[str]

    # Extra cards added during a follow-up question.
    supplemental_cards: list[str]

    # One of initial, followup_memory, or followup_supplement.
    turn_mode: str

    # Raw structured evidence from the GraphRAG parquet adapter.
    graph_knowledge: dict[str, Any]

    # Compact prompt payload sent to Gemini for the final reading.
    reader_payload: dict[str, Any]

    # Daily astrology context; unavailable lookups are represented explicitly.
    astro_context: dict[str, Any] | None

    # Optional markdown instruction block selected from app/Skills.
    skill_instructions: str

    # Runtime metadata for selected skills.
    selected_skills: list[dict[str, str]]

    # Whether optional skills were enabled for this turn.
    skills_enabled: bool

    # Gemini-generated draft, or a clear call-failed message.
    draft_response: str

    # Gemini safety classifier output, or an unavailable marker.
    safety_analysis: dict[str, Any]

    # Final user-facing answer after safety note application.
    final_response: str

    # Complete JSON result returned to CLI, ADK tools, or Kaggle notebook cells.
    result: dict[str, Any]


class GeminiService:
    """Small Google Gemini client wrapper used by the pipeline.

    The wrapper keeps Google-specific API details out of the orchestration
    class. This makes the main agent easier to read and lets Kaggle users see
    exactly where model calls happen.
    """

    def __init__(self, model: str | None = None):
        # Model name used for reading, safety classification, and daily-astro summarization.
        self.model = model or os.getenv("GOOGLE_ADK_MODEL") or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL

        # Client is None when google-genai is missing or credentials are absent.
        self.client = self._build_client()

    @property
    def available(self) -> bool:
        """Return true when the Google GenAI client was successfully created."""
        return self.client is not None

    def generate_text(self, system_instruction: str, payload: dict[str, Any], temperature: float = 0.35) -> str:
        """Call Gemini with a structured JSON payload and return plain text."""
        if not self.client or genai_types is None:
            raise RuntimeError("Google Gemini client is unavailable.")

        # A single string is easier to inspect in Kaggle output than nested Content objects.
        prompt = f"{system_instruction}\n\nINPUT_JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=temperature),
        )
        return str(getattr(response, "text", "") or "").strip()

    def generate_json(self, system_instruction: str, payload: dict[str, Any], temperature: float = 0.0) -> dict[str, Any]:
        """Call Gemini and parse a JSON object from the response text."""
        text = self.generate_text(system_instruction, payload, temperature=temperature)
        return parse_json_object(text)

    def search_daily_astro(self, sign: str, topic: str, query: str) -> dict[str, Any]:
        """Use Gemini grounding/search when available to obtain daily astrology context."""
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
        """Create either an AI Studio or Vertex AI Gemini client from environment variables."""
        if genai is None:
            return None
        try:
            use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
            if use_vertex:
                project = os.getenv("GOOGLE_CLOUD_PROJECT")
                location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
                if not project:
                    return None
                return genai.Client(vertexai=True, project=project, location=location)
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                return None
            return genai.Client(api_key=api_key)
        except Exception:
            return None


class GoogleTarotMainAgent:
    """Google/Kaggle implementation of the capstone tarot reader.

    The class remains deterministic at the orchestration layer:
    Gemini is used for generation, summarization, and safety classification,
    while Python functions decide when to load memory, retrieve GraphRAG data,
    call astrology lookup, and persist the run.
    """

    def __init__(
        self,
        model: str | None = None,
        max_memory_turns: int = 8,
        memory_root: str | Path | None = None,
        use_skills: bool = False,
        skill_path: str | Path | None = None,
    ):
        # `gemini` owns all model calls so the main agent can stay framework-agnostic.
        self.gemini = GeminiService(model=model)

        # `graph_tool` reads the existing GraphRAG parquet output from the original project.
        self.graph_tool = GraphRAGTool()

        # `max_memory_turns` limits how much session history is retained and prompted.
        self.max_memory_turns = max_memory_turns

        # `memory_store` persists lightweight JSON session history for follow-up questions.
        self.memory_store = FileConversationMemoryStore(max_turns=max_memory_turns, root=Path(memory_root) if memory_root else None)

        # `use_skills` is opt-in, matching the Capstone Project behavior.
        self.use_skills = use_skills

        # `skill_registry` loads markdown skill files, and `skill_selector` picks relevant ones.
        self.skill_registry = SkillRegistry(skill_path=skill_path)
        self.skill_selector = SkillSelector(self.skill_registry)

    def run_pipeline(
        self,
        user_query: str,
        cards: list[str] | None = None,
        session_id: str = DEFAULT_SESSION_ID,
        reset_session: bool = False,
        use_skills: bool | None = None,
        skill_instructions: str | None = None,
    ) -> dict[str, Any]:
        """Run one complete tarot reading turn and return a JSON-serializable result."""
        # Initial state contains only external inputs; every later helper adds its own fields.
        state: TarotRunState = {
            "user_query": user_query,
            "input_cards": standardize_card_inputs(cards or []),
            "session_id": session_id,
            "reset_session": reset_session,
        }

        # Stage 1: load memory, resolve cards, classify broad topic, and select optional skills.
        state.update(self._prepare_context(state, use_skills=use_skills, skill_instructions=skill_instructions))

        # Stage 2: retrieve GraphRAG evidence, or skip retrieval for a pure memory follow-up.
        state.update(self._retrieve_or_use_memory(state))

        # Stage 3: add daily astrology context only when the user's own zodiac sign is present.
        state["astro_context"] = self._read_daily_astro_if_available(state["user_query"], state["topic"])

        # Stage 4: ask Gemini to write the integrated tarot answer, or report a call failure.
        state["draft_response"] = self._generate_integrated_reading(state)

        # Stage 5: ask Gemini to classify safety, or report that the safety check could not run.
        state["safety_analysis"] = self._check_safety(state)

        # Stage 6: apply a safety note only when the classifier explicitly marks the draft unsafe.
        state["final_response"] = self._apply_safety_note(state["draft_response"], state["safety_analysis"])

        # Stage 7: persist memory and assemble the full result payload.
        state["result"] = self._persist_and_build_result(state)
        return state["result"]

    def get_memory_context(self, session_id: str = DEFAULT_SESSION_ID) -> dict[str, Any]:
        """Return the current structured memory context for debugging or Kaggle display."""
        return self.memory_store.load(session_id).as_context()

    def reset_memory(self, session_id: str = DEFAULT_SESSION_ID) -> None:
        """Delete one session's JSON memory file."""
        self.memory_store.delete(session_id)

    def _prepare_context(self, state: TarotRunState, use_skills: bool | None, skill_instructions: str | None) -> dict[str, Any]:
        """Prepare memory, cards, broad topic, and optional Skill.md instructions."""
        memory = self.memory_store.load(state["session_id"], reset_session=state.get("reset_session", False))
        card_resolution = self._resolve_turn_cards(state.get("input_cards", []), memory)
        topic = self._classify_topic_with_memory(state["user_query"], memory)
        skill_resolution = self._select_skills_for_turn(
            user_query=state["user_query"],
            cards=card_resolution["all_cards"],
            topic=topic,
            turn_mode=card_resolution["turn_mode"],
            enabled=self.use_skills if use_skills is None else use_skills,
            override=skill_instructions,
        )
        return {
            "memory": memory,
            "memory_context": memory.as_context(query=state["user_query"]),
            "topic": topic,
            "is_followup": bool(memory.turns),
            **card_resolution,
            **skill_resolution,
        }

    def _retrieve_or_use_memory(self, state: TarotRunState) -> dict[str, Any]:
        """Build retrieval evidence for first turns and extra-card follow-ups."""
        if state["turn_mode"] == "followup_memory":
            graph_knowledge = self._empty_memory_graph_result(state["all_cards"], state["topic"])
            reader_payload = self._build_memory_followup_payload(state["user_query"], state["memory_context"], state["topic"])
        else:
            graph_knowledge = self.graph_tool.retrieve(state["retrieval_cards"], topic=state["topic"])
            reader_payload = self._build_reader_payload(state["user_query"], graph_knowledge, state["topic"])

        # These fields are shared by both retrieval and memory-only modes.
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

    def _read_daily_astro_if_available(self, user_query: str, topic: str) -> dict[str, Any] | None:
        """Extract a zodiac sign and retrieve daily astrology context with Gemini Search."""
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

    def _generate_integrated_reading(self, state: TarotRunState) -> str:
        """Generate the final tarot reading with Gemini, without local substitute text."""
        if not self.gemini.available:
            #return "调用失败：Google Gemini 未配置或依赖不可用，因此没有生成替代解读。请配置 GOOGLE_API_KEY 或 Vertex AI 环境变量后重试。"
            return "Call failed: Google Gemini is not configured or is unavailable, so no interpretation was generated. Please configure the GOOGLE_API_KEY or Vertex AI and try again."
        try:
            payload = {
                "user_query": state["user_query"],
                **state["reader_payload"],
                "daily_astro_context": state.get("astro_context"),
                "skill_instructions": state.get("skill_instructions", ""),
            }
            return self.gemini.generate_text(self._reader_system_prompt(bool(state["reader_payload"].get("is_followup")), bool(state.get("skill_instructions"))), payload)
        except Exception as exc:
            #return f"调用失败：Google Gemini 主读牌调用失败，因此没有生成替代解读。错误信息：{exc}"
            return "Call Failed: The Google Gemini call failed, so no interpretation was generated. Error message: {exc}"

    def _check_safety(self, state: TarotRunState) -> dict[str, Any]:
        """Classify the generated answer for high-risk deterministic claims."""
        if not self.gemini.available:
            return {
                "available": False,
                "is_safe": True,
                "safety_reason": "Safety check was not run because Google Gemini is unavailable.",
                "source": "safety_unavailable",
            }
        try:
            result = self.gemini.generate_json(
                "Return ONLY JSON with keys is_safe and safety_reason. Unsafe means deterministic medical, death, legal, financial, relationship-fact, or harmful claims.",
                {
                    "user_query": state["user_query"],
                    "draft_reading": state["draft_response"],
                    "topic": state["topic"],
                    "astro_context": state.get("astro_context"),
                },
                temperature=0.0,
            )
            return {"available": True, "is_safe": bool(result.get("is_safe", True)), "safety_reason": str(result.get("safety_reason", "")), "source": "gemini"}
        except Exception as exc:
            return {"available": False, "is_safe": True, "safety_reason": f"Safety check failed: {exc}", "source": "safety_failed"}

    def _persist_and_build_result(self, state: TarotRunState) -> dict[str, Any]:
        """Save this turn to memory and assemble the complete output dictionary."""
        memory = state["memory"]
        memory.add_turn(
            user_query=state["user_query"],
            cards=state["all_cards"],
            topic=state["topic"],
            final_response=state["final_response"],
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
            "astro_retrieval_sources": self._astro_retrieval_sources(state.get("astro_context")),
            "safety_analysis": state["safety_analysis"],
            "element_summary": graph_knowledge["element_analysis"],
            "graph_chains": graph_knowledge["graph_chains"],
            "retrieval_meta": graph_knowledge["retrieval_meta"],
            "safety_triggered": not state["safety_analysis"]["is_safe"],
            "final_response": state["final_response"],
            "runtime": {
                "framework": "google_adk_compatible",
                "model": self.gemini.model if self.gemini.available else None,
                "gemini_available": self.gemini.available,
                "skill_loaded": bool(state.get("skill_instructions")),
                "skill_path": state["selected_skills"][0]["path"] if state.get("selected_skills") else None,
                "skills_enabled": state.get("skills_enabled", False),
                "selected_skills": state.get("selected_skills", []),
            },
        }

    def _resolve_turn_cards(self, cards: list[str], memory: ConversationMemory) -> dict[str, Any]:
        """Decide whether this turn is a first reading, memory follow-up, or extra-card follow-up."""
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

    def _classify_topic_with_memory(self, query: str, memory: ConversationMemory) -> str:
        """Use a broad topic label for retrieval, inheriting previous topic for vague follow-ups."""
        topic = classify_topic(query)
        return memory.last_topic if topic == "general" and memory.turns else topic

    def _build_reader_payload(self, user_query: str, graph_knowledge: dict[str, Any], topic: str) -> dict[str, Any]:
        """Compress GraphRAG output into the Gemini prompt payload."""
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

    def _build_memory_followup_payload(self, user_query: str, memory_context: dict[str, Any], topic: str) -> dict[str, Any]:
        """Build a prompt payload for follow-ups that reuse the previous spread."""
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

    def _empty_memory_graph_result(self, cards: list[str], topic: str) -> dict[str, Any]:
        """Return shape-compatible retrieval metadata when GraphRAG is skipped by design."""
        return {
            "card_meanings": [],
            "element_analysis": {"counts": {}, "dominant": None, "missing": [], "interpretation": "No new GraphRAG retrieval was run because this follow-up has no extra cards."},
            "astro_associations": [],
            "graph_chains": [],
            "retrieval_meta": {"status": "skipped", "source": "memory_only_followup", "topic": topic, "conversation_cards": cards},
        }

    def _select_skills_for_turn(self, user_query: str, cards: list[str], topic: str, turn_mode: str, enabled: bool, override: str | None) -> dict[str, Any]:
        """Select optional markdown reading guidance for this turn."""
        if not enabled:
            return {"skill_instructions": "", "selected_skills": [], "skills_enabled": False}
        if override is not None:
            return {
                "skill_instructions": shorten(override.strip(), 4000),
                "selected_skills": [{"name": "one_turn_override", "description": "Manually supplied skill instructions.", "path": ""}],
                "skills_enabled": True,
            }
        selected = self.skill_selector.select(user_query=user_query, cards=cards, context={"topic": topic, "turn_mode": turn_mode})
        return {
            "skill_instructions": self.skill_selector.build_instructions(selected),
            "selected_skills": [skill.to_runtime_dict() for skill in selected],
            "skills_enabled": True,
        }

    @staticmethod
    def _reader_system_prompt(is_followup: bool, has_skill: bool) -> str:
        """Create the Gemini instruction for first-turn and follow-up readings."""
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
        if has_skill:
            prompt += "\n\nOptional Skill.md guidance is included in the payload. Apply it only when relevant and keep safety constraints intact."
        return prompt

    @staticmethod
    def _apply_safety_note(draft_response: str, safety_result: dict[str, Any]) -> str:
        """Prepend a safety note only when Gemini explicitly flags the draft as unsafe."""
        if safety_result.get("is_safe", True):
            return draft_response
        reason = safety_result.get("safety_reason") or "Sensitive topic detected."
        return f"Safety note: {reason} Tarot can support reflection, but it should not replace qualified medical, legal, financial, or crisis support.\n\n{draft_response}"

    @staticmethod
    def _astro_retrieval_sources(astro_context: dict[str, Any] | None) -> list[dict[str, str]]:
        """Expose daily astrology web sources in a stable list format."""
        if not astro_context:
            return []
        sources = astro_context.get("sources")
        return sources if isinstance(sources, list) else []


def run_tarot_reading(user_query: str, cards_csv: str, session_id: str) -> dict[str, Any]:
    """Run the full tarot reader pipeline.

    Args:
        user_query: User's tarot question.
        cards_csv: Comma-separated card names, or an empty string for a follow-up.
        session_id: Conversation memory id.

    Returns:
        Full pipeline JSON including final_response and retrieval metadata.
    """
    agent = GoogleTarotMainAgent()
    cards = [item.strip() for item in cards_csv.split(",") if item.strip()]
    return agent.run_pipeline(user_query=user_query, cards=cards or None, session_id=session_id)


def get_tarot_memory(session_id: str) -> dict[str, Any]:
    """Return the saved memory context for a session."""
    return GoogleTarotMainAgent().get_memory_context(session_id)


def reset_tarot_memory(session_id: str) -> dict[str, str]:
    """Delete saved memory for a session."""
    GoogleTarotMainAgent().reset_memory(session_id)
    return {"status": "success", "message": f"Session {session_id!r} was reset."}


def classify_topic(query: str) -> str:
    """Classify broad topic for retrieval context only."""
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


def extract_user_sign(query: str) -> dict[str, Any]:
    """Extract the user's zodiac sign from a natural-language question."""
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


def tokenize(text: str) -> list[str]:
    """Lowercase and split text for small keyword/sign matchers."""
    return re.sub(r"[^\w\u4e00-\u9fff']+", " ", text.lower()).split()


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a model response that should contain one JSON object."""
    cleaned = text.strip()
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
    """Trim long evidence snippets before sending them to Gemini."""
    cleaned = " ".join(str(text).split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3].rstrip() + "..."


if Agent is not None:
    root_agent = Agent(
        name="google_tarot_reader",
        model=os.getenv("GOOGLE_ADK_MODEL", DEFAULT_GEMINI_MODEL),
        description="Google ADK tarot reader with GraphRAG retrieval, memory, daily astrology context, and safety checks.",
        instruction=(
            "You are the ADK wrapper for the Google Tarot Reader. "
            "For tarot questions, call run_tarot_reading with the user's question, comma-separated cards, and session id. "
            "Use get_tarot_memory only when the user asks to inspect memory, and reset_tarot_memory only when asked to reset."
        ),
        tools=[run_tarot_reading, get_tarot_memory, reset_tarot_memory],
    )
else:
    root_agent = None


# Refactored ADK 2.0 workflow implementation.
# The legacy definitions above are left only as a compatibility shell for older
# notebooks that import this module while the public symbols below point to the
# new SequentialAgent-based architecture.
from .agent_refactored import (  # noqa: E402,F401
    GeminiService,
    GoogleTarotMainAgent,
    TarotRunState,
    build_dynamic_skill_tools,
    classify_topic,
    create_reader_agent,
    create_safety_agent,
    create_tarot_pipeline,
    extract_user_sign,
    get_tarot_memory,
    reset_tarot_memory,
    root_agent,
    run_tarot_pipeline,
    run_tarot_reading,
)
