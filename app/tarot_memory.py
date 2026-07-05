from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_SESSION_ID = "default"


@dataclass
class ConversationMemory:
    """Small JSON-backed memory for follow-up questions."""

    max_turns: int = 8
    storage_path: Path | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)
    session_summary: str = ""
    last_cards: list[str] = field(default_factory=list)
    original_query: str = ""
    last_topic: str = "general"

    @classmethod
    def load(cls, storage_path: Path, max_turns: int) -> "ConversationMemory":
        if not storage_path.exists():
            return cls(max_turns=max_turns, storage_path=storage_path)
        try:
            payload = json.loads(storage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls(max_turns=max_turns, storage_path=storage_path)
        memory = cls(
            max_turns=max_turns,
            storage_path=storage_path,
            turns=payload.get("turns", [])[-max_turns:],
            session_summary=payload.get("session_summary", ""),
            last_cards=payload.get("last_cards", []),
            original_query=payload.get("original_query", ""),
            last_topic=payload.get("last_topic", payload.get("last_domain", "general")),
        )
        if memory.turns and not memory.session_summary:
            memory.session_summary = memory._summarize_recent_turns()
        return memory

    def add_turn(self, user_query: str, cards: list[str], topic: str, final_response: str, astro_context: dict[str, Any] | None = None) -> None:
        if not self.original_query:
            self.original_query = user_query
        self.last_cards = list(cards)
        self.last_topic = topic
        self.turns.append(
            {
                "user_query": user_query,
                "cards": list(cards),
                "topic": topic,
                "final_response": final_response,
                "astro_context": astro_context,
            }
        )
        self.turns = self.turns[-self.max_turns :]
        self.session_summary = self._summarize_recent_turns()
        self.save()

    def as_history(self) -> list[dict[str, Any]]:
        return [
            {
                "user": turn.get("user_query", ""),
                "assistant": turn.get("final_response", ""),
                "cards": turn.get("cards", []),
                "topic": turn.get("topic", "general"),
                "astro_context": turn.get("astro_context"),
            }
            for turn in self.turns
        ]

    def as_context(self, query: str | None = None, exclude_last: bool = False) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "session_summary": self.session_summary,
            "last_cards": self.last_cards,
            "last_topic": self.last_topic,
            "recent_history": self.as_history()[-4:],
            "relevant_history": self.retrieve_relevant(query, top_k=3, exclude_last=exclude_last),
            "turn_count": len(self.turns),
        }

    def retrieve_relevant(self, query: str | None, top_k: int = 3, exclude_last: bool = False) -> list[dict[str, Any]]:
        searchable_turns = self.turns[:-1] if exclude_last else self.turns
        if not query or not searchable_turns:
            return []
        query_tokens = self._search_tokens(query)
        scored = []
        for idx, turn in enumerate(searchable_turns):
            turn_tokens = self._search_tokens(self._turn_search_text(turn))
            score = len(query_tokens & turn_tokens) / max(len(query_tokens), 1)
            score += idx / max(len(searchable_turns), 1) * 0.001
            scored.append((score, idx, turn))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "turn": idx + 1,
                "user": turn.get("user_query", ""),
                "assistant": turn.get("final_response", ""),
                "cards": turn.get("cards", []),
                "topic": turn.get("topic", "general"),
                "relevance_score": round(score, 4),
            }
            for score, idx, turn in scored[:top_k]
        ]

    def save(self) -> None:
        if not self.storage_path:
            return
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(json.dumps(self._to_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _to_payload(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "session_summary": self.session_summary,
            "last_cards": self.last_cards,
            "original_query": self.original_query,
            "last_topic": self.last_topic,
        }

    def _summarize_recent_turns(self) -> str:
        snippets = []
        for idx, turn in enumerate(self.turns[-3:], start=max(1, len(self.turns) - 2)):
            query = self._shorten(turn.get("user_query", ""), 120)
            answer = self._shorten(turn.get("final_response", ""), 180)
            snippets.append(f"Turn {idx}: user asked {query!r}; answer focused on {answer!r}.")
        return " ".join(snippets)

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        cleaned = " ".join(str(text).split())
        return cleaned if len(cleaned) <= limit else cleaned[: limit - 3].rstrip() + "..."

    @classmethod
    def _turn_search_text(cls, turn: dict[str, Any]) -> str:
        return " ".join(
            [
                str(turn.get("user_query", "")),
                str(turn.get("final_response", "")),
                " ".join(str(card) for card in turn.get("cards", [])),
                str(turn.get("topic", "")),
                str(turn.get("astro_context", "")),
            ]
        )

    @staticmethod
    def _search_tokens(text: str) -> set[str]:
        lowered = str(text).lower()
        tokens = {token for token in re.findall(r"[a-z0-9_]+", lowered) if len(token) >= 2}
        cjk_chars = {char for char in lowered if "\u4e00" <= char <= "\u9fff"}
        return tokens | cjk_chars


class FileConversationMemoryStore:
    """Manage one JSON memory file per session."""

    def __init__(self, max_turns: int, root: Path | None = None):
        self.max_turns = max_turns
        self.root = root or Path(__file__).resolve().parent / "history"
        self.root.mkdir(parents=True, exist_ok=True)
        self._active_paths: set[Path] = set()

    def load(self, session_id: str, reset_session: bool = False) -> ConversationMemory:
        path = self._session_path(session_id)
        if reset_session:
            self.delete(session_id)
            self._active_paths.add(path)
            return ConversationMemory(max_turns=self.max_turns, storage_path=path)
        self._active_paths.add(path)
        return ConversationMemory.load(path, max_turns=self.max_turns)

    def delete(self, session_id: str) -> None:
        path = self._session_path(session_id)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        self._active_paths.discard(path)

    def cleanup(self) -> None:
        for path in list(self._active_paths):
            try:
                path.unlink()
            except (FileNotFoundError, OSError):
                pass
            self._active_paths.discard(path)

    def _session_path(self, session_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", session_id or DEFAULT_SESSION_ID).strip("._")
        return self.root / f"{(safe_id[:80] or DEFAULT_SESSION_ID)}.json"
