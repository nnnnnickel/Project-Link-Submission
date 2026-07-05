from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SkillDefinition:
    name: str
    description: str
    triggers: list[str]
    instructions: str
    path: Path

    def to_runtime_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description, "path": str(self.path)}


class SkillRegistry:
    """Load optional markdown skills from the local Skills folder."""

    def __init__(self, skills_dir: str | Path | None = None, skill_path: str | Path | None = None):
        self.base_dir = Path(__file__).resolve().parent
        self.skills_dir = Path(skills_dir) if skills_dir else self.base_dir / "Skills"
        self.skill_path = skill_path
        self.skills = self.reload()

    def reload(self) -> list[SkillDefinition]:
        if self.skill_path:
            path = self.resolve_skill_path(self.skill_path)
            skill = self._load_skill_file(path)
            return [skill] if skill else []
        if not self.skills_dir.exists():
            return []
        return [skill for path in sorted(self.skills_dir.glob("*.md")) if (skill := self._load_skill_file(path))]

    def resolve_skill_path(self, skill_path: str | Path | None = None) -> Path:
        path = Path(skill_path or self.skill_path or "")
        if path.is_absolute():
            return path
        return self.skills_dir / path

    def _load_skill_file(self, path: Path) -> SkillDefinition | None:
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8", errors="ignore")
        return SkillDefinition(
            name=self._section_value(content, "Name") or path.stem,
            description=self._section_value(content, "Description"),
            triggers=[item.strip().lower() for item in self._section_value(content, "Triggers").split(",") if item.strip()],
            instructions=self._section_value(content, "Instructions") or content.strip(),
            path=path,
        )

    @staticmethod
    def _section_value(content: str, heading: str) -> str:
        pattern = rf"(?ims)^#+\s*{re.escape(heading)}\s*$\s*(.*?)(?=^#+\s|\Z)"
        match = re.search(pattern, content)
        return match.group(1).strip() if match else ""


class SkillSelector:
    """Tiny keyword scorer for optional, human-authored reading guidance."""

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def select(self, user_query: str, cards: list[str] | None = None, context: dict[str, Any] | None = None) -> list[SkillDefinition]:
        search_text = self._turn_text(user_query, cards or [], context or {})
        scored = [(self._score(skill, search_text), skill) for skill in self.registry.skills]
        return [skill for score, skill in sorted(scored, key=lambda item: item[0], reverse=True) if score > 0][:2]

    def build_instructions(self, skills: list[SkillDefinition]) -> str:
        if not skills:
            return ""
        sections = []
        for skill in skills:
            sections.append(f"Skill: {skill.name}\nDescription: {skill.description}\nInstructions:\n{shorten(skill.instructions, 1600)}")
        return "\n\n".join(sections)

    def _score(self, skill: SkillDefinition, search_text: str) -> int:
        tokens = self._tokens(search_text)
        score = 0
        for trigger in skill.triggers:
            if trigger and trigger in search_text:
                score += 3
        for token in self._tokens(skill.description + " " + skill.instructions):
            if token in tokens:
                score += 1
        return score

    @staticmethod
    def _turn_text(user_query: str, cards: list[str], context: dict[str, Any]) -> str:
        return " ".join([user_query, " ".join(cards), str(context.get("topic", "")), str(context.get("turn_mode", ""))]).lower()

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower()) if token}


def shorten(text: str, limit: int) -> str:
    cleaned = " ".join(str(text).split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3].rstrip() + "..."
