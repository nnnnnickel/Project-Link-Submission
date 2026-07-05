from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CARD_ALIASES = {
    "the moon": "THE MOON",
    "the fool": "THE FOOL",
    "the magician": "THE MAGICIAN",
    "the high priestess": "THE HIGH PRIESTESS",
    "the empress": "THE EMPRESS",
    "the emperor": "THE EMPEROR",
    "the hierophant": "THE HIEROPHANT",
    "the lovers": "THE LOVERS",
    "the chariot": "THE CHARIOT",
    "strength": "STRENGTH",
    "wheel of fortune": "WHEEL OF FORTUNE",
    "justice": "JUSTICE",
    "the hanged man": "THE HANGED MAN",
    "death": "DEATH",
    "temperance": "TEMPERANCE",
    "the devil": "THE DEVIL",
    "the tower": "THE TOWER",
    "the star": "THE STAR",
    "the sun": "THE SUN",
    "judgement": "JUDGEMENT",
    "judgment": "JUDGEMENT",
    "the world": "THE WORLD",
}

SUIT_ELEMENTS = {"CUPS": "Water", "WANDS": "Fire", "SWORDS": "Air", "PENTACLES": "Earth"}
MAJOR_ARCANA_ELEMENTS = {
    "THE FOOL": "Air",
    "THE MAGICIAN": "Air",
    "THE HIGH PRIESTESS": "Water",
    "THE EMPRESS": "Earth",
    "THE EMPEROR": "Fire",
    "THE HIEROPHANT": "Earth",
    "THE LOVERS": "Air",
    "THE CHARIOT": "Water",
    "STRENGTH": "Fire",
    "THE HERMIT": "Earth",
    "WHEEL OF FORTUNE": "Fire",
    "JUSTICE": "Air",
    "THE HANGED MAN": "Water",
    "DEATH": "Water",
    "TEMPERANCE": "Fire",
    "THE DEVIL": "Earth",
    "THE TOWER": "Fire",
    "THE STAR": "Air",
    "THE MOON": "Water",
    "THE SUN": "Fire",
    "JUDGEMENT": "Fire",
    "THE WORLD": "Earth",
}


@dataclass
class GraphRAGResult:
    card_meanings: list[dict[str, Any]]
    element_analysis: dict[str, Any]
    astro_associations: list[dict[str, Any]]
    graph_chains: list[str]
    retrieval_meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GraphRAGTool:
    """Read GraphRAG parquet output and return compact evidence for the reader."""

    def __init__(self, graphrag_project_path: str | Path | None = None):
        self.base_dir = Path(__file__).resolve().parent
        default_source = self.base_dir.parent.parent / "tarot_project" / "graphrag-project-new"
        self.graphrag_project_path = Path(graphrag_project_path) if graphrag_project_path else default_source
        self.output_path = self.graphrag_project_path / "output"
        self.entities = None
        self.relationships = None
        self.community_reports = None
        self.load_error = ""
        self._load_tables()

    def retrieve(self, cards: list[str], topic: str = "general") -> dict[str, Any]:
        """Retrieve cards. If GraphRAG is unavailable, report the failure explicitly."""
        if self.entities is None or self.relationships is None:
            return self._failed_result(cards, topic, self.load_error or "GraphRAG parquet tables are unavailable.").to_dict()
        return self._retrieve_from_tables(cards, topic).to_dict()

    def _load_tables(self) -> None:
        try:
            import pandas as pd

            entities_path = self.output_path / "entities.parquet"
            relationships_path = self.output_path / "relationships.parquet"
            reports_path = self.output_path / "community_reports.parquet"
            if not entities_path.exists() or not relationships_path.exists():
                self.load_error = f"Missing parquet files under {self.output_path}."
                return
            self.entities = pd.read_parquet(entities_path)
            self.relationships = pd.read_parquet(relationships_path)
            if reports_path.exists():
                self.community_reports = pd.read_parquet(reports_path)
        except Exception as exc:  # import/parquet errors are reported, not hidden.
            self.entities = None
            self.relationships = None
            self.community_reports = None
            self.load_error = str(exc)

    def _retrieve_from_tables(self, cards: list[str], topic: str) -> GraphRAGResult:
        entity_by_title = {str(row["title"]).upper(): row for row in self.entities.to_dict("records") if row.get("title")}
        element_counter: Counter[str] = Counter()
        card_meanings: list[dict[str, Any]] = []
        astro_associations: list[dict[str, Any]] = []
        graph_chains: list[str] = []
        matched = 0

        for idx, raw_card in enumerate(cards):
            title = self._normalize_card_name(raw_card)
            entity = entity_by_title.get(title)
            if not entity:
                card_meanings.append(self._missing_card(raw_card, f"card_{idx + 1}"))
                continue
            matched += 1
            description = str(entity.get("description") or "")
            themes = self._extract_themes(description)
            element = self._infer_element(title, description)
            if element != "Unknown":
                element_counter[element] += 1
            one_hop = self._relationships_for(title)
            graph_chains.extend(self._two_hop_chains(title, one_hop)[:3])
            astro_associations.extend(self._astro_links(raw_card, title, one_hop))
            card_meanings.append(
                {
                    "position": f"card_{idx + 1}",
                    "card": raw_card,
                    "normalized_card": title,
                    "meaning": description,
                    "themes": themes,
                    "element": element,
                    "one_hop_relations": one_hop[:5],
                    "community_reports": self._community_summaries_for(title),
                }
            )

        return GraphRAGResult(
            card_meanings=card_meanings,
            element_analysis=self._element_analysis(element_counter, len(cards)),
            astro_associations=astro_associations[:8],
            graph_chains=graph_chains[:10],
            retrieval_meta={
                "status": "ok",
                "source": "graphrag_parquet",
                "project_path": str(self.graphrag_project_path),
                "topic": topic,
                "requested_cards": cards,
                "matched_cards": matched,
            },
        )

    def _normalize_card_name(self, card: str) -> str:
        clean = self._strip_card_orientation(card)
        lowered = clean.lower()
        if lowered in CARD_ALIASES:
            return CARD_ALIASES[lowered]
        match = re.match(r"^(ace|[2-9]|10|page|knight|queen|king|two|three|four|five|six|seven|eight|nine|ten)\s+of\s+(.+)$", lowered)
        if match:
            rank, suit = match.groups()
            rank_map = {"ace": "ACE", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
            return f"{rank_map.get(rank, rank.upper())} OF {suit.upper()}"
        return clean.upper()

    @staticmethod
    def _strip_card_orientation(card: str) -> str:
        clean = str(card).strip()
        orientation = r"(?:reversed|reverse|upright|\u9006\u4f4d|\u6b63\u4f4d)"
        clean = re.sub(rf"\s*[\(\[\uff08]\s*{orientation}\s*[\)\]\uff09]\s*", " ", clean, flags=re.I)
        clean = re.sub(rf"^\s*{orientation}\s*[:\uff1a\-]?\s+", "", clean, flags=re.I)
        clean = re.sub(rf"\s+[-:\uff1a]?\s*{orientation}\s*$", "", clean, flags=re.I)
        return re.sub(r"\s+", " ", clean).strip()

    def _relationships_for(self, title: str) -> list[dict[str, Any]]:
        rows = self.relationships[
            (self.relationships["source"].astype(str).str.upper() == title)
            | (self.relationships["target"].astype(str).str.upper() == title)
        ].sort_values("weight", ascending=False)
        return rows.head(12).to_dict("records")

    def _two_hop_chains(self, title: str, one_hop: list[dict[str, Any]]) -> list[str]:
        chains = []
        for rel in one_hop[:5]:
            middle = rel["target"] if str(rel["source"]).upper() == title else rel["source"]
            next_rows = self.relationships[
                (self.relationships["source"].astype(str).str.upper() == str(middle).upper())
                | (self.relationships["target"].astype(str).str.upper() == str(middle).upper())
            ].sort_values("weight", ascending=False)
            for next_rel in next_rows.head(2).to_dict("records"):
                end = next_rel["target"] if str(next_rel["source"]).upper() == str(middle).upper() else next_rel["source"]
                if str(end).upper() != title:
                    chains.append(f"{title} -> {middle} -> {end}")
        return chains

    def _community_summaries_for(self, title: str) -> list[dict[str, str]]:
        if self.community_reports is None:
            return []
        pretty_title = title.title()
        matches = self.community_reports[
            self.community_reports["full_content"].astype(str).str.contains(pretty_title, case=False, na=False)
            | self.community_reports["title"].astype(str).str.contains(pretty_title, case=False, na=False)
        ]
        return matches[["title", "summary"]].head(2).to_dict("records")

    @staticmethod
    def _extract_themes(description: str) -> list[str]:
        candidates = [
            "illusion",
            "intuition",
            "ambiguity",
            "subconscious",
            "fear",
            "heartbreak",
            "separation",
            "clarity",
            "conflict",
            "prosperity",
            "scarcity",
            "action",
            "passion",
            "stability",
            "communication",
            "choice",
            "transition",
            "transformation",
        ]
        found = [theme for theme in candidates if theme in description.lower()]
        return found[:6] or ["general interpretation"]

    @staticmethod
    def _infer_element(title: str, description: str) -> str:
        if title in MAJOR_ARCANA_ELEMENTS:
            return MAJOR_ARCANA_ELEMENTS[title]
        for suit, element in SUIT_ELEMENTS.items():
            if suit in title:
                return element
        lower = description.lower()
        for element in ["Water", "Fire", "Air", "Earth"]:
            if re.search(rf"\b{element.lower()}\b", lower):
                return element
        return "Unknown"

    def _astro_links(self, raw_card: str, title: str, one_hop: list[dict[str, Any]]) -> list[dict[str, Any]]:
        links = []
        for rel in one_hop:
            source = str(rel.get("source", ""))
            target = str(rel.get("target", ""))
            description = str(rel.get("description", ""))
            if self._looks_astro(source, target, description):
                links.append(
                    {
                        "card": raw_card,
                        "target": target if source.upper() == title else source,
                        "description": description,
                        "weight": rel.get("weight", 0),
                    }
                )
        return links

    @staticmethod
    def _looks_astro(source: str, target: str, description: str) -> bool:
        astro_terms = {
            "MOON",
            "SUN",
            "MERCURY",
            "VENUS",
            "MARS",
            "JUPITER",
            "SATURN",
            "ARIES",
            "TAURUS",
            "GEMINI",
            "CANCER",
            "LEO",
            "VIRGO",
            "LIBRA",
            "SCORPIO",
            "SAGITTARIUS",
            "CAPRICORN",
            "AQUARIUS",
            "PISCES",
        }
        return source.upper() in astro_terms or target.upper() in astro_terms or "associated with" in description.lower()

    def _element_analysis(self, counter: Counter[str], total_cards: int) -> dict[str, Any]:
        dominant = counter.most_common(1)[0][0] if counter else None
        missing = [element for element in ["Fire", "Water", "Air", "Earth"] if counter[element] == 0]
        if not dominant:
            interpretation = "Element balance is unavailable because no cards were matched."
        else:
            interpretation = f"{dominant} is the strongest visible element across {total_cards} card(s); missing elements: {', '.join(missing) if missing else 'none'}."
        return {"counts": dict(counter), "dominant": dominant, "missing": missing, "interpretation": interpretation}

    def _missing_card(self, card: str, position: str) -> dict[str, Any]:
        return {
            "position": position,
            "card": card,
            "normalized_card": self._normalize_card_name(card),
            "meaning": "",
            "themes": [],
            "element": "Unknown",
            "one_hop_relations": [],
            "community_reports": [],
            "status": "not_found",
            "message": "No exact GraphRAG entity was found for this card.",
        }

    def _failed_result(self, cards: list[str], topic: str, error: str) -> GraphRAGResult:
        return GraphRAGResult(
            card_meanings=[self._missing_card(card, f"card_{idx + 1}") for idx, card in enumerate(cards)],
            element_analysis=self._element_analysis(Counter(), len(cards)),
            astro_associations=[],
            graph_chains=[],
            retrieval_meta={
                "status": "failed",
                "source": "graphrag_parquet",
                "project_path": str(self.graphrag_project_path),
                "topic": topic,
                "requested_cards": cards,
                "matched_cards": 0,
                "error": error,
            },
        )
