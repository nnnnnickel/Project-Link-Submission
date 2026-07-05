from __future__ import annotations

import argparse
import json

from .agent import GoogleTarotMainAgent
from .tarot_memory import DEFAULT_SESSION_ID


def parse_cards(raw_cards: list[str] | None) -> list[str]:
    """Parse repeated or comma-separated card arguments."""
    if not raw_cards:
        return []
    cards: list[str] = []
    for item in raw_cards:
        cards.extend(card.strip() for card in item.split(",") if card.strip())
    return cards


def parse_args() -> argparse.Namespace:
    """Define the local/Kaggle command-line interface."""
    parser = argparse.ArgumentParser(description="Google ADK-compatible tarot capstone agent.")
    parser.add_argument("--query", "-q", required=True, help="User tarot question.")
    parser.add_argument("--cards", "-c", nargs="+", required=True, help="Drawn cards, repeated or comma-separated.")
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID, help="Conversation memory session id.")
    parser.add_argument("--reset-session", action="store_true", help="Clear this session before running.")
    parser.add_argument("--use-skills", action="store_true", help="Enable optional markdown skills.")
    parser.add_argument("--json", action="store_true", help="Print full JSON instead of final_response only.")
    return parser.parse_args()


def main() -> None:
    """Run one tarot reading from the command line."""
    args = parse_args()
    agent = GoogleTarotMainAgent(use_skills=args.use_skills)
    result = agent.run_pipeline(
        user_query=args.query,
        cards=parse_cards(args.cards),
        session_id=args.session_id,
        reset_session=args.reset_session,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["final_response"])


if __name__ == "__main__":
    main()
