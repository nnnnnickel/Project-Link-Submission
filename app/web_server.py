from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from .agent_refactored import GoogleTarotMainAgent
from .tarot_memory import DEFAULT_SESSION_ID


STATIC_DIR = Path(__file__).resolve().parent / "static"


class TarotWebHandler(BaseHTTPRequestHandler):
    """Small local web server for the Tarot Reader UI."""

    server_version = "TarotReaderWeb/1.0"

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_file(STATIC_DIR / "main.html")
            return
        requested = unquote(self.path.split("?", 1)[0]).lstrip("/")
        self._send_file(STATIC_DIR / requested)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/api/reading":
                self._send_json(self._run_reading(payload))
                return
            if self.path == "/api/exit":
                session_id = str(payload.get("session_id") or DEFAULT_SESSION_ID)
                GoogleTarotMainAgent().reset_memory(session_id)
                self._send_json({"status": "success", "message": "Session closed.", "session_id": session_id})
                return
            self._send_json({"error": "Unknown endpoint."}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"error": f"Server error: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[tarot-web] {self.address_string()} - {format % args}")

    def _run_reading(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question") or "").strip()
        cards = parse_cards(str(payload.get("cards") or ""))
        session_id = str(payload.get("session_id") or DEFAULT_SESSION_ID).strip() or DEFAULT_SESSION_ID
        is_followup = bool(payload.get("followup"))
        if not question:
            raise ValueError("Please enter a question.")
        if not is_followup and not cards:
            raise ValueError("Please enter at least one card for the first reading.")
        agent = GoogleTarotMainAgent(use_skills=bool(payload.get("use_skills", True)))
        result = agent.run_pipeline(
            user_query=question,
            cards=cards,
            session_id=session_id,
            reset_session=bool(payload.get("reset_session", False)),
        )
        return {
            "question": question,
            "answer": result.get("final_response", ""),
            "cards": result.get("drawn_cards", cards),
            "turn_mode": result.get("turn_mode", ""),
            "topic": result.get("topic", ""),
            "session_id": session_id,
            "runtime": result.get("runtime", {}),
        }

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object.")
        return data

    def _send_file(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            static_root = STATIC_DIR.resolve()
            if static_root not in resolved.parents and resolved != static_root:
                raise FileNotFoundError
            if not resolved.exists() or not resolved.is_file():
                raise FileNotFoundError
            content = resolved.read_bytes()
            content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def parse_cards(raw_cards: str) -> list[str]:
    """Parse cards separated by commas, Chinese commas, semicolons, or new lines."""
    normalized = raw_cards.replace("，", ",").replace("；", ",").replace(";", ",").replace("\n", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def main() -> None:
    host = "127.0.0.1"
    port = 7860
    server = ThreadingHTTPServer((host, port), TarotWebHandler)
    print(f"Tarot Reader UI running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
