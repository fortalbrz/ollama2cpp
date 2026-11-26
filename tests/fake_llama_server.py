"""Stand-in for ``llama-server``: serves ``/health`` and ``/props`` on a local port.

Usage: ``python fake_llama_server.py PORT [PORT_FILE] [LOAD_SECONDS]``. PORT 0 picks a free port; the chosen port is
written to PORT_FILE. ``/health`` answers 503 until LOAD_SECONDS have passed, then 200.
"""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

START = time.monotonic()
LOAD_SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            ready = time.monotonic() - START >= LOAD_SECONDS
            self._send(200 if ready else 503, b'{"status":"ok"}' if ready else b'{"error":"loading"}')
        elif self.path == "/props":
            self._send(200, json.dumps({"model_alias": "fake-model"}).encode())
        else:
            self._send(404, b"{}")

    def log_message(self, format: str, *args: Any) -> None:  # ruff: ignore[builtin-argument-shadowing]
        pass


def main() -> None:
    server = HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler)
    if len(sys.argv) > 2 and sys.argv[2]:
        Path(sys.argv[2]).write_text(str(server.server_address[1]), encoding="utf-8")
    server.serve_forever()


if __name__ == "__main__":
    main()
