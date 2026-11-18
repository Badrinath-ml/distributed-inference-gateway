"""Deterministic local OpenAI-compatible stub for gateway tests.

Runs on 127.0.0.1 with an ephemeral port. It is intentionally dependency-free
and only supports the subset of the API the gateway proxy uses.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        created = int(time.time())
        model = "stub-model"
        chunks = [
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Hello"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " from stub."},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            },
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            },
        ]

        try:
            for chunk in chunks:
                line = f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                self.wfile.write(line)
                self.wfile.flush()
                time.sleep(0.002)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(200, {"status": "healthy"})
            return
        if self.path == "/v1/models":
            self._json(200, {"object": "list", "data": []})
            return
        self._json(404, {"detail": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            payload = {}

        if self.path != "/v1/chat/completions":
            self._json(404, {"detail": "not found"})
            return

        if payload.get("simulate_error"):
            self._json(500, {"detail": "simulated upstream failure"})
            return

        if payload.get("simulate_slow"):
            time.sleep(float(payload.get("slow_seconds", 1.0)))

        if payload.get("stream"):
            self._stream()
            return

        self._json(
            200,
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get("model", "stub-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello from stub.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            },
        )


class StubServer:
    def __init__(self) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="stub-openai-server",
            daemon=True,
        )

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "StubServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)
