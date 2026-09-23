"""End-to-end local verification. All resulting runs are marked simulated."""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from qbench.cli import main


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        return

    def do_GET(self):
        if self.path.endswith("/models"):
            payload = json.dumps({"data": [{"id": "mock-model"}]}).encode()
            self.send_response(200)
        else:
            payload = b"{}"
            self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if data.get("model") != "mock-model":
            payload = b'{"error":{"message":"model not found"}}'
            self.send_response(404)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        completion_id = "chatcmpl-mock-" + uuid.uuid4().hex
        created = int(time.time())
        model = data["model"]
        answer = "\\boxed{42}"
        self.send_response(200)
        if data.get("stream"):
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            chunks = [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "checking"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}},
            ]
            for chunk in chunks:
                event = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                         "model": model, **chunk}
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
        else:
            payload = json.dumps({"id": completion_id, "object": "chat.completion", "created": created,
                                  "model": model,
                                  "choices": [{"index": 0, "message": {"role": "assistant", "content": answer,
                                                                        "reasoning_content": "checking"},
                                               "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}}).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


def run() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    config = Path("data/mock_models.yaml")
    config.parent.mkdir(exist_ok=True)
    config.write_text(f"""models:
  mock:
    provider: openai_compatible
    base_url: http://127.0.0.1:{server.server_port}/v1
    model: mock-model
    simulated: true
quality:
  output_budget: 32
  timeout_seconds: 30
performance:
  concurrency: [1]
  requests_per_round: 2
  rounds: 3
  warmup_requests: 1
  output_budget: 8
  timeout_seconds: 30
""", encoding="utf-8")
    try:
        for command in (["doctor", "--model", "mock"],
                        ["run", "--model", "mock", "--profile", "smoke",
                         "--concurrency", "1", "--requests", "2", "--output", "outputs/mock"]):
            code = main([command[0], "--config", str(config), *command[1:]])
            if code:
                return code
        return main(["summarize", "--input", "outputs/mock", "--reference", "mock",
                     "--output", "reports/mock", "--simulation-only"])
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(run())
