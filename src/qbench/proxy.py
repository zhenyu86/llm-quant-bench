"""Loopback timing proxy for EvalScope perf. It preserves the service response.

EvalScope treats missing usage without a tokenizer as a failed request. The proxy
adds zero *only to EvalScope's internal stream* in that case; authoritative
qbench metrics are computed from its separate per-request records and retain N/A.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx


class TimingServer(ThreadingHTTPServer):
    def __init__(self, url: str, key: str | None, path: Path, timeout: int):
        super().__init__(("127.0.0.1", 0), Handler)
        self.url = url
        self.key = key
        self.path = path
        self.timeout_seconds = timeout
        self.lock = threading.Lock()
        self.next_id = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def save(self, record: dict) -> None:
        with self.lock:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        server: TimingServer = self.server  # type: ignore[assignment]
        with server.lock:
            server.next_id += 1
            request_id = server.next_id
        start = time.perf_counter()
        record = {"id": request_id, "start": start, "first_content_s": None, "first_answer_s": None,
                  "usage": None, "finish_reason": None, "reasoning_chars": 0, "answer_chars": 0,
                  "http_status": None, "error": None}
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            headers = {"Content-Type": "application/json"}
            if server.key:
                headers["Authorization"] = "Bearer " + server.key
            with httpx.Client(timeout=server.timeout_seconds) as client:
                with client.stream("POST", server.url, content=body, headers=headers) as upstream:
                    record["http_status"] = upstream.status_code
                    if upstream.status_code >= 400:
                        payload = upstream.read()
                        record["error"] = f"HTTP {upstream.status_code}: " + payload.decode("utf-8", "replace")[:300]
                        self.send_response(upstream.status_code)
                        self.send_header("Content-Type", upstream.headers.get("content-type", "application/json"))
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        done = False
                        for line in upstream.iter_lines():
                            if line.startswith("data:"):
                                data = line[5:].strip()
                                if data == "[DONE]":
                                    done = True
                                    if record["usage"] is None:
                                        self._write_event({"choices": [], "usage": {"prompt_tokens": 0,
                                                                                     "completion_tokens": 0,
                                                                                     "total_tokens": 0}})
                                    self._write_raw("data: [DONE]\n\n")
                                    continue
                                if data:
                                    try:
                                        event = json.loads(data)
                                        self._accept(record, event, start)
                                    except ValueError:
                                        pass
                                    self._write_raw(line + "\n\n")
                        if not done:
                            if record["usage"] is None:
                                self._write_event({"choices": [], "usage": {"prompt_tokens": 0,
                                                                             "completion_tokens": 0,
                                                                             "total_tokens": 0}})
                            self._write_raw("data: [DONE]\n\n")
                        self.close_connection = True
        except Exception as exc:
            record["error"] = type(exc).__name__ + ": " + str(exc)[:300]
            try:
                payload = json.dumps({"error": {"message": record["error"]}}).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionError):
                pass
        finally:
            record["end"] = time.perf_counter()
            record["elapsed_s"] = record["end"] - start
            server.save(record)

    def _write_raw(self, value: str) -> None:
        self.wfile.write(value.encode("utf-8"))
        self.wfile.flush()

    def _write_event(self, value: dict) -> None:
        self._write_raw("data: " + json.dumps(value, ensure_ascii=False) + "\n\n")

    @staticmethod
    def _accept(record: dict, event: dict, start: float) -> None:
        if isinstance(event.get("usage"), dict):
            record["usage"] = event["usage"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning_content") or ""
            answer = delta.get("content") or ""
            if reasoning or answer:
                if record["first_content_s"] is None:
                    record["first_content_s"] = time.perf_counter() - start
                if isinstance(reasoning, str):
                    record["reasoning_chars"] += len(reasoning)
                if isinstance(answer, str):
                    record["answer_chars"] += len(answer)
                    if answer and record["first_answer_s"] is None:
                        record["first_answer_s"] = time.perf_counter() - start
            if choice.get("finish_reason"):
                record["finish_reason"] = choice["finish_reason"]


def start_proxy(url: str, key: str | None, path: Path, timeout: int) -> TimingServer:
    server = TimingServer(url, key, path, timeout)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
