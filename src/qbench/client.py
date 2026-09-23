from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import httpx

from .config import Model, request_body


@dataclass
class Reply:
    content: str = ""
    reasoning: str = ""
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None
    first_content_s: float | None = None
    first_answer_s: float | None = None
    elapsed_s: float | None = None
    error: str | None = None


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(x.get("text", "")) for x in value if isinstance(x, dict))
    return ""


def parse_nonstream(data: dict) -> Reply:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return Reply(content=_text(message.get("content")), reasoning=_text(message.get("reasoning_content")),
                 usage=data.get("usage") or {}, finish_reason=choice.get("finish_reason"))


def accept_chunk(reply: Reply, data: dict, elapsed: float) -> None:
    usage = data.get("usage")
    if isinstance(usage, dict):
        reply.usage = usage
    for choice in data.get("choices") or []:
        delta = choice.get("delta") or {}
        reasoning = _text(delta.get("reasoning_content"))
        content = _text(delta.get("content"))
        if reasoning or content:
            if reply.first_content_s is None:
                reply.first_content_s = elapsed
            reply.reasoning += reasoning
            reply.content += content
            if content and reply.first_answer_s is None:
                reply.first_answer_s = elapsed
        if choice.get("finish_reason"):
            reply.finish_reason = choice["finish_reason"]


def parse_sse_lines(lines: list[str]) -> Reply:
    reply = Reply()
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        accept_chunk(reply, json.loads(payload), 0.0)
    return reply


def _headers(model: Model) -> dict:
    key = model.api_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


def _safe_error(exc: Exception, model: Model) -> str:
    message = str(exc)
    key = model.api_key() if model.api_key_env else None
    if key:
        message = message.replace(key, "[REDACTED]")
    return message[:500]


def complete(model: Model, settings: dict, prompt: str, *, stream: bool) -> Reply:
    body = request_body(model, settings, prompt, stream=stream)
    if stream:
        body["stream_options"] = {"include_usage": True}
    timeout = float(settings.get("timeout_seconds", 120))
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout, headers=_headers(model)) as client:
            if not stream:
                response = client.post(model.chat_url, json=body)
                response.raise_for_status()
                result = parse_nonstream(response.json())
            else:
                result = Reply()
                with client.stream("POST", model.chat_url, json=body) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        accept_chunk(result, json.loads(payload), time.perf_counter() - started)
        result.elapsed_s = time.perf_counter() - started
        return result
    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
        return Reply(elapsed_s=time.perf_counter() - started, error=_safe_error(exc, model))


def doctor(model: Model, settings: dict) -> dict:
    result = {"model": model.alias, "provider": model.provider, "url": model.chat_url,
              "models_endpoint": None, "model_listed": None}
    with httpx.Client(timeout=float(settings.get("timeout_seconds", 120)), headers=_headers(model)) as client:
        try:
            response = client.get(model.api_root + "/models")
            result["models_endpoint"] = response.status_code
            if response.is_success:
                ids = [x.get("id") for x in response.json().get("data", []) if isinstance(x, dict)]
                result["model_listed"] = model.model in ids
        except (httpx.HTTPError, ValueError) as exc:
            result["models_endpoint_error"] = _safe_error(exc, model)
    probe = {**settings, "output_budget": min(16, int(settings.get("output_budget", 256)))}
    normal = complete(model, probe, "Reply with OK.", stream=False)
    streamed = complete(model, probe, "Reply with OK.", stream=True)
    result["normal"] = {"ok": normal.error is None, "error": normal.error,
                        "has_final_answer": bool(normal.content), "has_reasoning": bool(normal.reasoning),
                        "usage": normal.usage}
    result["stream"] = {"ok": streamed.error is None, "error": streamed.error,
                        "has_final_answer": bool(streamed.content), "has_reasoning": bool(streamed.reasoning),
                        "usage": streamed.usage, "first_content_s": streamed.first_content_s,
                        "first_answer_s": streamed.first_answer_s}
    result["ok"] = normal.error is None and streamed.error is None
    return result
