from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import yaml


PROVIDERS = {"sglang", "vllm", "deepseek", "openai_compatible"}
LOCAL_ONLY = {"ignore_eos", "min_tokens", "top_k", "stop_token_ids"}
STANDARD_GENERATION = {"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "stop"}


def endpoint(base_url: str) -> tuple[str, str]:
    """Return (API root, Chat Completions URL), accepting all three common forms."""
    parts = urlsplit(base_url.rstrip("/"))
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.query or parts.fragment:
        raise ValueError(f"无效 base_url: {base_url}")
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
    if not path.endswith("/v1"):
        path += "/v1"
    root = urlunsplit((parts.scheme, parts.netloc, path, "", ""))
    return root, root + "/chat/completions"


@dataclass(frozen=True)
class Model:
    alias: str
    provider: str
    base_url: str
    model: str
    api_key_env: str | None = None
    base_model: str | None = None
    variant: str | None = None
    hardware: str = "unknown"
    simulated: bool = False
    server_parameters: dict = field(default_factory=dict)
    extra_body: dict = field(default_factory=dict)

    @property
    def api_root(self) -> str:
        return endpoint(self.base_url)[0]

    @property
    def chat_url(self) -> str:
        return endpoint(self.base_url)[1]

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ValueError(f"缺少环境变量 {self.api_key_env}")
        return key

    def validated_extra(self) -> dict:
        if not isinstance(self.extra_body, dict):
            raise ValueError(f"{self.alias}: extra_body 必须是对象")
        bad = LOCAL_ONLY & self.extra_body.keys()
        if bad and self.provider not in {"sglang", "vllm"}:
            raise ValueError(f"{self.alias}: {sorted(bad)} 只允许本地服务使用")
        if self.provider == "deepseek" and "thinking" in self.extra_body:
            raise ValueError("DeepSeek thinking 使用 thinking_mode 配置，不能重复写在 extra_body")
        return dict(self.extra_body)

    def identity(self) -> str:
        data = {"alias": self.alias, "provider": self.provider, "url": self.api_root, "model": self.model,
                "variant": self.variant, "base_model": self.base_model}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

    def public_dict(self) -> dict:
        return {"alias": self.alias, "provider": self.provider, "base_url": self.api_root,
                "model": self.model, "api_key_env": self.api_key_env,
                "base_model": self.base_model, "variant": self.variant, "hardware": self.hardware,
                "server_parameters": self.server_parameters, "extra_body": self.validated_extra(),
                "simulated": self.simulated,
                "identity": self.identity()}


@dataclass(frozen=True)
class Config:
    models: dict[str, Model]
    quality: dict
    performance: dict
    path: Path

    def model(self, alias: str) -> Model:
        if alias not in self.models:
            raise ValueError(f"未知模型 {alias}; 可选: {', '.join(self.models)}")
        return self.models[alias]


def load_config(path: str | Path = "configs/models.yaml") -> Config:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("models")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("models 必须是非空映射")
    models = {}
    for alias, item in raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"{alias}: 模型配置必须是映射")
        if item.get("provider") not in PROVIDERS:
            raise ValueError(f"{alias}: provider 必须是 {sorted(PROVIDERS)}")
        for key in ("base_url", "model"):
            if not item.get(key) or not isinstance(item[key], str):
                raise ValueError(f"{alias}: 缺少 {key}")
        accepted = set(Model.__dataclass_fields__) - {"alias"}
        unknown = set(item) - accepted
        if unknown:
            raise ValueError(f"{alias}: 未知参数 {sorted(unknown)}")
        model = Model(alias=alias, **item)
        model.api_root
        model.validated_extra()
        models[alias] = model
    quality = data.get("quality") or {}
    performance = data.get("performance") or {}
    if not isinstance(quality, dict) or not isinstance(performance, dict):
        raise ValueError("quality/performance 必须是映射")
    allowed_quality = {"dataset_hub", "dataset_dir", "dataset_ids", "output_budget", "temperature",
                       "timeout_seconds", "concurrency", "thinking_mode", "retries", "top_p",
                       "frequency_penalty", "reasoning_effort", "stop"}
    allowed_performance = {"mode", "concurrency", "requests_per_round", "rounds", "warmup_requests",
                           "output_budget", "timeout_seconds", "temperature", "thinking_mode",
                           "tokenizer_path", "input_tokens", "output_tokens", "top_p", "frequency_penalty", "stop"}
    for name, settings, allowed in (("quality", quality, allowed_quality), ("performance", performance, allowed_performance)):
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(f"{name}: 不支持的参数 {sorted(unknown)}")
        if settings.get("thinking_mode", "default") not in {"default", "on", "off"}:
            raise ValueError(f'{name}.thinking_mode 必须是带引号的 "default"、"on" 或 "off"')
    return Config(models, quality, performance, path)


def request_body(model: Model, settings: dict, prompt: str, *, stream: bool) -> dict:
    mode = settings.get("thinking_mode", "default")
    if mode not in {"default", "on", "off"}:
        raise ValueError("thinking_mode 必须为 default/on/off")
    body = {"model": model.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(settings.get("output_budget", 256)),
            "temperature": settings.get("temperature", 0), "stream": stream}
    for key in STANDARD_GENERATION - {"temperature", "max_tokens"}:
        if key in settings:
            body[key] = settings[key]
    extra = model.validated_extra()
    if mode != "default":
        if model.provider == "deepseek":
            extra["thinking"] = {"type": "enabled" if mode == "on" else "disabled"}
        elif model.provider == "sglang":
            extra["chat_template_kwargs"] = {"enable_thinking": mode == "on"}
        elif model.provider == "vllm":
            extra["chat_template_kwargs"] = {"enable_thinking": mode == "on"}
        else:
            raise ValueError("openai_compatible 的思考开关需通过该服务自己的 extra_body 显式设置")
    collision = set(extra) & set(body)
    if collision:
        raise ValueError(f"extra_body 与标准字段冲突: {sorted(collision)}")
    body.update(extra)
    return body
