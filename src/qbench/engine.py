from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import Config, Model, request_body
from .selection import DATASETS, collect_manifest, dataset_args, limit, verify_counts
from .workload import write_workload


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run(alias: str, output: Path) -> Path:
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = output / f"{tag}_{alias}_{uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    return run


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _backend(spec: dict, log_path: Path, secret: str | None = None) -> None:
    """Run EvalScope, keep its full log, and show one compact progress line."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8:replace"
    process = subprocess.Popen(
        [sys.executable, "-m", "qbench.backend"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=0,
        env=env,
    )
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(json.dumps(spec))
    process.stdin.close()

    chunks: list[str] = []
    pending: list[str] = []
    progress_width = 0
    progress_visible = False
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    progress_pattern = re.compile(
        r"(?P<label>[^:\r\n]{1,80}):\s*(?P<percent>\d{1,3})%\|.*?\|\s*(?P<done>\d+)/(?P<total>\d+)"
    )

    def emit(value: str) -> None:
        nonlocal progress_width, progress_visible
        safe = value.replace(secret, "[REDACTED]") if secret else value
        chunks.append(safe)
        clean = ansi.sub("", safe).strip()
        for marker in (" - INFO: ", " - WARNING: ", " - ERROR: "):
            if marker in clean:
                clean = clean.rsplit(marker, 1)[-1]
        match = progress_pattern.search(clean)
        if not match:
            return
        label = match.group("label").strip()
        if label.startswith("Running["):
            return
        if label == "Processing records":
            label = "准备数据"
        elif label == "Generating[requests]":
            label = "生成请求"
        elif label.startswith("Evaluating["):
            label = "效果请求 " + label.removeprefix("Evaluating[").removesuffix("]")
        elif label.startswith("Processing["):
            label = "性能请求"
        elif label.startswith("Warmup["):
            label = "预热请求"
        line = (
            f"[实时进度] {label}：{int(match.group('percent')):3d}% "
            f"({match.group('done')}/{match.group('total')})"
        )
        terminal_encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        visible = line.encode(terminal_encoding, errors="replace").decode(terminal_encoding, errors="replace")
        padding = " " * max(0, progress_width - len(visible))
        print("\r" + visible + padding, end="", flush=True)
        progress_width = len(visible)
        progress_visible = True

    try:
        while True:
            char = process.stdout.read(1)
            if not char:
                break
            pending.append(char)
            if char in "\r\n":
                emit("".join(pending))
                pending.clear()
        if pending:
            emit("".join(pending))
        returncode = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        print("\n[已停止] 用户中断了当前任务。", flush=True)
        raise

    if progress_visible:
        print()

    log = "".join(chunks)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log, encoding="utf-8")
    if secret:
        for path in log_path.parent.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".yaml", ".yml", ".txt", ".log", ".html"}:
                try:
                    content = path.read_text(encoding="utf-8")
                    if secret in content:
                        path.write_text(content.replace(secret, "[REDACTED]"), encoding="utf-8")
                except (UnicodeError, OSError):
                    continue
    if returncode:
        raise RuntimeError(f"EvalScope 失败；查看 {log_path}\n{log[-1500:]}")

def _quality_spec(config: Config, model: Model | None, dataset: str, profile: str,
                  work_dir: Path, *, cache: Path | None = None) -> dict:
    settings = config.quality
    generation = {"max_tokens": int(settings.get("output_budget", 1024)),
                  "temperature": settings.get("temperature", 0),
                  "timeout": float(settings.get("timeout_seconds", 120)),
                  "retries": int(settings.get("retries", 0)), "stream": True}
    for key in ("top_p", "frequency_penalty", "reasoning_effort"):
        if key in settings:
            generation[key] = settings[key]
    if "stop" in settings:
        generation["stop_seqs"] = settings["stop"]
    spec = {"kind": "quality" if model else "prepare", "model": model.model if model else "qbench_prepare",
            "model_id": (model.alias + "_" + model.identity()) if model else "qbench_prepare",
            "eval_type": "openai_api" if model else "mock_llm", "datasets": [dataset],
            "dataset_args": {dataset: dataset_args(dataset, profile, settings.get("dataset_ids"))},
            "dataset_hub": settings.get("dataset_hub", "modelscope"),
            "dataset_dir": str(Path(settings.get("dataset_dir", "data/evalscope")).resolve()),
            "generation_config": generation, "eval_batch_size": int(settings.get("concurrency", 1)),
            "seed": 42, "work_dir": str(work_dir), "no_timestamp": True, "ignore_errors": True}
    cap = limit(dataset, profile)
    if cap is not None:
        spec["limit"] = cap
    if model:
        spec["api_url"] = model.api_root
        spec["api_key_env"] = model.api_key_env
        extra = request_body(model, settings, "probe", stream=True)
        standard = {"model", "messages", "stream", "max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty", "seed", "stop"}
        extra = {k: v for k, v in extra.items() if k not in standard}
        if extra:
            generation["extra_body"] = extra
    if cache:
        spec["use_cache"] = str(cache)
    return spec


def prepare(config: Config, profile: str, datasets: list[str], data_root: Path) -> Path:
    print(f"\n[数据准备] profile={profile}，数据集={', '.join(datasets)}", flush=True)
    identity = {"profile": profile, "hub": config.quality.get("dataset_hub", "modelscope"),
                "dataset_ids": config.quality.get("dataset_ids"), "evalscope": _evalscope_version(),
                "datasets": {name: dataset_args(name, profile, config.quality.get("dataset_ids")) for name in DATASETS}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    root = data_root / "prepared" / profile / fingerprint
    manifest_path = root / "manifest.json"
    existing = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") == identity:
            existing = manifest.get("datasets", {})
        if all(name in existing for name in datasets):
            print(f"[数据准备] 已存在完整缓存，直接复用：{manifest_path}", flush=True)
            return manifest_path
    all_items = dict(existing)
    for index, name in enumerate(datasets, 1):
        if name in all_items:
            print(f"[数据准备 {index}/{len(datasets)}] {name}：已有缓存", flush=True)
            continue
        print(f"[数据准备 {index}/{len(datasets)}] {name}：正在下载或读取数据并固定样本...", flush=True)
        folder = root / f"{name}_{uuid.uuid4().hex[:8]}"
        spec = _quality_spec(config, None, name, profile, folder)
        _backend(spec, folder / "prepare.log")
        all_items[name] = collect_manifest(folder, [name])[name]
        print(f"[数据准备 {index}/{len(datasets)}] {name}：完成，共 {all_items[name]['count']} 个样本", flush=True)
    problems = verify_counts(all_items, profile, datasets)
    if problems:
        raise RuntimeError("抽样数量不一致: " + "; ".join(problems))
    manifest = {"profile": profile, "seed": 42, "datasets": all_items,
                "protocol": {name: dataset_args(name, profile) for name in datasets}, "prepared_at": now(),
                "evalscope_version": _evalscope_version(), "identity": identity}
    save_json(manifest_path, manifest)
    print(f"[数据准备] 全部完成，样本清单：{manifest_path}", flush=True)
    return manifest_path


def _evalscope_version() -> str:
    from importlib.metadata import version
    try:
        return version("evalscope")
    except Exception:
        return "unknown"


def _metadata(config: Config, model: Model, profile: str, run: Path, manifest: Path | None = None) -> dict:
    return {"run_id": run.name, "created_at": now(), "model": model.public_dict(), "profile": profile,
            "qbench_version": __version__, "evalscope_version": _evalscope_version(),
            "quality_settings": config.quality, "performance_settings": config.performance,
            "manifest": str(manifest) if manifest else None, "simulated": model.simulated,
            "config_path": str(config.path)}


def quality(config: Config, model: Model, profile: str, datasets: list[str], run: Path,
            data_root: Path, *, resume: bool = False) -> dict:
    manifest_path = prepare(config, profile, datasets, data_root)
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))["datasets"]
    root = run / "quality"
    outcomes = {}
    print(f"\n[效果测试] 模型={model.alias}，profile={profile}，共 {len(datasets)} 个数据集", flush=True)
    for index, name in enumerate(datasets, 1):
        folder = root / name
        spec = _quality_spec(config, model, name, profile, folder, cache=folder if resume else None)
        save_json(folder / "effective_config.json", {k: v for k, v in spec.items() if k != "api_key_env"})
        print(
            f"[效果测试 {index}/{len(datasets)}] {name}：开始，"
            f"预计 {expected[name]['count']} 题，并发={config.quality.get('concurrency', 1)}",
            flush=True,
        )
        try:
            _backend(spec, folder / "evalscope.log", model.api_key())
            observed = collect_manifest(folder, [name])[name]
            same = observed["sha256"] == expected[name]["sha256"]
            from .report import quality_items
            items = quality_items(run, name)
            with (folder / "questions.jsonl").open("w", encoding="utf-8") as file:
                for item in items:
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")
            valid = same and len(items) == expected[name]["count"] and all(
                x["request_status"] == "success" and x["correct"] is not None for x in items)
            outcomes[name] = {"status": "complete" if valid else "incomplete", "expected": expected[name]["count"],
                              "observed": observed["count"], "sample_hash": observed["sha256"],
                              "reason": None if valid else "样本不一致、请求失败或评分缺失"}
            print(
                f"[效果测试 {index}/{len(datasets)}] {name}：{outcomes[name]['status']}，"
                f"有效结果={len(items)}/{expected[name]['count']}",
                flush=True,
            )
        except Exception as exc:
            outcomes[name] = {"status": "incomplete", "reason": str(exc), "expected": expected[name]["count"]}
            print(f"[效果测试 {index}/{len(datasets)}] {name}：incomplete，{exc}", flush=True)
    save_json(root / "quality_status.json", outcomes)
    return outcomes


def perf(config: Config, model: Model, profile: str, run: Path, *, concurrency: list[int] | None = None,
         requests: int | None = None) -> list[dict]:
    settings = config.performance
    mode = settings.get("mode", "api")
    if mode not in {"api", "fixed_tokens"}:
        raise ValueError("performance.mode 必须是 api 或 fixed_tokens")
    if mode == "fixed_tokens" and model.provider not in {"sglang", "vllm"}:
        raise ValueError("fixed_tokens 只支持已知支持相关参数的本地服务")
    levels = concurrency or [int(x) for x in settings.get("concurrency", [1, 4, 8])]
    n = requests or int(settings.get("requests_per_round", 128))
    rounds = int(settings.get("rounds", 3))
    warmup = int(settings.get("warmup_requests", max(4, max(levels))))
    scenes = len(levels) * rounds
    formal_total = scenes * n
    warmup_total = scenes * warmup
    print(
        f"\n[性能测试] 模型={model.alias}，并发级别={levels}，每轮正式请求={n}，"
        f"轮数={rounds}，每轮预热={warmup}",
        flush=True,
    )
    print(
        f"[性能测试] 共 {scenes} 个场景，预计发送 {formal_total} 个正式请求 + "
        f"{warmup_total} 个预热请求 = {formal_total + warmup_total} 个请求",
        flush=True,
    )
    root = run / "performance"
    workload = root / "workload.txt"
    workload_hash = write_workload(workload, max(128, n))
    extra = request_body(model, settings, "probe", stream=True)
    standard = {"model", "messages", "stream", "max_tokens", "temperature", "top_p", "frequency_penalty", "stop"}
    extra = {k: v for k, v in extra.items() if k not in standard}
    result = []
    scene_index = 0
    for level in levels:
        for round_index in range(1, rounds + 1):
            scene_index += 1
            print(
                f"[性能测试 {scene_index}/{scenes}] 并发={level}，第 {round_index}/{rounds} 轮"
                f"：开始，正式请求={n}，预热请求={warmup}",
                flush=True,
            )
            folder = root / f"c{level}_r{round_index}"
            spec = {"kind": "perf", "model": model.model, "url": model.chat_url, "api": "openai",
                    "api_key_env": model.api_key_env, "parallel": [level], "number": [n],
                    "dataset": "line_by_line", "dataset_path": str(workload), "max_tokens": int(settings.get("output_budget", 256)),
                    "temperature": settings.get("temperature", 0), "stream": True,
                    "warmup_num": warmup,
                    "no_test_connection": True,
                    "total_timeout": int(settings.get("timeout_seconds", 120)),
                    "outputs_dir": str(folder), "no_timestamp": True,
                    "extra_args": extra}
            for key in ("top_p", "frequency_penalty", "stop"):
                if key in settings:
                    spec[key] = settings[key]
            if mode == "fixed_tokens":
                tokenizer = settings.get("tokenizer_path")
                if not tokenizer:
                    raise ValueError("fixed_tokens 需要 tokenizer_path")
                spec.update({"dataset": "random", "dataset_path": None, "tokenizer_path": tokenizer,
                             "min_prompt_length": int(settings["input_tokens"]),
                             "max_prompt_length": int(settings["input_tokens"]),
                             "min_tokens": int(settings["output_tokens"]),
                             "max_tokens": int(settings["output_tokens"]),
                             "prefix_length": 0})
            save_json(folder / "effective_config.json", {k: v for k, v in spec.items() if k != "api_key_env"})
            try:
                _backend(spec, folder / "evalscope.log", model.api_key())
                status = "complete"
                reason = None
            except Exception as exc:
                status = "incomplete"
                reason = str(exc)
            entry = {"concurrency": level, "round": round_index, "requests": n, "mode": mode,
                     "workload_sha256": workload_hash if mode == "api" else None,
                     "output_budget": spec["max_tokens"], "thinking_mode": settings.get("thinking_mode", "default"),
                     "status": status, "reason": reason, "folder": str(folder)}
            result.append(entry)
            save_json(root / "performance_status.json", result)
            print(
                f"[性能测试 {scene_index}/{scenes}] 并发={level}，第 {round_index}/{rounds} 轮：{status}",
                flush=True,
            )
    return result
