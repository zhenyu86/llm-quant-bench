"""Small child-process bridge to EvalScope's documented Python APIs.

The parent sends only public settings. API keys are read from the environment here,
never put in a generated config, command line, or report.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    spec = json.load(sys.stdin)
    kind = spec.pop("kind")
    key_env = spec.pop("api_key_env", None)
    key = os.environ.get(key_env) if key_env else None
    if key_env and not key:
        raise ValueError(f"缺少环境变量 {key_env}")
    if kind in {"quality", "prepare"}:
        from evalscope import TaskConfig, run_task

        params = dict(spec)
        if kind == "quality":
            params["api_key"] = key or "EMPTY"
        else:
            params.pop("api_url", None)
            params["eval_type"] = "mock_llm"
        run_task(TaskConfig(**params))
    elif kind == "perf":
        from evalscope.perf.arguments import Arguments
        from evalscope.perf.main import run_perf_benchmark
        from .proxy import start_proxy

        output = Path(spec["outputs_dir"])
        proxy = start_proxy(spec["url"], key, output / "requests.jsonl", spec.get("total_timeout", 120))
        try:
            spec["url"] = f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions"
            spec["api_key"] = "EMPTY"
            result = run_perf_benchmark(Arguments(**spec))
            output.mkdir(parents=True, exist_ok=True)
            output.joinpath("return_value.json").write_text(
                json.dumps({k: {"metrics": v["metrics"].to_dict(),
                                "percentiles": v["percentiles"].to_list()}
                            for k, v in result.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            proxy.shutdown()
            proxy.server_close()
    else:
        raise ValueError(f"未知后端任务 {kind}")


if __name__ == "__main__":
    main()
