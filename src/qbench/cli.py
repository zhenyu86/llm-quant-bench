from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .client import doctor
from .config import load_config
from .engine import _metadata, new_run, now, perf, prepare, quality, save_json
from .selection import DATASETS


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python main.py", description="大模型效果与性能对比工具")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("doctor", "prepare", "quality", "perf", "run", "summarize"):
        command = sub.add_parser(name)
        if name != "summarize":
            default_config = "configs/models.local.yaml" if Path("configs/models.local.yaml").exists() else "configs/models.yaml"
            command.add_argument("--config", default=default_config)
        if name in {"doctor", "quality", "perf", "run"}:
            command.add_argument("--model", action="append", required=True)
        if name in {"prepare", "quality", "perf", "run"}:
            command.add_argument("--profile", choices=("smoke", "quick", "full"), default="quick")
        if name in {"prepare", "quality", "run"}:
            command.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
        if name in {"perf", "run"}:
            command.add_argument("--concurrency", nargs="+", type=int, metavar="N",
                                 help="要测试的并发级别；每个数表示最多同时进行多少个请求")
            command.add_argument("--requests", type=int, metavar="N",
                                 help="每个并发级别、每一轮发送的正式请求总数")
        if name in {"quality", "run"}:
            command.add_argument("--resume", type=Path)
        if name in {"quality", "perf", "run"}:
            command.add_argument("--output", type=Path, default=Path("outputs"))
        if name == "prepare":
            command.add_argument("--data", type=Path, default=Path("data"))
        if name == "summarize":
            command.add_argument("--input", type=Path, default=Path("outputs"))
            command.add_argument("--reference", "--baseline", dest="baseline", required=True,
                                 metavar="MODEL", help="作为对照的模型配置名称；--baseline 是兼容旧版本的写法")
            command.add_argument("--output", type=Path, default=Path("reports"))
            command.add_argument("--simulation-only", action="store_true", help="仅汇总标记 simulated 的模拟运行")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "summarize":
            from .report import summarize
            summarize(args.input, args.baseline, args.output, simulation_only=args.simulation_only)
            print(f"报告已写入 {args.output.resolve()}")
            return 0
        config = load_config(args.config)
        if args.command == "doctor":
            failed = False
            for alias in args.model:
                result = doctor(config.model(alias), {**config.performance, **config.quality})
                print(json.dumps(result, ensure_ascii=False, indent=2))
                failed |= not result["ok"]
            return 1 if failed else 0
        if args.command == "prepare":
            print(prepare(config, args.profile, args.datasets, args.data))
            return 0
        if getattr(args, "concurrency", None) and any(x < 1 for x in args.concurrency):
            raise ValueError("并发必须 >= 1")
        if getattr(args, "requests", None) is not None and args.requests < 1:
            raise ValueError("请求数必须 >= 1")
        if getattr(args, "resume", None) and (args.command != "quality" or len(args.model) != 1):
            raise ValueError("--resume 仅可用于单个模型的 quality")
        incomplete = False
        for alias in args.model:
            model = config.model(alias)
            run = args.resume.resolve() if getattr(args, "resume", None) else new_run(alias, args.output)
            if getattr(args, "resume", None):
                meta_path = run / "run.json"
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta["model"]["identity"] != model.identity() or meta["profile"] != args.profile:
                    raise ValueError("续跑的模型身份或 profile 不一致")
            else:
                meta = _metadata(config, model, args.profile, run)
                save_json(run / "run.json", meta)
            if args.command in {"quality", "run"}:
                outcome = quality(config, model, args.profile, args.datasets, run, Path("data"), resume=bool(getattr(args, "resume", None)))
                print(json.dumps({"run": str(run), "quality": outcome}, ensure_ascii=False, indent=2))
                incomplete |= any(item.get("status") != "complete" for item in outcome.values())
            if args.command in {"perf", "run"}:
                outcome = perf(config, model, args.profile, run, concurrency=args.concurrency, requests=args.requests)
                print(json.dumps({"run": str(run), "performance": outcome}, ensure_ascii=False, indent=2))
                incomplete |= any(item.get("status") != "complete" for item in outcome)
            meta["finished_at"] = now()
            save_json(run / "run.json", meta)
        return 1 if incomplete else 0
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
