from __future__ import annotations

import csv
import html
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .selection import DATASETS, sample_key, read_jsonl


def _average(values: list[float | None]) -> float | None:
    valid = [float(x) for x in values if x is not None]
    return statistics.mean(valid) if valid else None


def _percentile(values: list[float | None], p: float) -> float | None:
    data = sorted(float(x) for x in values if x is not None)
    if not data:
        return None
    index = (len(data) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    return data[low] * (high - index) + data[high] * (index - low) if low != high else data[low]


def _value(score: dict) -> float | None:
    values = score.get("value") or {}
    key = score.get("main_score_name")
    if key in values:
        return float(values[key])
    return float(next(iter(values.values()))) if values else None


def _answer_and_reasoning(content: object) -> tuple[str, str]:
    if isinstance(content, str):
        return content, ""
    if isinstance(content, list):
        answer, reasoning = [], []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "reasoning" or "reasoning" in part:
                    reasoning.append(str(part.get("reasoning") or ""))
                elif "text" in part:
                    answer.append(str(part["text"]))
        return "".join(answer), "".join(reasoning)
    return "", ""


def quality_items(run: Path, dataset: str) -> list[dict]:
    root = run / "quality" / dataset
    items = []
    for file in (root / "predictions").rglob(f"{dataset}*.jsonl"):
        stem = file.stem
        review_files = list((root / "reviews").rglob(stem + ".jsonl"))
        reviews = {row.get("index"): row for path in review_files for row in read_jsonl(path)}
        subject = stem[len(dataset) + 1:] if stem.startswith(dataset + "_") else "all"
        for prediction in read_jsonl(file):
            review = reviews.get(prediction.get("index"), {})
            score = (review.get("sample_score") or {}).get("score") or {}
            output = prediction.get("model_output") or {}
            choice = (output.get("choices") or [{}])[0]
            message = choice.get("message") or output.get("message") or {}
            final_text, reasoning = _answer_and_reasoning(message.get("content"))
            text = score.get("prediction") or final_text
            error = output.get("error") or prediction.get("error")
            items.append({"key": sample_key(prediction, stem), "dataset": dataset, "subject": subject,
                          "answer": text, "reasoning": reasoning or message.get("reasoning_content") or "",
                          "extracted_answer": score.get("extracted_prediction"), "correct_answer": review.get("target"),
                          "correct": _value(score), "request_status": "failed" if error or score.get("status") not in (None, "success") else "success",
                          "error": str(error)[:300] if error else None,
                          "truncated": (choice.get("finish_reason") or choice.get("stop_reason")) == "length"})
    return items


def summarize_quality(run: Path, dataset: str) -> dict:
    status_path = run / "quality" / "quality_status.json"
    statuses = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status = statuses.get(dataset, {})
    items = quality_items(run, dataset)
    not_run = dataset not in statuses and not items
    failed = sum(x["request_status"] == "failed" for x in items)
    unscored = sum(x["correct"] is None for x in items)
    complete = status.get("status") == "complete" and not failed and not unscored and len(items) == status.get("expected")
    groups = defaultdict(list)
    for item in items:
        if item["correct"] is not None:
            groups[item["subject"]].append(item["correct"])
    macro = dataset in {"mmlu", "ceval"} and len(groups) > 1
    accuracy = _average([_average(x) for x in groups.values()]) if macro else _average([x["correct"] for x in items])
    return {"dataset": dataset, "accuracy": accuracy if complete else None, "partial_accuracy": accuracy,
            "aggregation": "subject_macro" if macro else "micro", "samples": len(items),
            "expected": status.get("expected"), "failed": failed, "unscored": unscored,
            "status": "complete" if complete else "not_run" if not_run else "incomplete",
            "sample_hash": status.get("sample_hash"),
            "reason": "未运行" if not_run else status.get("reason"), "items": items}


def _paired_ci(base: list[dict], target: list[dict], macro: bool) -> tuple[float, float] | None:
    left = {x["key"]: x for x in base}
    right = {x["key"]: x for x in target}
    if left.keys() != right.keys() or not left:
        return None
    pairs = [(left[k]["subject"], left[k]["correct"], right[k]["correct"]) for k in sorted(left)]
    if any(a is None or b is None for _, a, b in pairs):
        return None
    by_subject = defaultdict(list)
    for subject, a, b in pairs:
        by_subject[subject].append((a, b))
    rng = random.Random(42)
    differences = []
    for _ in range(1000):
        if macro:
            deltas = []
            for group in by_subject.values():
                drawn = rng.choices(group, k=len(group))
                deltas.append(statistics.mean(b - a for a, b in drawn))
            differences.append(statistics.mean(deltas) * 100)
        else:
            drawn = rng.choices(pairs, k=len(pairs))
            differences.append(statistics.mean(b - a for _, a, b in drawn) * 100)
    return (_percentile(differences, .025), _percentile(differences, .975))


def perf_round(entry: dict, settings: dict) -> dict:
    folder = Path(entry["folder"])
    file = folder / "requests.jsonl"
    all_rows = read_jsonl(file) if file.exists() else []
    warmup = int(settings.get("warmup_requests", max(4, entry["concurrency"])))
    rows = sorted(all_rows, key=lambda x: x["id"])[warmup:]
    success = [x for x in rows if x.get("http_status") == 200 and not x.get("error")]
    duration = max((x["end"] for x in rows), default=0) - min((x["start"] for x in rows), default=0)
    tokens = [x.get("usage", {}).get("completion_tokens") if x.get("usage") else None for x in success]
    known = len(success) > 0 and all(isinstance(x, int) and x >= 0 for x in tokens)
    total_tokens = sum(tokens) if known else None
    tpot = [(x["elapsed_s"] - x["first_content_s"]) / max(n - 1, 1)
            if n and x.get("first_content_s") is not None else None
            for x, n in zip(success, tokens)]
    reasons = defaultdict(int)
    for x in rows:
        if x not in success:
            reasons[x.get("error") or str(x.get("http_status"))] += 1
    return {**entry, "actual_requests": len(rows), "success": len(success), "failed": len(rows) - len(success),
            "failure_reasons": dict(reasons), "window_s": duration or None,
            "success_rate": len(success) / len(rows) if rows else None,
            "request_throughput": len(success) / duration if duration > 0 else None,
            "output_tokens_total": total_tokens,
            "output_token_throughput": total_tokens / duration if known and duration > 0 else None,
            "output_tokens_avg": _average(tokens) if known else None,
            "ttft_mean": _average([x.get("first_content_s") for x in success]),
            "ttft_p50": _percentile([x.get("first_content_s") for x in success], .5),
            "ttft_p95": _percentile([x.get("first_content_s") for x in success], .95),
            "first_answer_mean": _average([x.get("first_answer_s") for x in success]),
            "first_answer_p50": _percentile([x.get("first_answer_s") for x in success], .5),
            "first_answer_p95": _percentile([x.get("first_answer_s") for x in success], .95),
            "tpot_mean": _average(tpot) if known else None,
            "tpot_p50": _percentile(tpot, .5) if known else None,
            "tpot_p95": _percentile(tpot, .95) if known else None,
            "latency_mean": _average([x.get("elapsed_s") for x in success]),
            "latency_p50": _percentile([x.get("elapsed_s") for x in success], .5),
            "latency_p95": _percentile([x.get("elapsed_s") for x in success], .95),
            "status": "complete" if entry["status"] == "complete" and len(rows) == entry["requests"] else "incomplete"}


def _csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "N/A" if row.get(key) is None else row[key] for key in fields})


def _fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def summarize(input_dir: Path, baseline: str, output: Path, *, simulation_only: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for file in input_dir.glob("*/run.json"):
        try:
            meta = json.loads(file.read_text(encoding="utf-8"))
            if bool(meta.get("simulated")) == simulation_only:
                runs.append((file.parent, meta))
        except (ValueError, KeyError):
            continue
    runs.sort(key=lambda x: x[1].get("finished_at", x[1].get("created_at", "")))
    latest = {meta["model"]["alias"]: (run, meta) for run, meta in runs}
    quality_rows = []
    perf_rows = []
    details = {}
    for alias, (run, meta) in latest.items():
        q = {name: summarize_quality(run, name) for name in DATASETS}
        details[alias] = {"run_id": meta["run_id"], "model": meta["model"], "profile": meta["profile"],
                          "quality": {name: {k: v for k, v in result.items() if k != "items"} for name, result in q.items()}}
        for name, result in q.items():
            quality_rows.append({"model": alias, "run_id": meta["run_id"], "profile": meta["profile"],
                                 "simulated": bool(meta.get("simulated")),
                                 "dataset": name, "comparison_type": "reference" if alias == baseline else None,
                                 **{k: v for k, v in result.items() if k != "items"}})
        status_path = run / "performance" / "performance_status.json"
        if status_path.exists():
            entries = json.loads(status_path.read_text(encoding="utf-8"))
            for entry in entries:
                perf_rows.append({"model": alias, "run_id": meta["run_id"], "profile": meta["profile"],
                                  "simulated": bool(meta.get("simulated")),
                                  **perf_round(entry, meta.get("performance_settings") or {})})
    base = latest.get(baseline)
    for row in quality_rows:
        row["delta_pp"] = None
        row["paired_ci_low_pp"] = None
        row["paired_ci_high_pp"] = None
        row["comparison_reason"] = None
        row["controlled_comparison"] = None
        if base and row["model"] != baseline:
            target_run, target_meta = latest[row["model"]]
            base_run, base_meta = base
            row["comparison_type"] = "model_comparison"
            target = summarize_quality(target_run, row["dataset"])
            source = summarize_quality(base_run, row["dataset"])
            reasons = []
            if target_meta["profile"] != base_meta["profile"]:
                reasons.append("profile 不同")
            if target["sample_hash"] != source["sample_hash"]:
                reasons.append("样本不同")
            if target["status"] != "complete" or source["status"] != "complete":
                reasons.append("测试未完成")
            for key in ("output_budget", "temperature", "thinking_mode"):
                if target_meta.get("quality_settings", {}).get(key) != base_meta.get("quality_settings", {}).get(key):
                    reasons.append(f"{key} 不同")
            if reasons:
                row["comparison_reason"] = "; ".join(reasons)
                row["controlled_comparison"] = False
            else:
                row["controlled_comparison"] = True
                row["delta_pp"] = (target["accuracy"] - source["accuracy"]) * 100
                ci = _paired_ci(source["items"], target["items"], target["aggregation"] == "subject_macro")
                if ci:
                    row["paired_ci_low_pp"], row["paired_ci_high_pp"] = ci
    grouped = defaultdict(list)
    for row in perf_rows:
        grouped[(row["model"], row["mode"], row["concurrency"])].append(row)
    for group in grouped.values():
        tps = [x["output_token_throughput"] for x in group if x["output_token_throughput"] is not None]
        variation = statistics.stdev(tps) / statistics.mean(tps) if len(tps) > 1 and statistics.mean(tps) else None
        for row in group:
            row["round_cv"] = variation
            row["comparison_type"] = "reference" if row["model"] == baseline else "model_comparison"
            row["comparison_reason"] = None
            row["controlled_comparison"] = None
            if base and row["model"] != baseline:
                candidate = next((x for x in perf_rows if x["model"] == baseline and x["mode"] == row["mode"] and
                                  x["concurrency"] == row["concurrency"] and x["round"] == row["round"]), None)
                reasons = []
                if candidate is None:
                    reasons.append("参考模型缺少同场景")
                else:
                    for key in ("workload_sha256", "output_budget", "thinking_mode", "requests"):
                        if row.get(key) != candidate.get(key):
                            reasons.append(f"{key} 不同")
                    if row["status"] != "complete" or candidate["status"] != "complete":
                        reasons.append("测试未完成")
                model = latest[row["model"]][1]["model"]
                reference = base[1]["model"]
                if model.get("hardware") == "unknown" or reference.get("hardware") == "unknown":
                    reasons.append("硬件信息缺失")
                elif model.get("hardware") != reference.get("hardware"):
                    reasons.append("硬件不同")
                if model.get("server_parameters") != reference.get("server_parameters"):
                    reasons.append("服务参数不同")
                row["controlled_comparison"] = not reasons
                row["comparison_reason"] = "; ".join(reasons) if reasons else None
    q_fields = ["model", "run_id", "profile", "simulated", "dataset", "accuracy", "delta_pp", "paired_ci_low_pp", "paired_ci_high_pp",
                "samples", "expected", "failed", "unscored", "aggregation", "status", "comparison_type", "controlled_comparison", "comparison_reason", "sample_hash"]
    p_fields = ["model", "run_id", "profile", "simulated", "mode", "concurrency", "round", "requests", "actual_requests", "success", "failed",
                "success_rate", "window_s", "request_throughput", "output_token_throughput", "output_tokens_avg",
                "ttft_mean", "ttft_p50", "ttft_p95", "first_answer_mean", "first_answer_p50", "first_answer_p95",
                "tpot_mean", "tpot_p50", "tpot_p95", "latency_mean", "latency_p50", "latency_p95",
                "round_cv", "thinking_mode", "output_budget", "status", "comparison_type", "comparison_reason",
                "controlled_comparison", "workload_sha256", "failure_reasons"]
    _csv(output / "quality_summary.csv", quality_rows, q_fields)
    _csv(output / "performance_summary.csv", perf_rows, p_fields)
    comparison = {"reference_model": baseline, "simulation_only": simulation_only, "models": details, "quality": quality_rows, "performance": perf_rows,
                  "notes": ["所有准确率为 0-1；delta_pp 与配对区间单位为百分点。",
                            "任意两个模型或服务都可以对比；实验条件不同时，结果表示端到端体验差异，不能只归因于模型本身。",
                            "通用 API 模式按实际输出长度统计；token 指标缺失为 N/A。"]}
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report_scope = "模拟运行；结果仅用于验证流程，不代表真实模型性能或能力。" if simulation_only else "真实服务运行。"
    lines = ["# 大模型能力与服务性能报告", "", f"参考模型：{baseline}。{report_scope}仅使用已保存结果。", "",
             "## 能力", "", "|模型|数据集|准确率|差值(pp)|95%配对区间(pp)|样本|状态|", "|---|---|---:|---:|---|---:|---|"]
    for row in quality_rows:
        ci = f"{_fmt(row['paired_ci_low_pp'])} 至 {_fmt(row['paired_ci_high_pp'])}"
        lines.append(f"|{row['model']}|{row['dataset']}|{_fmt(row['accuracy'])}|{_fmt(row['delta_pp'])}|{ci}|{row['samples']}|{row['status']}|")
    lines += ["", "## 性能（每轮）", "", "|模型|模式|并发|轮次|输出 token/s|TTFT P95(s)|TPOT P95(s)|延迟 P95(s)|成功率|",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in perf_rows:
        lines.append(f"|{row['model']}|{row['mode']}|{row['concurrency']}|{row['round']}|{_fmt(row['output_token_throughput'])}|{_fmt(row['ttft_p95'])}|{_fmt(row['tpot_p95'])}|{_fmt(row['latency_p95'])}|{_fmt(row['success_rate'])}|")
    lines += ["", "## 解释", "", *["- " + note for note in comparison["notes"]], ""]
    md = "\n".join(lines)
    (output / "report.md").write_text(md, encoding="utf-8")
    # Self-contained HTML, no remote CSS or scripts.
    cells = lambda values: "".join("<td>" + html.escape(_fmt(x)) + "</td>" for x in values)
    def table(fields: list[str], rows: list[dict]) -> str:
        head = "".join("<th>" + html.escape(x) + "</th>" for x in fields)
        body = "".join("<tr>" + cells([row.get(x) for x in fields]) + "</tr>" for row in rows)
        return "<div class='scroll'><table><thead><tr>" + head + "</tr></thead><tbody>" + body + "</tbody></table></div>"
    page = "<!doctype html><html lang='zh'><meta charset='utf-8'><title>qbench 报告</title><style>body{font:15px system-ui;margin:2rem;color:#17212b}h1,h2{color:#123b55}.scroll{overflow:auto}table{border-collapse:collapse;margin-bottom:2rem}td,th{padding:.55rem;border:1px solid #ccd7df;white-space:nowrap}th{background:#e9f2f7}tr:nth-child(even){background:#f8fbfc}</style><h1>大模型能力与服务性能报告</h1><p>参考模型：" + html.escape(baseline) + "。" + html.escape(report_scope) + "仅使用已保存结果。</p><h2>能力</h2>" + table(q_fields, quality_rows) + "<h2>性能</h2>" + table(p_fields, perf_rows) + "<p>任意两个模型或服务都可以对比；实验条件不同时，结果表示端到端体验差异，不能只归因于模型本身。</p></html>"
    (output / "report.html").write_text(page, encoding="utf-8")
    return comparison
