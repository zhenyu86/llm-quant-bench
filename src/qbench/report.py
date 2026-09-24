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
    targets = [alias for alias in latest if alias != baseline]
    quality_by_key = {(row["model"], row["dataset"]): row for row in quality_rows}
    quality_view = []
    for target_alias in targets:
        for dataset in DATASETS:
            reference = quality_by_key.get((baseline, dataset))
            target = quality_by_key.get((target_alias, dataset))
            if not reference or not target:
                continue
            issues = []
            if reference["status"] != "complete":
                issues.append("参考模型" + reference["status"])
            if target["status"] != "complete":
                issues.append("对比模型" + target["status"])
            if target.get("comparison_reason"):
                for reason in target["comparison_reason"].split("; "):
                    if reason != "测试未完成" and reason not in issues:
                        issues.append(reason)
            ci = None
            if target.get("paired_ci_low_pp") is not None and target.get("paired_ci_high_pp") is not None:
                ci = f"{_fmt(target['paired_ci_low_pp'])} 至 {_fmt(target['paired_ci_high_pp'])}"
            quality_view.append({
                "数据集": dataset,
                "参考模型": baseline,
                "参考准确率(%)": reference["accuracy"] * 100 if reference["accuracy"] is not None else None,
                "对比模型": target_alias,
                "对比准确率(%)": target["accuracy"] * 100 if target["accuracy"] is not None else None,
                "差值(百分点)": target.get("delta_pp"),
                "95%置信区间(百分点)": ci,
                "题数(参考/对比)": f"{reference['samples']}/{target['samples']}",
                "状态": "完整" if not issues else "无法完整比较",
                "说明": "；".join(issues) if issues else "可直接比较",
            })

    def average_metric(rows: list[dict], key: str) -> float | None:
        return _average([row.get(key) for row in rows])

    perf_summary = {}
    for (model_alias, mode, concurrency_level), rows in grouped.items():
        reasons = []
        for row in rows:
            if row.get("comparison_reason"):
                for reason in row["comparison_reason"].split("; "):
                    if reason and reason not in reasons:
                        reasons.append(reason)
        perf_summary[(model_alias, mode, concurrency_level)] = {
            "rounds": len(rows),
            "output_token_throughput": average_metric(rows, "output_token_throughput"),
            "ttft_p95": average_metric(rows, "ttft_p95"),
            "latency_p95": average_metric(rows, "latency_p95"),
            "success_rate": average_metric(rows, "success_rate"),
            "complete": all(row["status"] == "complete" for row in rows),
            "reasons": reasons,
        }

    def metric_pair(reference: float | None, target: float | None, *, scale: float = 1,
                    unit: str = "", show_change: bool = True) -> str:
        if reference is None or target is None:
            return "N/A"
        left, right = reference * scale, target * scale
        change = (target - reference) / reference * 100 if reference else None
        result = f"{_fmt(left)}{unit} → {_fmt(right)}{unit}"
        if show_change and change is not None:
            result += f" ({change:+.1f}%)"
        return result

    performance_view = []
    for target_alias in targets:
        target_keys = sorted(
            ((mode, level) for model_alias, mode, level in perf_summary if model_alias == target_alias),
            key=lambda item: (item[0], item[1]),
        )
        for mode, level in target_keys:
            reference = perf_summary.get((baseline, mode, level))
            target = perf_summary[(target_alias, mode, level)]
            issues = list(target["reasons"])
            if reference is None:
                issues.insert(0, "参考模型缺少同并发场景")
            else:
                if not reference["complete"]:
                    issues.insert(0, "参考模型测试未完成")
                if not target["complete"]:
                    issues.insert(0, "对比模型测试未完成")
            performance_view.append({
                "模式": mode,
                "并发": level,
                "轮数": target["rounds"],
                "参考模型": baseline,
                "对比模型": target_alias,
                "输出速度 token/s": metric_pair(
                    reference["output_token_throughput"] if reference else None,
                    target["output_token_throughput"],
                ),
                "首Token时间P95(s)": metric_pair(
                    reference["ttft_p95"] if reference else None,
                    target["ttft_p95"],
                ),
                "总延迟P95(s)": metric_pair(
                    reference["latency_p95"] if reference else None,
                    target["latency_p95"],
                ),
                "成功率": metric_pair(
                    reference["success_rate"] if reference else None,
                    target["success_rate"],
                    scale=100,
                    unit="%",
                    show_change=False,
                ),
                "说明": "；".join(dict.fromkeys(issues)) if issues else "可直接比较",
            })

    q_fields = ["数据集", "参考模型", "参考准确率(%)", "对比模型", "对比准确率(%)", "差值(百分点)",
                "95%置信区间(百分点)", "题数(参考/对比)", "状态", "说明"]
    p_fields = ["模式", "并发", "轮数", "参考模型", "对比模型", "输出速度 token/s",
                "首Token时间P95(s)", "总延迟P95(s)", "成功率", "说明"]
    _csv(output / "quality_summary.csv", quality_view, q_fields)
    _csv(output / "performance_summary.csv", performance_view, p_fields)
    comparison = {
        "reference_model": baseline,
        "simulation_only": simulation_only,
        "models": details,
        "quality": quality_rows,
        "performance": perf_rows,
        "quality_comparison": quality_view,
        "performance_comparison": performance_view,
        "notes": [
            "能力表准确率使用百分比，差值与置信区间使用百分点。",
            "性能表按同一并发下的多轮结果取平均；箭头左侧是参考模型，右侧是对比模型。",
            "性能指标括号内是对比模型相对参考模型的变化；输出速度越高越好，时间越低越好。",
            "完整内部字段仍保存在 comparison.json，简表只展示判断模型差异所需的主要指标。",
        ],
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    report_scope = "模拟运行；结果仅用于验证流程，不代表真实模型性能或能力。" if simulation_only else "真实服务运行。"

    def markdown_table(fields: list[str], rows: list[dict]) -> list[str]:
        result = ["|" + "|".join(fields) + "|", "|" + "|".join("---" for _ in fields) + "|"]
        result.extend("|" + "|".join(_fmt(row.get(field)) for field in fields) + "|" for row in rows)
        return result

    lines = [
        "# 大模型能力与服务性能报告",
        "",
        f"参考模型：{baseline}。{report_scope}仅使用已保存结果。",
        "",
        "## 能力对比",
        "",
        *markdown_table(q_fields, quality_view),
        "",
        "## 性能对比（同一并发的多轮平均）",
        "",
        *markdown_table(p_fields, performance_view),
        "",
        "## 怎么看",
        "",
        *["- " + note for note in comparison["notes"]],
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")

    cells = lambda values: "".join("<td>" + html.escape(_fmt(x)) + "</td>" for x in values)

    def table(fields: list[str], rows: list[dict]) -> str:
        head = "".join("<th>" + html.escape(x) + "</th>" for x in fields)
        body = "".join("<tr>" + cells([row.get(x) for x in fields]) + "</tr>" for row in rows)
        if not body:
            body = f"<tr><td colspan='{len(fields)}'>没有可比较的数据</td></tr>"
        return "<div class='scroll'><table><thead><tr>" + head + "</tr></thead><tbody>" + body + "</tbody></table></div>"

    notes_html = "".join("<li>" + html.escape(note) + "</li>" for note in comparison["notes"])
    page = (
        "<!doctype html><html lang='zh'><meta charset='utf-8'><title>模型对比报告</title>"
        "<style>body{font:15px system-ui;margin:2rem;color:#17212b}h1,h2{color:#123b55}"
        ".scroll{overflow:auto}table{border-collapse:collapse;margin-bottom:2rem;width:100%}"
        "td,th{padding:.55rem;border:1px solid #ccd7df;white-space:nowrap;text-align:left}"
        "th{background:#e9f2f7}tr:nth-child(even){background:#f8fbfc}</style>"
        "<h1>大模型能力与服务性能报告</h1><p>参考模型：" + html.escape(baseline) + "。"
        + html.escape(report_scope) + "仅使用已保存结果。</p><h2>能力对比</h2>"
        + table(q_fields, quality_view)
        + "<h2>性能对比（同一并发的多轮平均）</h2>"
        + table(p_fields, performance_view)
        + "<h2>怎么看</h2><ul>" + notes_html + "</ul></html>"
    )
    (output / "report.html").write_text(page, encoding="utf-8")
    return comparison
