import json
from pathlib import Path

import pytest

import qbench.cli as cli
from qbench.client import accept_chunk, parse_nonstream, parse_sse_lines, Reply
from qbench.config import Model, endpoint, load_config, request_body
from qbench.report import _paired_ci, perf_round, summarize
from qbench.selection import collect_manifest, dataset_args, limit


@pytest.mark.parametrize("url", ["http://localhost:8000", "http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"])
def test_endpoint(url):
    assert endpoint(url) == ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions")


def test_provider_specific_extra():
    local = Model("a", "vllm", "http://localhost:8000", "m", extra_body={"ignore_eos": True})
    assert request_body(local, {}, "x", stream=True)["ignore_eos"]
    remote = Model("a", "deepseek", "https://api.deepseek.com", "m", extra_body={"ignore_eos": True})
    with pytest.raises(ValueError):
        request_body(remote, {}, "x", stream=True)
    assert request_body(Model("d", "deepseek", "https://api.deepseek.com", "m"),
                        {"thinking_mode": "on"}, "x", stream=False)["thinking"] == {"type": "enabled"}


def test_config_rejects_unquoted_yaml_thinking_mode(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("""models:
  mock:
    provider: openai_compatible
    base_url: http://localhost:8000/v1
    model: mock
quality:
  thinking_mode: off
""", encoding="utf-8")
    with pytest.raises(ValueError, match="thinking_mode"):
        load_config(path)


def test_stream_reasoning_answer_and_usage():
    lines = ["data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "think"}}]}),
             "data: " + json.dumps({"choices": [{"delta": {"content": "OK"}}]}),
             "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": 3}}),
             "data: [DONE]"]
    result = parse_sse_lines(lines)
    assert (result.reasoning, result.content, result.usage["completion_tokens"]) == ("think", "OK", 3)
    assert parse_nonstream({"choices": [{"message": {"content": "a", "reasoning_content": "r"}}]}).content == "a"
    timed = Reply()
    accept_chunk(timed, {"choices": [{"delta": {"reasoning_content": "r"}}]}, .1)
    accept_chunk(timed, {"choices": [{"delta": {"content": "a"}}]}, .3)
    assert timed.first_content_s == .1 and timed.first_answer_s == .3


def test_sampling_scope_and_manifest(tmp_path):
    assert limit("mmlu", "quick") == 10
    assert limit("gsm8k", "quick") == 300
    assert len(dataset_args("mmlu", "smoke")["subset_list"]) == 5
    directory = tmp_path / "predictions" / "model"
    directory.mkdir(parents=True)
    row = {"index": 0, "messages": [{"role": "user", "content": "Q"}], "model_output": {"choices": []}}
    (directory / "gsm8k_main.jsonl").write_text(json.dumps(row) + "\n")
    first = collect_manifest(tmp_path, ["gsm8k"])["gsm8k"]["sha256"]
    row["model_output"]["choices"] = [{"message": {"content": "answer"}}]
    (directory / "gsm8k_main.jsonl").write_text(json.dumps(row) + "\n")
    assert collect_manifest(tmp_path, ["gsm8k"])["gsm8k"]["sha256"] == first


def test_paired_ci():
    left = [{"key": str(i), "subject": "a", "correct": 0} for i in range(10)]
    right = [{"key": str(i), "subject": "a", "correct": 1} for i in range(10)]
    assert _paired_ci(left, right, False) == (100.0, 100.0)


def test_perf_missing_usage_and_window(tmp_path):
    folder = tmp_path / "round"
    folder.mkdir()
    rows = [{"id": 1, "start": 1, "end": 2, "elapsed_s": 1, "http_status": 200, "error": None,
             "first_content_s": .2, "first_answer_s": .3, "usage": {"completion_tokens": 8}},
            {"id": 2, "start": 2, "end": 4, "elapsed_s": 2, "http_status": 200, "error": None,
             "first_content_s": .4, "first_answer_s": .5, "usage": None}]
    (folder / "requests.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    result = perf_round({"folder": str(folder), "concurrency": 1, "requests": 2, "status": "complete"},
                        {"warmup_requests": 0})
    assert result["window_s"] == 3
    assert result["output_token_throughput"] is None
    assert result["request_throughput"] == pytest.approx(2 / 3)


def test_empty_summary(tmp_path):
    output = tmp_path / "reports"
    summarize(tmp_path / "outputs", "baseline", output)
    assert {"quality_summary.csv", "performance_summary.csv", "comparison.json", "report.md", "report.html"} <= {x.name for x in output.iterdir()}


def test_simulated_report_preserves_incomplete_and_unrun_status(tmp_path):
    run = tmp_path / "outputs" / "mock-run"
    (run / "quality").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": "mock-run", "created_at": "2026-01-01",
        "model": {"alias": "mock", "simulated": True}, "profile": "smoke", "simulated": True}), encoding="utf-8")
    (run / "quality" / "quality_status.json").write_text(json.dumps({
        "gsm8k": {"status": "incomplete", "expected": 5, "reason": "请求失败"}}), encoding="utf-8")
    output = tmp_path / "reports"
    result = summarize(tmp_path / "outputs", "mock", output, simulation_only=True)
    rows = {row["dataset"]: row for row in result["quality"]}
    assert rows["gsm8k"]["status"] == "incomplete"
    assert rows["gsm8k"]["accuracy"] is None
    assert rows["mmlu"]["status"] == "not_run"
    assert all(row["simulated"] for row in rows.values())
    assert "模拟运行" in (output / "report.md").read_text(encoding="utf-8")
    assert "模拟运行" in (output / "report.html").read_text(encoding="utf-8")
    assert result["reference_model"] == "mock"
    assert "baseline" not in result


def test_cli_returns_failure_for_incomplete_result(tmp_path, monkeypatch):
    config = tmp_path / "models.yaml"
    config.write_text("""models:
  mock:
    provider: openai_compatible
    base_url: http://127.0.0.1:1/v1
    model: mock-model
""", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(cli, "new_run", lambda alias, output: run)
    monkeypatch.setattr(cli, "quality", lambda *args, **kwargs: {
        "gsm8k": {"status": "incomplete", "reason": "request failed"}})
    code = cli.main(["quality", "--config", str(config), "--model", "mock",
                     "--profile", "smoke", "--datasets", "gsm8k", "--output", str(tmp_path)])
    assert code == 1
    monkeypatch.setattr(cli, "quality", lambda *args, **kwargs: {
        "gsm8k": {"status": "complete", "reason": None}})
    code = cli.main(["quality", "--config", str(config), "--model", "mock",
                     "--profile", "smoke", "--datasets", "gsm8k", "--output", str(tmp_path)])
    assert code == 0


def test_any_two_models_can_be_compared(tmp_path):
    outputs = tmp_path / "outputs"
    prediction = {"index": 0, "messages": [{"role": "user", "content": "2+2=?"}],
                  "model_output": {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]}}
    for alias, family, score in (("model_a", "family-a", 0.0), ("model_b", "family-b", 1.0)):
        run = outputs / alias
        pred_dir = run / "quality" / "gsm8k" / "predictions" / alias
        review_dir = run / "quality" / "gsm8k" / "reviews" / alias
        pred_dir.mkdir(parents=True)
        review_dir.mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({
            "run_id": alias, "created_at": "2026-01-01", "finished_at": "2026-01-01",
            "model": {"alias": alias, "base_model": family, "hardware": "same-gpu",
                      "server_parameters": {}},
            "profile": "smoke", "quality_settings": {"output_budget": 32, "temperature": 0,
                                                      "thinking_mode": "off"},
            "simulated": False}), encoding="utf-8")
        (run / "quality" / "quality_status.json").write_text(json.dumps({
            "gsm8k": {"status": "complete", "expected": 1, "sample_hash": "same-samples"}}),
            encoding="utf-8")
        (pred_dir / "gsm8k_main.jsonl").write_text(json.dumps(prediction) + "\n", encoding="utf-8")
        review = {"index": 0, "target": ["4"], "sample_score": {"score": {
            "main_score_name": "accuracy", "value": {"accuracy": score},
            "prediction": "4", "extracted_prediction": "4"}}}
        (review_dir / "gsm8k_main.jsonl").write_text(json.dumps(review) + "\n", encoding="utf-8")

    result = summarize(outputs, "model_a", tmp_path / "reports")
    row = next(x for x in result["quality"] if x["model"] == "model_b" and x["dataset"] == "gsm8k")
    assert row["comparison_type"] == "model_comparison"
    assert row["controlled_comparison"] is True
    assert row["delta_pp"] == 100.0
