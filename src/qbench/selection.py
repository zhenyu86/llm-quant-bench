from __future__ import annotations

import hashlib
import json
from pathlib import Path

DATASETS = ("mmlu", "ceval", "gsm8k", "arc", "hellaswag")
FEW_SHOT = {"mmlu": 5, "ceval": 5, "gsm8k": 4, "arc": 0, "hellaswag": 0}
SPLITS = {"mmlu": "test", "ceval": "val", "gsm8k": "test", "arc": "test", "hellaswag": "validation"}
# Smoke samples one subject from each of five subjects. Quick runs every subject.
SMOKE_SUBSETS = {"mmlu": ["abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge"],
                 "ceval": ["computer_network", "operating_system", "computer_architecture", "college_programming", "college_physics"],
                 "arc": ["ARC-Challenge"]}
EXPECTED = {
    "smoke": {name: 5 for name in DATASETS},
    "quick": {"mmlu": 570, "ceval": 520, "gsm8k": 300, "arc": 300, "hellaswag": 300},
    "full": {name: None for name in DATASETS},
}


def dataset_args(dataset: str, profile: str, dataset_ids: dict | None = None) -> dict:
    if dataset not in DATASETS or profile not in EXPECTED:
        raise ValueError("无效数据集或 profile")
    args = {"few_shot_num": FEW_SHOT[dataset], "few_shot_random": False,
            "shuffle": profile != "full"}
    if profile == "smoke" and dataset in SMOKE_SUBSETS:
        args["subset_list"] = SMOKE_SUBSETS[dataset]
    if dataset == "arc":
        args["subset_list"] = ["ARC-Challenge"]
    if dataset_ids and dataset in dataset_ids:
        args["dataset_id"] = dataset_ids[dataset]
    return args


def limit(dataset: str, profile: str) -> int | None:
    if profile == "full":
        return None
    if profile == "quick":
        return 10 if dataset in {"mmlu", "ceval"} else 300
    return 1 if dataset in {"mmlu", "ceval"} else 5


def sample_key(item: dict, dataset: str) -> str:
    messages = item.get("messages") or []
    prompts = [(m.get("role"), m.get("content")) for m in messages
               if isinstance(m, dict) and m.get("role") in {"system", "user"}]
    value = [item.get("index"), item.get("metadata"), prompts]
    payload = json.dumps([dataset, value], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_manifest(root: Path, datasets: list[str]) -> dict:
    items = {}
    for dataset in datasets:
        files = list((root / "predictions").rglob(f"{dataset}*.jsonl"))
        rows = [row for file in files for row in read_jsonl(file)]
        keys = sorted(sample_key(row, file.stem) for file in files for row in read_jsonl(file))
        items[dataset] = {"count": len(keys), "keys": keys,
                          "sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest()}
    return items


def verify_counts(manifest: dict, profile: str, datasets: list[str]) -> list[str]:
    errors = []
    for name in datasets:
        count = manifest.get(name, {}).get("count", 0)
        expected = EXPECTED[profile][name]
        if expected is not None and count != expected:
            errors.append(f"{name}: 实际 {count}，预期 {expected}")
    return errors
