from __future__ import annotations

import hashlib
from pathlib import Path

TOPICS = ["缓存命中", "二分查找", "TCP 拥塞控制", "矩阵乘法", "数据库索引", "向量检索", "排队论", "软件测试",
          "量化误差", "注意力机制", "并发控制", "图遍历", "概率分布", "内存管理", "网络延迟", "分布式一致性"]
VERBS = ["用三句话解释", "给出一个具体例子说明", "列出主要步骤并解释", "说明常见误区及正确做法"]


def prompts(count: int = 128) -> list[str]:
    return [f"{VERBS[(i // len(TOPICS)) % len(VERBS)]}{TOPICS[i % len(TOPICS)]}。请使用清晰的中文回答。"
            for i in range(count)]


def write_workload(path: Path, count: int = 128) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(prompts(count)) + "\n"
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()
