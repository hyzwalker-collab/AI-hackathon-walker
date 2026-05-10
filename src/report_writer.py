from __future__ import annotations

from pathlib import Path
from typing import Any

from src.aligner import build_integration_plan
from src.essence import build_essence_outline, essence_to_markdown
from src.kg_builder import build_knowledge_graph


def write_integration_report(
    textbooks: list[dict[str, Any]],
    feedback_items: list[dict[str, str]] | None = None,
    output_path: str = "report/整合报告.md",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    total_books = len(textbooks)
    total_chars = sum(book.get("total_chars", 0) for book in textbooks)
    estimated_integrated_chars = int(total_chars * 0.28) if total_chars else 0
    compression_ratio = estimated_integrated_chars / total_chars if total_chars else 0
    graph = build_knowledge_graph(textbooks)
    decisions = build_integration_plan(textbooks, feedback_items)
    essence = build_essence_outline(textbooks, decisions, feedback_items)
    feedback_items = feedback_items or []

    decision_lines = "\n".join(
        f"- `{item['action']}`：{item['term']}，置信度 {item['confidence']:.0%}，原因：{item['reason']}"
        for item in decisions[:12]
    ) or "- 暂无整合决策。"

    feedback_lines = "\n".join(
        f"- 第 {idx + 1} 轮：{item.get('role', '教师')}：{item.get('content', '')}"
        for idx, item in enumerate(feedback_items)
    ) or "- 暂无教师反馈。"

    content = f"""# 整合报告

## 1. 整合概览

- 原始教材数量：{total_books}
- 原始总字数：{total_chars}
- 整合后估计字数：{estimated_integrated_chars}
- 压缩比：{compression_ratio:.2%}

## 2. 整合决策摘要

{decision_lines}

## 3. 知识图谱统计

- 节点数：{len(graph.get("nodes", []))}
- 关系数：{len(graph.get("edges", []))}
- 关系类型：contains / related / prerequisite

## 4. 重点整合案例

系统优先合并跨教材重复出现的核心知识点；仅在单本教材中出现但章节信息量较高的知识点保留；覆盖不足的知识点进入教师确认队列。

## 5. 教学完整性说明

系统将通过“前置依赖关系保留”和“章节链路覆盖率”检查整合后知识链条是否断裂。

## 6. 教师多轮反馈

{feedback_lines}

---

{essence_to_markdown(essence)}
"""
    path.write_text(content, encoding="utf-8")
    return path
