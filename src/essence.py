from __future__ import annotations

from typing import Any

from src.aligner import build_integration_plan, protected_terms_from_feedback
from src.kg_builder import extract_terms


def build_essence_outline(
    textbooks: list[dict[str, Any]],
    decisions: list[dict[str, Any]] | None = None,
    feedback_items: list[dict[str, str]] | None = None,
    target_ratio: float = 0.3,
) -> dict[str, Any]:
    total_chars = sum(book.get("total_chars", 0) for book in textbooks)
    target_chars = int(total_chars * target_ratio)
    decisions = decisions or build_integration_plan(textbooks, feedback_items)
    protected_terms = protected_terms_from_feedback(feedback_items)
    decision_by_term = {item["term"]: item for item in decisions}

    candidates: list[dict[str, Any]] = []
    for book in textbooks:
        for chapter in book.get("chapters", []):
            content = chapter.get("content", "")
            terms = extract_terms(content, limit=10)
            score = 0.0
            reasons: list[str] = []

            for term in terms:
                decision = decision_by_term.get(term)
                if not decision:
                    continue
                if decision["action"] == "merge":
                    score += 3.0
                    reasons.append(f"{term}: 跨教材重叠，合并保留")
                elif decision["action"] == "keep":
                    score += 2.5
                    reasons.append(f"{term}: 信息量高或教师要求保留")
                elif decision["action"] == "supplement":
                    score += 1.2
                    reasons.append(f"{term}: 覆盖不足，进入补充队列")

            if protected_terms.intersection(terms):
                score += 4.0
                reasons.append("命中教师反馈保护词")

            if not reasons and terms:
                reasons.append("保留章节高频核心概念")

            candidates.append({
                "textbook": book.get("filename", "-"),
                "chapter": chapter.get("title", "-"),
                "page_start": chapter.get("page_start", "-"),
                "page_end": chapter.get("page_end", "-"),
                "char_count": chapter.get("char_count", 0),
                "terms": terms[:6],
                "score": score + min(chapter.get("char_count", 0) / 2000, 2.0),
                "reason": "；".join(dict.fromkeys(reasons[:4])),
                "summary": content[:520].replace("\n", " "),
            })

    selected: list[dict[str, Any]] = []
    selected_chars = 0
    for item in sorted(candidates, key=lambda row: row["score"], reverse=True):
        if not item["summary"].strip():
            continue
        remaining_chars = target_chars - selected_chars
        if remaining_chars <= 0:
            break
        budget_chars = min(max(300, min(item["char_count"], 900)), remaining_chars)
        if selected and budget_chars < 120:
            continue
        item["essence_chars"] = budget_chars
        item["summary"] = item["summary"][:budget_chars]
        selected.append(item)
        selected_chars += budget_chars

    compression_ratio = selected_chars / total_chars if total_chars else 0
    return {
        "total_chars": total_chars,
        "target_chars": target_chars,
        "selected_chars": selected_chars,
        "compression_ratio": compression_ratio,
        "sections": selected,
    }


def essence_to_markdown(outline: dict[str, Any]) -> str:
    lines = [
        "# 30% 精华版教材草案",
        "",
        "## 压缩概览",
        "",
        f"- 原始总字符数：{outline.get('total_chars', 0)}",
        f"- 目标字符数：{outline.get('target_chars', 0)}",
        f"- 当前精华版字符数：{outline.get('selected_chars', 0)}",
        f"- 当前压缩比：{outline.get('compression_ratio', 0):.2%}",
        "",
        "## 精华版目录与内容",
        "",
    ]
    for idx, section in enumerate(outline.get("sections", []), start=1):
        terms = "、".join(section.get("terms", [])) or "-"
        lines.extend([
            f"### {idx}. {section.get('chapter', '-')}",
            "",
            f"- 来源教材：{section.get('textbook', '-')}",
            f"- 来源页码/片段：{section.get('page_start', '-')} - {section.get('page_end', '-')}",
            f"- 核心知识点：{terms}",
            f"- 保留理由：{section.get('reason', '-')}",
            "",
            section.get("summary", ""),
            "",
        ])
    return "\n".join(lines)
