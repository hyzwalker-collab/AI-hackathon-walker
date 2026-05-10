from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Lazy import for AI features
_ai_available = None


def _check_ai_available() -> bool:
    """检查 AI 是否可用"""
    global _ai_available
    if _ai_available is None:
        try:
            from src.ai_analyzer import is_ai_configured
            _ai_available = is_ai_configured()
        except Exception:
            _ai_available = False
    return _ai_available


def build_syllabus_outline(
    textbooks: list[dict[str, Any]],
    integration_plan: list[dict[str, Any]],
    feedback_items: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    基于整合方案生成教学大纲结构

    Returns:
        {
            "title": "课程名称",
            "objectives": ["学习目标1", ...],
            "chapters": [
                {
                    "title": "章节标题",
                    "hours": 2,
                    "key_points": ["要点1", ...],
                    "knowledge_terms": ["知识点1", ...],
                    "difficulty": "基础|进阶|高级"
                }
            ],
            "total_hours": 32,
            "total_points": 15
        }
    """
    if not textbooks:
        return _empty_syllabus()

    # 提取课程标题
    course_title = _extract_course_title(textbooks)

    # 从整合决策中提取知识点
    knowledge_units = _organize_knowledge_units(integration_plan)

    # 生成章节结构
    chapters = _build_chapter_structure(knowledge_units, textbooks)

    # 计算总学时
    total_hours = sum(ch.get("hours", 2) for ch in chapters)

    return {
        "title": course_title,
        "objectives": _generate_learning_objectives(knowledge_units),
        "chapters": chapters,
        "total_hours": total_hours,
        "total_points": len(knowledge_units),
        "compression_ratio": _calc_compression(textbooks, knowledge_units),
    }


def _empty_syllabus() -> dict[str, Any]:
    """返回空大纲"""
    return {
        "title": "课程教学大纲",
        "objectives": ["掌握本学科核心概念和基本原理"],
        "chapters": [],
        "total_hours": 0,
        "total_points": 0,
        "compression_ratio": 0,
    }


def _extract_course_title(textbooks: list[dict[str, Any]]) -> str:
    """从教材中提取课程标题"""
    if textbooks:
        first_book = textbooks[0]
        title = first_book.get("title", "")
        if title:
            return title
        filename = first_book.get("filename", "")
        if filename:
            return filename.rsplit(".", 1)[0]
    return "学科知识整合课程"


def _organize_knowledge_units(
    integration_plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """组织知识单元"""
    units = []

    for item in integration_plan:
        action = item.get("action", "")
        term = item.get("term", "")

        if action in ("keep", "merge") and term:
            units.append({
                "name": term,
                "action": action,
                "confidence": item.get("confidence", 0.5),
                "sources": item.get("textbooks", ""),
                "needs_review": item.get("needs_teacher_review", False),
            })

    # 按置信度排序
    units.sort(key=lambda x: (x.get("needs_review", False), -x["confidence"]))

    return units


def _build_chapter_structure(
    units: list[dict[str, Any]],
    textbooks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """构建章节结构"""
    chapters = []

    # 按知识点构建章节
    # 这里简化处理：按知识点数量分配章节
    points_per_chapter = 5
    current_chapter = {"title": "", "hours": 2, "key_points": [], "knowledge_terms": [], "difficulty": "基础"}

    for i, unit in enumerate(units):
        if i % points_per_chapter == 0:
            if current_chapter["knowledge_terms"]:
                chapters.append(current_chapter)
            chapter_num = i // points_per_chapter + 1
            current_chapter = {
                "title": f"第{chapter_num}部分",
                "hours": 2,
                "key_points": [],
                "knowledge_terms": [],
                "difficulty": "基础",
            }

        current_chapter["knowledge_terms"].append(unit["name"])

        # 根据置信度判断难度
        if unit["confidence"] < 0.6:
            current_chapter["difficulty"] = "进阶"

    if current_chapter["knowledge_terms"]:
        chapters.append(current_chapter)

    # 如果没有章节，从教材章节提取
    if not chapters and textbooks:
        for book in textbooks[:1]:
            for chapter in book.get("chapters", [])[:6]:
                chapters.append({
                    "title": chapter.get("title", f"章节{len(chapters)+1}"),
                    "hours": 2,
                    "key_points": [],
                    "knowledge_terms": [],
                    "difficulty": "基础",
                })

    return chapters


def _generate_learning_objectives(units: list[dict[str, Any]]) -> list[str]:
    """生成学习目标"""
    objectives = [
        "理解本学科的基本概念和核心原理",
        "掌握主要知识点之间的关系和逻辑结构",
    ]

    high_conf_units = [u for u in units if u.get("confidence", 0) >= 0.7]
    if high_conf_units:
        sample_terms = [u["name"] for u in high_conf_units[:3]]
        objectives.append(f"重点掌握：{'、'.join(sample_terms)}")

    objectives.append("能够综合运用所学知识解决实际问题")

    return objectives


def _calc_compression(
    textbooks: list[dict[str, Any]],
    units: list[dict[str, Any]],
) -> float:
    """计算压缩比"""
    original_chars = sum(b.get("total_chars", 0) for b in textbooks)
    if not original_chars:
        return 0

    # 估算精华字数（每个知识点约300字）
    essence_chars = len(units) * 300

    return essence_chars / original_chars if original_chars > 0 else 0


def syllabus_to_markdown(syllabus: dict[str, Any]) -> str:
    """将大纲转换为 Markdown 格式"""
    lines = [
        f"# {syllabus.get('title', '教学大纲')}",
        "",
        "## 📋 课程信息",
        f"- **总学时**: {syllabus.get('total_hours', 0)} 学时",
        f"- **知识点数**: {syllabus.get('total_points', 0)} 个",
        f"- **压缩比**: {syllabus.get('compression_ratio', 0):.1%}",
        "",
        "## 🎯 学习目标",
    ]

    for obj in syllabus.get("objectives", []):
        lines.append(f"- {obj}")

    lines.extend(["", "## 📚 章节结构", ""])

    for i, chapter in enumerate(syllabus.get("chapters", []), 1):
        chapter_title = chapter.get("title") or f"第{i}章"
        lines.append(f"### {chapter_title}")
        lines.append(f"- 学时: {chapter.get('hours', 2)} 小时")
        lines.append(f"- 难度: {chapter.get('difficulty', '基础')}")

        terms = chapter.get("knowledge_terms", [])
        if terms:
            lines.append(f"- 核心知识点 ({len(terms)} 个):")
            for term in terms[:8]:
                lines.append(f"  - {term}")
            if len(terms) > 8:
                lines.append(f"  - ...及其他 {len(terms) - 8} 个")

        lines.append("")

    lines.extend([
        "---",
        "*本大纲由 EduMerge Agent 自动生成*",
    ])

    return "\n".join(lines)


def build_ai_syllabus(
    textbooks: list[dict[str, Any]],
    integration_plan: list[dict[str, Any]],
    feedback_items: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """使用 AI 生成更智能的教学大纲"""
    if not _check_ai_available():
        return build_syllabus_outline(textbooks, integration_plan, feedback_items)

    try:
        from src.ai_analyzer import call_ai_json

        # 构建摘要
        summaries = []
        for book in textbooks[:3]:
            summaries.append({
                "title": book.get("title", book.get("filename", "-")),
                "chapters": [
                    ch.get("title", "-")
                    for ch in book.get("chapters", [])[:10]
                ],
            })

        plan_summary = [
            {"term": d.get("term"), "action": d.get("action"), "confidence": d.get("confidence")}
            for d in integration_plan[:30]
        ]

        prompt = f"""
基于以下教材和整合方案，生成一份结构化教学大纲。

教材信息：
{json.dumps(summaries, ensure_ascii=False)}

整合决策：
{json.dumps(plan_summary, ensure_ascii=False)}

只输出 JSON：
{{
  "title": "课程名称",
  "objectives": ["学习目标1", "学习目标2"],
  "chapters": [
    {{
      "title": "章节名称",
      "hours": 2,
      "key_points": ["重点1", "重点2"],
      "difficulty": "基础|进阶",
      "prerequisites": ["前置章节"]
    }}
  ],
  "total_hours": 32
}}
"""

        data = call_ai_json(
            "你是课程教学设计专家，必须输出有效 JSON。",
            prompt,
            cache_tag="syllabus",
        )

        if data.get("title"):
            # 计算压缩比
            data["compression_ratio"] = _calc_compression(textbooks, integration_plan)
            data["total_points"] = len(integration_plan)
            return data

    except Exception:
        pass

    return build_syllabus_outline(textbooks, integration_plan, feedback_items)
