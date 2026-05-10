from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from src.kg_builder import extract_terms

# 教师反馈中的保护信号词
PROTECT_SIGNALS = {
    "保留", "不能删", "不能压缩", "考试", "重点", "实验",
    "案例", "前置", "基础", "核心", "必须掌握"
}


def protected_terms_from_feedback(
    feedback_items: list[dict[str, str]] | None,
) -> set[str]:
    """
    从教师反馈中提取需要保护的知识点
    """
    protected: set[str] = set()
    for item in feedback_items or []:
        content = item.get("content", "")
        if any(signal in content for signal in PROTECT_SIGNALS):
            protected.update(extract_terms(content, limit=12))
    return protected


def _semantic_align_nodes(
    nodes1: list[dict[str, Any]],
    nodes2: list[dict[str, Any]],
    threshold: float = 0.78,
) -> list[tuple[dict, dict, float]]:
    """
    基于 embedding 的语义对齐 - 识别跨教材的相似知识点

    Args:
        nodes1: 第一本教材的节点
        nodes2: 第二本教材的节点
        threshold: 相似度阈值

    Returns:
        [(node1, node2, similarity_score), ...]
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch

        if not hasattr(_semantic_align_nodes, "model"):
            _semantic_align_nodes.model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2",
                device="cpu",
            )

        model = _semantic_align_nodes.model

        # 提取节点名称
        names1 = [n.get("name", "") for n in nodes1]
        names2 = [n.get("name", "") for n in nodes2]

        # 编码
        emb1 = model.encode(names1, convert_to_tensor=True)
        emb2 = model.encode(names2, convert_to_tensor=True)

        # 计算余弦相似度矩阵
        sim_matrix = torch.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=2)

        # 找出高相似度匹配
        aligned_pairs = []
        for i, row in enumerate(sim_matrix):
            max_score, j = torch.max(row, dim=0)
            score = max_score.item()
            if score >= threshold:
                aligned_pairs.append((nodes1[i], nodes2[j.item()], score))

        return aligned_pairs

    except Exception:
        return []


def _token_based_align(
    nodes1: list[dict[str, Any]],
    nodes2: list[dict[str, Any]],
) -> list[tuple[dict, dict, float]]:
    """
    基于词标的的对齐 - 识别名称完全或高度相似的知识点
    """
    aligned = []
    for n1 in nodes1:
        name1 = n1.get("name", "").lower().strip()
        for n2 in nodes2:
            name2 = n2.get("name", "").lower().strip()
            # 完全匹配
            if name1 == name2:
                aligned.append((n1, n2, 1.0))
            # 部分匹配（包含关系）
            elif name1 in name2 or name2 in name1:
                if len(name1) >= 2 and len(name2) >= 2:
                    score = len(name1) / max(len(name1), len(name2))
                    if score >= 0.6:
                        aligned.append((n1, n2, score))
    return aligned


def build_cross_textbook_alignment(
    textbooks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    跨教材知识点对齐分析

    Returns:
        {
            "aligned_pairs": [(node1, node2, score), ...],
            "unique_terms": {textbook_id: set of unique terms},
            "missing_terms": {textbook_id: set of terms not in other books}
        }
    """
    if len(textbooks) < 2:
        return {
            "aligned_pairs": [],
            "unique_terms": {},
            "missing_terms": {},
        }

    # 提取每本教材的节点
    all_nodes: dict[str, list[dict[str, Any]]] = {}
    for book in textbooks:
        book_id = book.get("textbook_id", "-")
        # 从章节中提取知识点节点
        nodes = []
        for chapter in book.get("chapters", []):
            terms = extract_terms(chapter.get("content", ""), limit=8)
            for term in terms:
                nodes.append({
                    "id": f"{book_id}_{chapter.get('chapter_id', '')}_{term}",
                    "name": term,
                    "chapter": chapter.get("title", "-"),
                    "source": book.get("filename", "-"),
                    "book_id": book_id,
                })
        all_nodes[book_id] = nodes

    # 两两对齐
    aligned_pairs: list[tuple[dict, dict, float]] = []
    book_ids = list(all_nodes.keys())

    for i in range(len(book_ids)):
        for j in range(i + 1, len(book_ids)):
            nodes_i = all_nodes[book_ids[i]]
            nodes_j = all_nodes[book_ids[j]]

            # 词标对齐
            token_pairs = _token_based_align(nodes_i, nodes_j)
            aligned_pairs.extend(token_pairs)

            # 语义对齐
            if len(nodes_i) <= 50 and len(nodes_j) <= 50:  # 避免大矩阵
                sem_pairs = _semantic_align_nodes(nodes_i, nodes_j, threshold=0.82)
                # 过滤已通过词标对齐的
                token_names = {(p[0]["name"].lower(), p[1]["name"].lower()) for p in token_pairs}
                for p in sem_pairs:
                    key = (p[0]["name"].lower(), p[1]["name"].lower())
                    if key not in token_names:
                        aligned_pairs.append(p)

    # 统计每本教材的独有知识点
    unique_terms: dict[str, set[str]] = {}
    all_term_pairs: set[tuple] = set()
    for n1, n2, _ in aligned_pairs:
        all_term_pairs.add((n1["book_id"], n1["name"].lower()))
        all_term_pairs.add((n2["book_id"], n2["name"].lower()))

    for book_id, nodes in all_nodes.items():
        book_terms = {n["name"].lower() for n in nodes}
        aligned_in_book = {t[1] for t in all_term_pairs if t[0] == book_id}
        unique_terms[book_id] = book_terms - aligned_in_book

    return {
        "aligned_pairs": aligned_pairs,
        "unique_terms": unique_terms,
        "alignment_count": len(aligned_pairs),
    }


def build_integration_plan(
    textbooks: list[dict[str, Any]],
    feedback_items: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """
    生成跨教材整合决策

    决策类型：
    - merge: 多本教材共有，建议合并
    - keep: 单本独有但信息量大，建议保留
    - supplement: 仅出现在少数教材，需确认是否缺失
    - remove: 重复内容过多，考虑删除
    """
    term_sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    protected_terms = protected_terms_from_feedback(feedback_items)

    # 提取知识点来源
    for book in textbooks:
        book_id = book.get("textbook_id", "-")
        filename = book.get("filename", "-")
        for chapter in book.get("chapters", []):
            terms = extract_terms(chapter.get("content", ""), limit=10)
            for term in terms:
                term_sources[term].append({
                    "textbook": filename,
                    "book_id": book_id,
                    "chapter": chapter.get("title", "-"),
                    "page": chapter.get("page_start", "-"),
                    "char_count": chapter.get("char_count", 0),
                })

    # 跨教材对齐分析
    alignment = build_cross_textbook_alignment(textbooks)
    aligned_names = {
        (p[0]["name"].lower(), p[1]["name"].lower())
        for p in alignment["aligned_pairs"]
    }

    # 生成决策
    decisions: list[dict[str, Any]] = []
    for term, sources in sorted(term_sources.items(), key=lambda item: len(item[1]), reverse=True):
        unique_books = sorted({source["textbook"] for source in sources})
        term_lower = term.lower()

        # 检查是否为对齐的知识点
        is_aligned = any(
            (term_lower, other.lower()) in aligned_names
            for other in [s for s in unique_books]
        )

        if term in protected_terms:
            # 教师标记保护
            action = "keep"
            reason = "教师反馈标记为强制保留或人工复核"
            confidence = 1.0
        elif len(unique_books) >= 2:
            # 多本教材共有
            if is_aligned:
                action = "merge"
                reason = f"在 {len(unique_books)} 本教材中语义对齐，建议合并为统一知识点"
            else:
                action = "merge"
                reason = f"出现在 {len(unique_books)} 本教材中，建议合并表述"
            confidence = min(0.95, 0.48 + 0.12 * len(unique_books))
        elif sources[0]["char_count"] > 1200:
            # 单本独有但信息量大
            action = "keep"
            reason = "只在单本教材中出现，但所在章节信息量较高，建议保留"
            confidence = 0.65
        else:
            # 可能缺失
            action = "supplement"
            reason = "覆盖教材较少，建议教师确认是否为缺失或边缘知识点"
            confidence = 0.45

        decisions.append({
            "term": term,
            "action": action,
            "confidence": round(confidence, 2),
            "source_count": len(sources),
            "textbooks": "；".join(unique_books),
            "is_aligned": is_aligned,
            "needs_teacher_review": term in protected_terms or action == "supplement",
            "reason": reason,
        })

    return decisions[:80]


def summarize_integration_metrics(
    textbooks: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    统计整合指标
    """
    total_chars = sum(
        book.get("total_chars", 0)
        for book in textbooks
    )

    # 计算各类决策数量
    action_counts = Counter(d.get("action", "unknown") for d in decisions)

    # 估算精华版字数（基于保留和合并决策）
    kept_terms = [d for d in decisions if d.get("action") in ("keep", "merge")]
    # 假设每个知识点平均 300 字
    estimated_essence = len(kept_terms) * 300

    return {
        "total_chars": total_chars,
        "original_books": len(textbooks),
        "decision_count": len(decisions),
        "merge_count": action_counts.get("merge", 0),
        "keep_count": action_counts.get("keep", 0),
        "supplement_count": action_counts.get("supplement", 0),
        "aligned_pairs": sum(1 for d in decisions if d.get("is_aligned")),
        "estimated_essence_chars": estimated_essence,
        "compression_ratio": estimated_essence / total_chars if total_chars > 0 else 0,
    }
