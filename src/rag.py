from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Lazy imports for heavy dependencies
_cached_tfidf: TfidfVectorizer | None = None
_cached_chunks: list[dict[str, Any]] | None = None


def _get_tokenizer() -> callable:
    """获取中文分词器"""
    return lambda text: list(jieba.cut(text))


def build_chunks_placeholder(
    textbooks: list[dict[str, Any]], chunk_size: int = 700, overlap: int = 120
) -> list[dict[str, Any]]:
    """将教材内容分块"""
    chunks: list[dict[str, Any]] = []
    for book in textbooks:
        for chapter in book.get("chapters", []):
            text = chapter.get("content", "")
            step = max(chunk_size - overlap, 1)
            for start in range(0, len(text), step):
                piece = text[start : start + chunk_size]
                if not piece.strip():
                    continue
                chunks.append({
                    "text": piece,
                    "textbook": book.get("filename"),
                    "chapter": chapter.get("title"),
                    "page": chapter.get("page_start"),
                    "chunk_id": len(chunks),
                })
    return chunks


def _build_tfidf_index(chunks: list[dict[str, Any]]) -> tuple[TfidfVectorizer, np.ndarray]:
    """构建 TF-IDF 索引"""
    global _cached_tfidf, _cached_chunks
    if _cached_tfidf is not None and _cached_chunks == chunks:
        # Return cached index
        pass
    else:
        _cached_chunks = chunks
        tokenizer = _get_tokenizer()
        texts = [chunk["text"] for chunk in chunks]
        _cached_tfidf = TfidfVectorizer(
            tokenizer=tokenizer,
            ngram_range=(1, 2),
            max_features=10000,
        )
        _cached_tfidf.fit(texts)
    vectors = _cached_tfidf.transform([chunk["text"] for chunk in chunks])
    return _cached_tfidf, vectors


def _bm25_search(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """
    BM25 检索 - 基于词频的精确匹配
    返回: list of (chunk_index, score)
    """
    tokenizer = _get_tokenizer()
    query_terms = list(tokenizer(query))
    
    # 计算 BM25 分数
    tfidf, vectors = _build_tfidf_index(chunks)
    
    # 获取查询的 TF-IDF 向量
    query_vec = tfidf.transform([" ".join(query_terms)])
    
    # 计算余弦相似度作为 BM25 的近似
    scores = cosine_similarity(query_vec, vectors).flatten()
    
    # 返回 top-k
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


def _embedding_search(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """
    基于 Embedding 的语义检索
    返回: list of (chunk_index, score)
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        # 复用或创建模型实例
        if not hasattr(_embedding_search, "model"):
            _embedding_search.model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2",
                device="cpu",
            )
        
        model = _embedding_search.model
        
        # 编码查询和文本块
        query_emb = model.encode(query, convert_to_tensor=True)
        chunk_embs = model.encode(
            [chunk["text"] for chunk in chunks],
            convert_to_tensor=True,
        )
        
        # 计算余弦相似度
        similarities = torch.cosine_similarity(query_emb.unsqueeze(0), chunk_embs)
        
        # 返回 top-k
        top_indices = torch.argsort(similarities, descending=True)[:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0.3]
    
    except Exception:
        # 如果 embedding 不可用，返回空
        return []


def _rrf_fusion(
    results_list: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion (RRF) - 混合检索融合
    将多个检索结果按排名融合，适用于 BM25 + Embedding 组合
    """
    scores: dict[int, float] = defaultdict(float)
    
    for results in results_list:
        for rank, (chunk_idx, _) in enumerate(results):
            scores[chunk_idx] += 1.0 / (k + rank + 1)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
    use_embedding: bool = True,
) -> list[dict[str, Any]]:
    """
    混合检索 - BM25 + Embedding + RRF 融合
    
    Args:
        query: 用户问题
        chunks: 文本块列表
        top_k: 返回结果数量
        use_embedding: 是否使用 embedding 检索
    
    Returns:
        检索结果列表，包含 chunk 信息和融合分数
    """
    if not chunks:
        return []
    
    # 1. BM25 检索（精确匹配）
    bm25_results = _bm25_search(query, chunks, top_k=top_k * 2)
    
    # 2. Embedding 检索（语义匹配）
    if use_embedding:
        embed_results = _embedding_search(query, chunks, top_k=top_k * 2)
        # 3. RRF 融合
        fused = _rrf_fusion([bm25_results, embed_results], k=60)
    else:
        fused = bm25_results
    
    # 返回 top-k 结果
    hits = []
    for chunk_idx, score in fused[:top_k]:
        hit = dict(chunks[chunk_idx])
        hit["fusion_score"] = round(score, 4)
        hits.append(hit)
    
    return hits


def answer_question_locally(
    textbooks: list[dict[str, Any]],
    question: str,
    top_k: int = 4,
    use_embedding: bool = True,
) -> dict[str, Any]:
    """
    本地问答 - 使用混合检索

    Args:
        textbooks: 教材列表
        question: 用户问题
        top_k: 返回引用数量
        use_embedding: 是否使用 embedding 检索
    
    Returns:
        {"answer": str, "citations": list}
    """
    chunks = build_chunks_placeholder(textbooks, chunk_size=700, overlap=120)
    
    # 使用混合检索
    hits = hybrid_search(question, chunks, top_k=top_k, use_embedding=use_embedding)
    
    if not hits:
        return {
            "answer": "当前本地检索没有找到直接匹配的片段。建议换一个更具体的关键词，或先上传并解析更多教材。",
            "citations": [],
        }
    
    # 构建答案
    summary_lines = []
    for idx, hit in enumerate(hits[:3], start=1):
        snippet = hit["text"].replace("\n", " ")[:220]
        source = f"{hit['textbook']} / {hit['chapter']}"
        summary_lines.append(f"[{idx}] {source}\n{snippet}")
    
    return {
        "answer": "根据已解析教材，相关内容集中在以下片段：\n\n" + "\n\n".join(summary_lines),
        "citations": hits,
    }
