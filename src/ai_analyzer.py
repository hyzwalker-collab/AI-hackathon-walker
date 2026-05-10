from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.aligner import build_integration_plan
from src.api_usage import (
    classify_api_error,
    estimate_tokens,
    get_api_base_url,
    load_api_events,
    record_api_event,
    sanitize_error_message,
)
from src.kg_builder import build_knowledge_graph
from src.rag import answer_question_locally, build_chunks_placeholder


load_dotenv()

CACHE_DIR = Path(".cache/ai")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BLOCKING_ERRORS = {"quota_exhausted", "rate_limited", "timeout", "network_error"}


def is_ai_configured() -> bool:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(api_key and api_key != "your_api_key_here")


def get_model_name() -> str:
    default_model = "deepseek-ai/DeepSeek-V4-Pro"
    return os.getenv("LLM_MODEL", default_model).strip() or default_model


def get_model_options() -> list[str]:
    configured = os.getenv("LLM_MODEL_OPTIONS", "").strip()
    options = [item.strip() for item in configured.split(",") if item.strip()]
    current = get_model_name()
    if current not in options:
        options.insert(0, current)
    return options


def set_runtime_model(model_name: str) -> None:
    os.environ["LLM_MODEL"] = model_name.strip()


def get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    timeout = get_float_env("AI_TIMEOUT_SECONDS", 45.0)
    max_retries = get_int_env("AI_MAX_RETRIES", 0)
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
    return OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)


def get_max_output_tokens() -> int | None:
    value = get_int_env("AI_MAX_OUTPUT_TOKENS", 0)
    return value if value > 0 else None


def cache_path(cache_tag: str, system_prompt: str, user_prompt: str) -> Path:
    digest = hashlib.sha256(
        f"{get_model_name()}\n{cache_tag}\n{system_prompt}\n{user_prompt}".encode("utf-8")
    ).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def read_cached_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_cached_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def should_skip_business_ai() -> bool:
    """Skip heavy AI calls briefly after provider-side failures."""
    events = load_api_events(limit=12)
    for event in reversed(events):
        if event.get("operation") == "quota_probe" and event.get("status") == "success":
            return False
        if event.get("error_type") in BLOCKING_ERRORS:
            return get_bool_env("AI_SKIP_AFTER_PROVIDER_ERROR", True)
    return False


def extract_json(text: str) -> dict[str, Any]:
    """提取并解析 JSON，支持多种 AI 返回格式"""
    if not text or not text.strip():
        raise ValueError("Empty response")
    text = text.strip()
    
    # 1. 去除 Markdown 代码块标记
    text = text.strip()
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text)
    
    # 2. 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 3. 找 JSON 对象区间
    start = text.find('{"')
    if start < 0:
        start = text.find('{')
    end = text.rfind('}')
    
    if start >= 0 and end > start:
        json_str = text[start:end + 1]
        # 4. 清理尾部多余逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # 5. 尝试修复常见问题
    text = text.strip()
    text = text.replace("'", '"')
    text = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_raw_text": text[:4000], "_parse_error": "json_decode_failed"}


def collect_stream_text(messages: list[dict[str, str]]) -> str:
    kwargs: dict[str, Any] = {
        "model": get_model_name(),
        "messages": messages,
        "temperature": 0.2,
        "stream": True,
    }
    max_tokens = get_max_output_tokens()
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    client = get_client()
    stream = client.chat.completions.create(**kwargs)
    parts: list[str] = []
    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        delta = getattr(choices[0], "delta", None) if choices else None
        if delta is None:
            continue
        content = getattr(delta, "content", None) or ""
        if content:
            parts.append(content)
    return "".join(parts)


def call_ai_json(system_prompt: str, user_prompt: str, cache_tag: str = "default") -> dict[str, Any]:
    if not is_ai_configured():
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    input_chars = len(system_prompt) + len(user_prompt)
    started_at = time.perf_counter()
    if cache_tag != "quota_probe" and should_skip_business_ai():
        record_api_event({
            "status": "skipped",
            "operation": cache_tag,
            "model": get_model_name(),
            "base_url": get_api_base_url(),
            "duration_ms": (time.perf_counter() - started_at) * 1000,
            "input_chars": input_chars,
            "output_chars": 0,
            "estimated_input_tokens": estimate_tokens(input_chars),
            "estimated_output_tokens": 0,
            "error_type": "recent_provider_failure",
            "error_message": "Skipped after a recent quota, rate limit, timeout, or network failure.",
        })
        raise RuntimeError("Recent provider failure; skipped business AI call.")

    path = cache_path(cache_tag, system_prompt, user_prompt)
    cached = read_cached_json(path)
    if cached is not None:
        record_api_event({
            "status": "cache_hit",
            "operation": cache_tag,
            "model": get_model_name(),
            "base_url": get_api_base_url(),
            "duration_ms": (time.perf_counter() - started_at) * 1000,
            "input_chars": input_chars,
            "output_chars": 0,
            "estimated_input_tokens": estimate_tokens(input_chars),
            "estimated_output_tokens": 0,
        })
        return cached

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    client = get_client()
    kwargs: dict[str, Any] = {
        "model": get_model_name(),
        "messages": messages,
        "temperature": 0.2,
    }
    max_tokens = get_max_output_tokens()
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    try:
        actual_prompt_tokens = None
        actual_completion_tokens = None
        actual_total_tokens = None
        if get_bool_env("AI_FORCE_STREAM", False):
            content = collect_stream_text(messages)
        else:
            try:
                response = client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
            except Exception:
                content = collect_stream_text(messages)
            else:
                try:
                    usage = getattr(response, "usage", None)
                    actual_prompt_tokens = getattr(usage, "prompt_tokens", None)
                    actual_completion_tokens = getattr(usage, "completion_tokens", None)
                    actual_total_tokens = getattr(usage, "total_tokens", None)
                    content = response.choices[0].message.content or "{}"
                except Exception:
                    content = collect_stream_text(messages)

        data = extract_json(content)
        write_cached_json(path, data)
        record_api_event({
            "status": "success",
            "operation": cache_tag,
            "model": get_model_name(),
            "base_url": get_api_base_url(),
            "duration_ms": (time.perf_counter() - started_at) * 1000,
            "input_chars": input_chars,
            "output_chars": len(content),
            "estimated_input_tokens": estimate_tokens(input_chars),
            "estimated_output_tokens": estimate_tokens(content),
            "actual_prompt_tokens": actual_prompt_tokens,
            "actual_completion_tokens": actual_completion_tokens,
            "actual_total_tokens": actual_total_tokens,
        })
        return data
    except Exception as exc:
        record_api_event({
            "status": "failure",
            "operation": cache_tag,
            "model": get_model_name(),
            "base_url": get_api_base_url(),
            "duration_ms": (time.perf_counter() - started_at) * 1000,
            "input_chars": input_chars,
            "output_chars": 0,
            "estimated_input_tokens": estimate_tokens(input_chars),
            "estimated_output_tokens": 0,
            "error_type": classify_api_error(exc),
            "error_message": sanitize_error_message(exc),
        })
        raise


def probe_ai_status() -> dict[str, Any]:
    if not is_ai_configured():
        return {"ok": False, "message": "OPENAI_API_KEY 未配置。"}

    started_at = time.perf_counter()
    messages = [{"role": "user", "content": "OK"}]
    input_chars = len(messages[0]["content"])
    try:
        content = collect_stream_text(messages).strip()
        ok = bool(content)
        record_api_event({
            "status": "success" if ok else "failure",
            "operation": "quota_probe",
            "model": get_model_name(),
            "base_url": get_api_base_url(),
            "duration_ms": (time.perf_counter() - started_at) * 1000,
            "input_chars": input_chars,
            "output_chars": len(content),
            "estimated_input_tokens": estimate_tokens(input_chars),
            "estimated_output_tokens": estimate_tokens(content),
            "error_type": "" if ok else "empty_response",
            "error_message": "" if ok else "API returned empty response.",
        })
        return {"ok": ok, "message": content or "空响应"}
    except Exception as exc:
        error_type = classify_api_error(exc)
        message = sanitize_error_message(exc)
        record_api_event({
            "status": "failure",
            "operation": "quota_probe",
            "model": get_model_name(),
            "base_url": get_api_base_url(),
            "duration_ms": (time.perf_counter() - started_at) * 1000,
            "input_chars": input_chars,
            "output_chars": 0,
            "estimated_input_tokens": estimate_tokens(input_chars),
            "estimated_output_tokens": 0,
            "error_type": error_type,
            "error_message": message,
        })
        return {"ok": False, "message": message, "error_type": error_type}


def select_representative_chapters(book: dict[str, Any]) -> list[dict[str, Any]]:
    chapters = book.get("chapters", [])
    max_chapters = get_int_env("AI_MAX_CHAPTERS", 5)
    if len(chapters) <= max_chapters:
        return chapters

    indexed = list(enumerate(chapters))
    first = indexed[:1]
    last = indexed[-1:]
    largest = sorted(indexed[1:-1], key=lambda item: item[1].get("char_count", 0), reverse=True)
    selected = first + largest[: max(0, max_chapters - 2)] + last
    return [chapter for _, chapter in sorted(selected, key=lambda item: item[0])]


def trim_chapter(chapter: dict[str, Any], char_limit: int) -> dict[str, Any]:
    content = chapter.get("content", "")
    return {
        "chapter_id": chapter.get("chapter_id"),
        "title": chapter.get("title"),
        "page_start": chapter.get("page_start"),
        "char_count": chapter.get("char_count", 0),
        "content": content[:char_limit],
    }


def build_ai_knowledge_graph(book: dict[str, Any]) -> dict[str, Any]:
    fallback = build_knowledge_graph([book])
    char_limit = get_int_env("AI_CHAPTER_CHAR_LIMIT", 900)
    source = [trim_chapter(chapter, char_limit) for chapter in select_representative_chapters(book)]

    prompt = f"""
为教材《{book.get('filename', '-')}》抽取小型知识图谱。
只返回紧凑 JSON，不要解释，不要 Markdown。最多 8 个 nodes、10 条 edges。
JSON 格式：
{{
  "nodes": [
    {{
      "id": "n1",
      "name": "知识点",
      "category": "章节/核心概念/一般概念",
      "chapter": "来源章节",
      "page": "页码",
      "definition": "短定义",
      "source_textbook": "{book.get('filename', '-')}",
      "size": 14
    }}
  ],
  "edges": [
    {{
      "source": "节点id",
      "target": "节点id",
      "relation_type": "prerequisite/contains/related/application",
      "description": "关系说明"
    }}
  ]
}}
关系类型限于 prerequisite/contains/parallel/application/related。
教材片段：
{json.dumps(source, ensure_ascii=False)}
"""

    try:
        graph = call_ai_json(
            "你是面向教材整合的知识图谱专家，必须输出可解析 JSON。",
            prompt,
            cache_tag="knowledge_graph",
        )
        if graph.get("nodes") and graph.get("edges"):
            return graph
    except Exception:
        return fallback
    return fallback


def build_ai_integration_plan(
    textbooks: list[dict[str, Any]],
    feedback_items: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    fallback = build_integration_plan(textbooks)
    char_limit = get_int_env("AI_INTEGRATION_CHAR_LIMIT", 700)
    summaries = []
    for book in textbooks:
        summaries.append({
            "textbook_id": book.get("textbook_id"),
            "filename": book.get("filename"),
            "chapters": [
                trim_chapter(chapter, char_limit)
                for chapter in select_representative_chapters(book)
            ],
        })
    feedback_items = feedback_items or []

    prompt = f"""
比较教材片段，输出跨教材整合决策。
只返回紧凑 JSON，不要解释，不要 Markdown。最多 12 条 decisions。
JSON 格式：
{{
  "decisions": [
    {{
      "term": "知识点",
      "action": "merge/keep/supplement/reorder",
      "confidence": 0.85,
      "source_count": 2,
      "textbooks": "来源教材",
      "reason": "为什么这样处理"
    }}
  ]
}}
目标：保留核心概念和前置依赖，压缩到 30% 精华版。
教师反馈：
{json.dumps(feedback_items, ensure_ascii=False)}

教材片段：
{json.dumps(summaries, ensure_ascii=False)}
"""
    try:
        data = call_ai_json(
            "你是教材融合与课程知识体系设计专家，必须输出可解析 JSON。",
            prompt,
            cache_tag="integration_plan",
        )
        decisions = data.get("decisions", [])
        if decisions:
            return decisions
    except Exception:
        return fallback
    return fallback


def answer_question_with_ai(textbooks: list[dict[str, Any]], question: str) -> dict[str, Any]:
    local_answer = answer_question_locally(textbooks, question, top_k=5)
    chunks = local_answer.get("citations") or build_chunks_placeholder(textbooks, chunk_size=800, overlap=120)[:5]
    context = "\n\n".join(
        f"[{idx + 1}] {chunk['textbook']} / {chunk['chapter']} / {chunk.get('page', '-')}\n{chunk['text'][:800]}"
        for idx, chunk in enumerate(chunks)
    )

    prompt = f"""
基于教材片段回答问题。只返回紧凑 JSON，不要 Markdown。
问题：{question}

教材片段：
{context}

{{
  "answer": "回答正文",
  "citations": ["1", "2"]
}}
"""
    try:
        data = call_ai_json(
            "你是严谨的教材问答助手，只能基于给定片段回答。",
            prompt,
            cache_tag="rag_answer",
        )
        if data.get("answer"):
            return {
                "answer": data["answer"],
                "citations": local_answer.get("citations", []),
            }
        if data.get("_raw_text"):
            return {
                "answer": data["_raw_text"],
                "citations": local_answer.get("citations", []),
            }
    except Exception:
        return local_answer
    return local_answer
