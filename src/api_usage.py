from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any


USAGE_LOG = Path(".cache/ai/usage.jsonl")
USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)


def estimate_tokens(text_or_chars: str | int) -> int:
    char_count = text_or_chars if isinstance(text_or_chars, int) else len(text_or_chars)
    return max(1, math.ceil(char_count / 1.6)) if char_count else 0


def get_api_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"


def classify_api_error(exc: Exception | str) -> str:
    message = str(exc).lower()
    if "insufficient_quota" in message or "quota" in message:
        return "quota_exhausted"
    if "429" in message or "rate limit" in message or "ratelimit" in message:
        return "rate_limited"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if "connect" in message or "connection" in message or "10013" in message:
        return "network_error"
    return "api_error"


def sanitize_error_message(exc: Exception | str, limit: int = 240) -> str:
    return str(exc).replace("\n", " ")[:limit]


def record_api_event(event: dict[str, Any]) -> None:
    payload = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": event.get("status", "unknown"),
        "operation": event.get("operation", "-"),
        "model": event.get("model", "-"),
        "base_url": event.get("base_url", get_api_base_url()),
        "duration_ms": int(event.get("duration_ms", 0)),
        "input_chars": int(event.get("input_chars", 0)),
        "output_chars": int(event.get("output_chars", 0)),
        "estimated_input_tokens": int(event.get("estimated_input_tokens", 0)),
        "estimated_output_tokens": int(event.get("estimated_output_tokens", 0)),
        "actual_prompt_tokens": event.get("actual_prompt_tokens"),
        "actual_completion_tokens": event.get("actual_completion_tokens"),
        "actual_total_tokens": event.get("actual_total_tokens"),
        "error_type": event.get("error_type", ""),
        "error_message": event.get("error_message", ""),
    }
    with USAGE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_api_events(limit: int = 300) -> list[dict[str, Any]]:
    if not USAGE_LOG.exists():
        return []
    rows = []
    with USAGE_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


def summarize_api_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    success = [event for event in events if event.get("status") == "success"]
    failures = [event for event in events if event.get("status") == "failure"]
    cache_hits = [event for event in events if event.get("status") == "cache_hit"]
    quota_errors = [event for event in failures if event.get("error_type") == "quota_exhausted"]
    rate_limits = [event for event in failures if event.get("error_type") == "rate_limited"]

    total_input_tokens = sum(int(event.get("estimated_input_tokens") or 0) for event in events)
    total_output_tokens = sum(int(event.get("estimated_output_tokens") or 0) for event in events)
    actual_total_tokens = sum(int(event.get("actual_total_tokens") or 0) for event in events)
    last_event = events[-1] if events else {}

    if quota_errors:
        health = "额度不足"
    elif rate_limits:
        health = "限流中"
    elif failures and not success:
        health = "异常"
    elif success or cache_hits:
        health = "可用"
    else:
        health = "未检测"

    return {
        "health": health,
        "total_calls": len(events),
        "success_calls": len(success),
        "failure_calls": len(failures),
        "cache_hits": len(cache_hits),
        "quota_errors": len(quota_errors),
        "rate_limits": len(rate_limits),
        "estimated_tokens": total_input_tokens + total_output_tokens,
        "actual_tokens": actual_total_tokens,
        "last_status": last_event.get("status", "-"),
        "last_error_type": last_event.get("error_type", ""),
        "last_error_message": last_event.get("error_message", ""),
        "last_ts": last_event.get("ts", "-"),
    }
