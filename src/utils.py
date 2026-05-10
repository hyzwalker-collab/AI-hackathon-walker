from __future__ import annotations


def format_file_size(size_bytes: int) -> str:
    # Format file size for UI display.
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 ** 2):.1f} MB"


def safe_text(text: str) -> str:
    # Basic cleanup for extracted textbook text.
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)
