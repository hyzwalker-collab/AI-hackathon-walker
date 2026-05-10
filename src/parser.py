from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable

import fitz  # PyMuPDF
import pandas as pd

from src.utils import safe_text


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


CHAPTER_PATTERNS = [
    re.compile(r"^\s*第\s*[一二三四五六七八九十百零0-9]+\s*章\s+.+", re.MULTILINE),
    re.compile(r"^\s*Chapter\s+[0-9IVXLC]+\s+.+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*第\s*[一二三四五六七八九十百零0-9]+\s*节\s+.+", re.MULTILINE),
]


ProgressCallback = Callable[[float, str], None]


def emit_progress(callback: ProgressCallback | None, value: float, message: str) -> None:
    if callback is None:
        return
    callback(max(0.0, min(1.0, value)), message)


def save_uploaded_file(uploaded_file: Any) -> Path:
    target = UPLOAD_DIR / uploaded_file.name
    with target.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return target


def parse_uploaded_file(
    file_path: Path,
    textbook_id: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(file_path, textbook_id, progress_callback)
    if suffix == ".docx":
        return parse_docx(file_path, textbook_id, progress_callback)
    if suffix in {".xlsx", ".xls"}:
        return parse_excel(file_path, textbook_id, progress_callback)
    if suffix in {".md", ".markdown"}:
        return parse_plain_text(file_path, textbook_id, progress_callback)
    if suffix == ".txt":
        return parse_plain_text(file_path, textbook_id, progress_callback)

    raise ValueError(f"Unsupported file type: {suffix}")


def parse_pdf(
    file_path: Path,
    textbook_id: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    emit_progress(progress_callback, 0.03, "打开 PDF")
    doc = fitz.open(file_path)

    page_texts: list[dict[str, Any]] = []
    total_chars = 0
    total_pages = len(doc)

    for page_idx in range(total_pages):
        page = doc.load_page(page_idx)
        text = safe_text(page.get_text("text"))
        page_texts.append({"page": page_idx + 1, "text": text, "char_count": len(text)})
        total_chars += len(text)
        if page_idx == total_pages - 1 or page_idx % 5 == 0:
            emit_progress(
                progress_callback,
                0.05 + 0.78 * ((page_idx + 1) / max(total_pages, 1)),
                f"解析 PDF 页面 {page_idx + 1}/{total_pages}",
            )

    emit_progress(progress_callback, 0.88, "识别章节结构")
    chapters = detect_chapters_from_pages(page_texts)
    doc.close()
    emit_progress(progress_callback, 1.0, "解析完成")

    return {
        "textbook_id": textbook_id,
        "filename": file_path.name,
        "title": file_path.stem,
        "total_pages": total_pages,
        "total_chars": total_chars,
        "chapters": chapters,
    }


def parse_plain_text(
    file_path: Path,
    textbook_id: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    emit_progress(progress_callback, 0.15, "读取文本文件")
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    emit_progress(progress_callback, 0.55, "清洗文本")
    text = safe_text(raw)
    emit_progress(progress_callback, 0.75, "拆分页片段")
    pseudo_pages = split_text_into_pseudo_pages(text, page_char_size=1800)
    emit_progress(progress_callback, 0.9, "识别章节结构")
    chapters = detect_chapters_from_pages(pseudo_pages)
    emit_progress(progress_callback, 1.0, "解析完成")

    return {
        "textbook_id": textbook_id,
        "filename": file_path.name,
        "title": file_path.stem,
        "total_pages": len(pseudo_pages),
        "total_chars": len(text),
        "chapters": chapters,
    }


def parse_docx(
    file_path: Path,
    textbook_id: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    paragraphs: list[str] = []
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    emit_progress(progress_callback, 0.15, "读取 Word 文档")
    with zipfile.ZipFile(file_path) as docx:
        xml_bytes = docx.read("word/document.xml")

    emit_progress(progress_callback, 0.45, "提取段落文本")
    root = ET.fromstring(xml_bytes)
    for paragraph in root.findall(".//w:p", namespace):
        text_runs = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        paragraph_text = safe_text("".join(text_runs))
        if paragraph_text:
            paragraphs.append(paragraph_text)

    emit_progress(progress_callback, 0.72, "拆分页片段")
    text = safe_text("\n".join(paragraphs))
    pseudo_pages = split_text_into_pseudo_pages(text, page_char_size=1800)
    emit_progress(progress_callback, 0.9, "识别章节结构")
    chapters = detect_chapters_from_pages(pseudo_pages)
    emit_progress(progress_callback, 1.0, "解析完成")

    return {
        "textbook_id": textbook_id,
        "filename": file_path.name,
        "title": file_path.stem,
        "total_pages": len(pseudo_pages),
        "total_chars": len(text),
        "chapters": chapters,
    }


def parse_excel(
    file_path: Path,
    textbook_id: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    emit_progress(progress_callback, 0.1, "打开 Excel 工作簿")
    workbook = pd.ExcelFile(file_path)
    parts: list[str] = []

    for sheet_idx, sheet_name in enumerate(workbook.sheet_names):
        frame = workbook.parse(sheet_name, dtype=str)
        frame = frame.fillna("")
        parts.append(f"# {sheet_name}")
        if not frame.empty:
            parts.append(frame.to_csv(index=False))
        emit_progress(
            progress_callback,
            0.15 + 0.65 * ((sheet_idx + 1) / max(len(workbook.sheet_names), 1)),
            f"解析工作表 {sheet_idx + 1}/{len(workbook.sheet_names)}",
        )

    emit_progress(progress_callback, 0.84, "拆分页片段")
    text = safe_text("\n".join(parts))
    pseudo_pages = split_text_into_pseudo_pages(text, page_char_size=1800)
    emit_progress(progress_callback, 0.94, "识别章节结构")
    chapters = detect_chapters_from_pages(pseudo_pages)
    emit_progress(progress_callback, 1.0, "解析完成")

    return {
        "textbook_id": textbook_id,
        "filename": file_path.name,
        "title": file_path.stem,
        "total_pages": len(pseudo_pages),
        "total_chars": len(text),
        "chapters": chapters,
    }


def split_text_into_pseudo_pages(text: str, page_char_size: int = 1800) -> list[dict[str, Any]]:
    pages = []
    for i in range(0, len(text), page_char_size):
        chunk = text[i : i + page_char_size]
        pages.append({"page": len(pages) + 1, "text": chunk, "char_count": len(chunk)})
    return pages or [{"page": 1, "text": "", "char_count": 0}]


def detect_chapters_from_pages(page_texts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Detect chapters by regex. If no headings are found, fall back to every 5 pages.
    candidates: list[dict[str, Any]] = []

    for page_obj in page_texts:
        page_no = page_obj["page"]
        text = page_obj["text"]
        for line in text.splitlines():
            line = line.strip()
            if is_chapter_heading(line):
                candidates.append({"title": normalize_heading(line), "page": page_no})
                break

    if not candidates:
        return fallback_chapters_by_page_group(page_texts, group_size=5)

    deduped: list[dict[str, Any]] = []
    seen_titles = set()
    for item in candidates:
        key = item["title"]
        if key in seen_titles:
            continue
        deduped.append(item)
        seen_titles.add(key)

    chapters: list[dict[str, Any]] = []
    for idx, item in enumerate(deduped):
        start_page = item["page"]
        end_page = deduped[idx + 1]["page"] - 1 if idx + 1 < len(deduped) else page_texts[-1]["page"]
        content_pages = [p for p in page_texts if start_page <= p["page"] <= end_page]
        content = "\n".join(p["text"] for p in content_pages)
        chapters.append({
            "chapter_id": f"ch_{idx + 1:02d}",
            "title": item["title"],
            "page_start": start_page,
            "page_end": end_page,
            "content": content,
            "char_count": len(content),
        })

    return chapters


def is_chapter_heading(line: str) -> bool:
    if len(line) > 80:
        return False
    for pattern in CHAPTER_PATTERNS:
        if pattern.search(line):
            return True
    if re.match(r"^#{1,3}\s+.+", line):
        return True
    return False


def normalize_heading(line: str) -> str:
    line = re.sub(r"^#{1,6}\s*", "", line).strip()
    line = re.sub(r"\s+", " ", line)
    return line[:80]


def fallback_chapters_by_page_group(page_texts: list[dict[str, Any]], group_size: int = 5) -> list[dict[str, Any]]:
    chapters: list[dict[str, Any]] = []
    for i in range(0, len(page_texts), group_size):
        group = page_texts[i : i + group_size]
        if not group:
            continue
        start_page = group[0]["page"]
        end_page = group[-1]["page"]
        content = "\n".join(p["text"] for p in group)
        chapters.append({
            "chapter_id": f"ch_{len(chapters) + 1:02d}",
            "title": f"自动章节 {len(chapters) + 1}",
            "page_start": start_page,
            "page_end": end_page,
            "content": content,
            "char_count": len(content),
        })
    return chapters
