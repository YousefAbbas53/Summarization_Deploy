import os
import re
from pathlib import Path
from typing import List, Tuple

import fitz  # pymupdf
from pdf2image import convert_from_path
import pytesseract
from PIL import ImageOps, ImageEnhance

OCR_LANG = "eng+ara"
OCR_DPI = 180
NATIVE_MIN_CHARS_PER_PAGE = 60  # if native extracted text < this => OCR that page

_SENT_BOUNDARY_RE = re.compile(r"(?<=[\.\!\?\u061F\u06D4\u061B…])\s+")  # . ! ? ؟ ۔ ؛ …

def normalize_text(text: str) -> str:
    """Normalizes text by removing excessive whitespace and fixing newlines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def ocr_image_pil(img):
    """Applies light preprocessing to improve OCR accuracy."""
    img = img.convert("RGB")
    img = ImageOps.grayscale(img)
    img = ImageEnhance.Contrast(img).enhance(1.6)
    return img

def ocr_pdf_page(pdf_path: str, page_number_1based: int, dpi: int = OCR_DPI, lang: str = OCR_LANG) -> str:
    """OCRs a single PDF page."""
    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=page_number_1based,
        last_page=page_number_1based,
        fmt="png",
        thread_count=2,
    )
    if not images:
        return ""
    img = images[0]
    img = ocr_image_pil(img)
    return pytesseract.image_to_string(img, lang=lang)

def pdf_to_text_smart(pdf_path: str, native_min_chars_per_page: int = NATIVE_MIN_CHARS_PER_PAGE) -> str:
    """Extracts text from PDF, falling back to OCR for scanned pages."""
    doc = fitz.open(str(pdf_path))
    parts = []

    for i in range(doc.page_count):
        page = doc.load_page(i)
        native = (page.get_text("text") or "").strip()
        native_compact_len = len(re.sub(r"\s+", "", native))

        if native_compact_len >= native_min_chars_per_page:
            parts.append(native)
        else:
            ocr = ocr_pdf_page(pdf_path, page_number_1based=i+1)
            parts.append(ocr)

    doc.close()
    return normalize_text("\n\n".join(parts))

def extract_text_from_file(file_path: str) -> str:
    """Extracts text from a .txt or .pdf file."""
    path = Path(file_path)
    suf = path.suffix.lower()

    if suf == ".txt":
        raw = path.read_text(encoding="utf-8", errors="ignore")
        return normalize_text(raw)
    
    if suf == ".pdf":
        return pdf_to_text_smart(str(path))
        
    raise ValueError(f"Unsupported file type '{suf}'. Please upload .pdf or .txt only.")

def split_into_chapters(text: str) -> List[Tuple[str, str]]:
    """
    Best effort chapter split:
    - Detect lines that look like: CHAPTER 1 / Chapter One / CHAPTER ONE etc.
    - If not found, return one chapter = full text.
    Returns: list of (title, body)
    """
    text = normalize_text(text)
    lines = text.splitlines()

    chapter_re = re.compile(r"^\s*(chapter|CHAPTER)\s+([0-9]+|[IVXLC]+|[A-Za-z]+)\b.*$", re.IGNORECASE)

    idxs = []
    titles = []
    for i, ln in enumerate(lines):
        if chapter_re.match(ln.strip()):
            idxs.append(i)
            titles.append(ln.strip())

    if len(idxs) < 2:
        return [("BOOK", text)]

    chapters = []
    for k in range(len(idxs)):
        start = idxs[k]
        end = idxs[k+1] if k+1 < len(idxs) else len(lines)
        title = titles[k]
        body = "\n".join(lines[start:end]).strip()
        chapters.append((title, body))
    return chapters

def split_sentences(paragraph: str) -> List[str]:
    """Splits a paragraph into sentences."""
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    if not any(ch in paragraph for ch in ".!?\u061F\u06D4\u061B…"):
        ls = [ln.strip() for ln in paragraph.split("\n") if ln.strip()]
        return ls if ls else [paragraph]
    return [s.strip() for s in _SENT_BOUNDARY_RE.split(paragraph) if s.strip()]

def iter_paragraphs(text: str):
    """Yields paragraphs from text."""
    for p in re.split(r"\n\s*\n+", text):
        p = p.strip()
        if p:
            yield p
