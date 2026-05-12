# =========================
# Cell 1 — Install deps (Colab)
# =========================
!apt-get -qq update
!apt-get -qq install -y poppler-utils tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara
!pip -q install -U transformers accelerate sentencepiece pymupdf pdf2image pytesseract pillow tqdm
# =========================
# Cell 2 — Imports + Config
# =========================
import os, re, json
from pathlib import Path
from math import ceil

import torch
from tqdm.auto import tqdm

import fitz  # pymupdf
from pdf2image import convert_from_path
import pytesseract
from PIL import ImageOps, ImageEnhance

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

OUTPUT_DIR = Path("/content/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model (English-focused)
MODEL_NAME = "facebook/bart-large-cnn"  # https://huggingface.co/facebook/bart-large-cnn

# OCR
OCR_LANG = "eng+ara"
OCR_DPI = 250
NATIVE_MIN_CHARS_PER_PAGE = 60  # if native extracted text < this => OCR that page

# Summarization quality/speed knobs
BATCH_SIZE = 4
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 3
EARLY_STOPPING = False

# Chunking
MAX_INPUT_TOKENS = 1024
HEADROOM_TOKENS = 16
EFFECTIVE_MAX_INPUT = MAX_INPUT_TOKENS - HEADROOM_TOKENS
OVERLAP_SENTENCES = 2

# Output size (big + محترم)
CHAPTER_MAX_NEW_TOKENS_CAP = 320   # max tokens generated per chapter summary
CHAPTER_MIN_NEW_TOKENS_FLOOR = 120
BOOK_PARTS = 8  # final organized "big" summary in N parts

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print("Output folder:", OUTPUT_DIR)
# =========================
# Cell 3 — Upload input (PDF or TXT)
# =========================
from google.colab import files

uploaded = files.upload()
INPUT_PATH = Path(next(iter(uploaded.keys()))).resolve()

print("Uploaded:", INPUT_PATH)
print("Suffix:", INPUT_PATH.suffix.lower())
# =========================
# Cell 4 — PDF/TXT -> Clean TXT (robust native + per-page OCR fallback)
# =========================
_SENT_BOUNDARY_RE = re.compile(r"(?<=[\.\!\?\u061F\u06D4\u061B…])\s+")  # . ! ? ؟ ۔ ؛ …

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def ocr_image_pil(img):
    # Light preprocessing to improve OCR
    img = img.convert("RGB")
    img = ImageOps.grayscale(img)
    img = ImageEnhance.Contrast(img).enhance(1.6)
    return img

def ocr_pdf_page(pdf_path: Path, page_number_1based: int, dpi: int = OCR_DPI, lang: str = OCR_LANG) -> str:
    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=page_number_1based,
        last_page=page_number_1based,
        fmt="png",
        thread_count=2,
    )
    img = images[0]
    img = ocr_image_pil(img)
    return pytesseract.image_to_string(img, lang=lang)

def pdf_to_text_smart(pdf_path: Path,
                     native_min_chars_per_page: int = NATIVE_MIN_CHARS_PER_PAGE) -> str:
    doc = fitz.open(str(pdf_path))
    parts = []

    for i in tqdm(range(doc.page_count), desc="Extracting pages"):
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

def ensure_txt(input_path: Path) -> Path:
    out_txt = OUTPUT_DIR / f"{input_path.stem}.txt"
    suf = input_path.suffix.lower()

    if suf == ".txt":
        raw = input_path.read_text(encoding="utf-8", errors="ignore")
        out_txt.write_text(normalize_text(raw), encoding="utf-8")
        return out_txt

    if suf == ".pdf":
        text = pdf_to_text_smart(input_path)
        out_txt.write_text(text, encoding="utf-8")
        return out_txt

    raise ValueError("Unsupported type. Upload .pdf or .txt only.")

BOOK_TXT_PATH = ensure_txt(INPUT_PATH)
BOOK_TEXT = BOOK_TXT_PATH.read_text(encoding="utf-8", errors="ignore")

print("Saved TXT:", BOOK_TXT_PATH)
print("Chars:", len(BOOK_TEXT))
print("Head preview:\n", BOOK_TEXT[:800])
# FIX Pillow broken install (PIL._typing/_Ink issue)
!pip -q uninstall -y Pillow pillow-simd
!pip -q install --no-cache-dir --force-reinstall "Pillow==10.4.0"

import PIL, sys
print("Pillow version:", PIL.__version__)
print("Python:", sys.version)
# =========================
# Cell 5 — Load tokenizer + model (from Hugging Face)
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

if device == "cuda":
    try:
        model.half()
    except Exception:
        pass

torch.set_grad_enabled(False)
print("Model loaded:", MODEL_NAME)
# =========================
# Cell 6 — Chapter splitting + token-aware chunking
# =========================
def split_into_chapters(text: str):
    """
    Best effort chapter split:
    - Detect lines that look like: CHAPTER 1 / Chapter One / CHAPTER ONE etc.
    - If not found, return one chapter = full text.
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

def split_sentences(paragraph: str):
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    if not any(ch in paragraph for ch in ".!?\u061F\u06D4\u061B…"):
        ls = [ln.strip() for ln in paragraph.split("\n") if ln.strip()]
        return ls if ls else [paragraph]
    return [s.strip() for s in _SENT_BOUNDARY_RE.split(paragraph) if s.strip()]

def iter_paragraphs(text: str):
    for p in re.split(r"\n\s*\n+", text):
        p = p.strip()
        if p:
            yield p

def tok_len(s: str) -> int:
    return len(tokenizer.encode(s, add_special_tokens=False))

def split_by_tokens(s: str, max_len: int, overlap_tokens: int = 64):
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) <= max_len:
        return [s.strip()]
    overlap_tokens = max(0, min(overlap_tokens, max_len // 3))
    step = max(1, max_len - overlap_tokens)
    parts = []
    for i in range(0, len(ids), step):
        chunk_ids = ids[i:i+max_len]
        if not chunk_ids:
            continue
        t = tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        if t:
            parts.append(t)
    return parts

def chunk_text(text: str, max_input_tokens: int = EFFECTIVE_MAX_INPUT, overlap_sentences: int = OVERLAP_SENTENCES):
    """
    Professional chunking:
    - pack sentences under token limit
    - add sentence overlap between chunks for continuity
    - if a single sentence is too long => token-split it
    """
    text = normalize_text(text)
    if not text:
        return []

    chunks = []
    cur_sents, cur_tok = [], 0

    def flush():
        nonlocal cur_sents, cur_tok
        if cur_sents:
            ch = " ".join(cur_sents).strip()
            if ch:
                chunks.append(ch)
        cur_sents, cur_tok = [], 0

    for para in iter_paragraphs(text):
        for sent in split_sentences(para):
            st = sent.strip()
            if not st:
                continue
            st_tok = tok_len(st)

            if st_tok > max_input_tokens:
                flush()
                chunks.extend(split_by_tokens(st, max_len=max_input_tokens, overlap_tokens=64))
                continue

            if cur_tok + st_tok <= max_input_tokens:
                cur_sents.append(st)
                cur_tok += st_tok
            else:
                prev = cur_sents[:]
                flush()
                overlap = prev[-overlap_sentences:] if overlap_sentences and prev else []
                cur_sents = overlap + [st]
                cur_tok = tok_len(" ".join(cur_sents))

    flush()
    return chunks
# =========================
# Cell 7 — Summarization helpers (map -> reduce) + "organized big summary"
# =========================
@torch.no_grad()
def generate_summaries(texts, min_new_tokens, max_new_tokens, batch_size=BATCH_SIZE):
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch, return_tensors="pt",
            truncation=True, padding=True,
            max_length=EFFECTIVE_MAX_INPUT
        ).to(device)

        try:
            gen = model.generate(
                **enc,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                early_stopping=EARLY_STOPPING,
            )
        except TypeError:
            # fallback for older transformers
            gen = model.generate(
                **enc,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                min_length=min_new_tokens,
                max_length=max_new_tokens,
                early_stopping=EARLY_STOPPING,
            )

        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        outs.extend([d.strip() for d in decoded])
    return outs

def summarize_long_text(text: str, min_new: int, max_new: int):
    """
    Summarize very long text reliably:
    - chunk -> summarize each chunk
    - if multiple chunk summaries, reduce them into one (still ordered)
    """
    chunks = chunk_text(text)
    if not chunks:
        return ""

    # summarize chunks
    chunk_summaries = []
    for ch in chunks:
        tlen = tok_len(ch)
        # dynamic summary size per chunk (keeps it detailed)
        dyn_max = int(min(max_new, max(min_new, round(tlen * 0.18))))
        dyn_min = max(30, min(min_new, dyn_max - 10))
        chunk_summaries.append(generate_summaries([ch], dyn_min, dyn_max, batch_size=1)[0])

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    # reduce in groups (keeps order)
    current = chunk_summaries
    for _ in range(6):
        combined = "\n".join([f"Part {i+1}: {t}" for i, t in enumerate(current)])
        if tok_len(combined) <= EFFECTIVE_MAX_INPUT:
            return generate_summaries([combined], min_new, max_new, batch_size=1)[0]

        # too long -> chunk combined summaries and summarize each chunk
        sub_chunks = chunk_text(combined, overlap_sentences=1)
        current = generate_summaries(
            sub_chunks,
            min_new_tokens=max(60, min_new // 2),
            max_new_tokens=max(180, max_new // 2),
            batch_size=BATCH_SIZE
        )
    return "\n".join(current).strip()

def make_big_book_summary(chapter_summaries, parts=BOOK_PARTS):
    """
    Organized "big" summary:
    - group chapter summaries into N parts
    - summarize each group into a longer part-summary
    - output stays structured and chronological
    """
    chap_summaries = [s for s in chapter_summaries if s.strip()]
    if not chap_summaries:
        return []

    n = len(chap_summaries)
    group_size = max(1, ceil(n / parts))
    groups = [chap_summaries[i:i+group_size] for i in range(0, n, group_size)]

    part_summaries = []
    for gi, g in enumerate(tqdm(groups, desc="Building big organized summary")):
        combined = "\n".join([f"ChapterSummary {gi+1}.{i+1}: {t}" for i, t in enumerate(g)])
        ps = summarize_long_text(combined, min_new=220, max_new=520)
        part_summaries.append(ps.strip())
    return part_summaries
# =========================
# Cell 8 — RUN: chapter summaries + big organized summary + save all outputs
# =========================
chapters = split_into_chapters(BOOK_TEXT)
print("Detected chapters:", len(chapters))
print("First chapter title:", chapters[0][0])

# Save chapters as separate txt files (for debugging)
chapters_dir = OUTPUT_DIR / f"{BOOK_TXT_PATH.stem}_chapters"
chapters_dir.mkdir(parents=True, exist_ok=True)

chapter_summaries = []
chapter_meta = []

for idx, (title, body) in enumerate(tqdm(chapters, desc="Summarizing chapters")):
    safe_title = re.sub(r"[^A-Za-z0-9 _-]+", "", title)[:80].strip().replace(" ", "_")
    ch_txt_path = chapters_dir / f"{idx+1:03d}_{safe_title or 'CHAPTER'}.txt"
    ch_txt_path.write_text(body, encoding="utf-8")

    # chapter summary (detailed)
    # (إذا الفصل طويل جدًا summarize_long_text هيعمل chunking داخليًا)
    summary = summarize_long_text(
        body,
        min_new=CHAPTER_MIN_NEW_TOKENS_FLOOR,
        max_new=CHAPTER_MAX_NEW_TOKENS_CAP
    )

    chapter_summaries.append(summary)
    chapter_meta.append({"index": idx+1, "title": title, "txt_path": str(ch_txt_path)})

# 1) Save per-chapter summaries (organized)
chapter_summaries_path = OUTPUT_DIR / f"{BOOK_TXT_PATH.stem}.chapter_summaries.txt"
with chapter_summaries_path.open("w", encoding="utf-8") as f:
    for i, (meta, summ) in enumerate(zip(chapter_meta, chapter_summaries), start=1):
        f.write(f"===== CHAPTER {i}: {meta['title']} =====\n")
        f.write(summ.strip() + "\n\n")

# 2) Save "big organized book summary" (multi-part, محترم وكبير)
big_parts = make_big_book_summary(chapter_summaries, parts=BOOK_PARTS)
big_summary_path = OUTPUT_DIR / f"{BOOK_TXT_PATH.stem}.BIG_book_summary_parts.txt"
big_summary_path.write_text(
    "\n\n".join([f"=== BOOK SUMMARY PART {i+1} ===\n{p}" for i, p in enumerate(big_parts)]),
    encoding="utf-8"
)

# 3) Also save a single-file "full" summary by concatenating chapter summaries (very long, but super clear)
full_concat_path = OUTPUT_DIR / f"{BOOK_TXT_PATH.stem}.FULL_chapter_summaries_concat.txt"
full_concat_path.write_text("\n\n".join(chapter_summaries), encoding="utf-8")

# 4) Metadata
meta_path = OUTPUT_DIR / f"{BOOK_TXT_PATH.stem}.meta.json"
meta_path.write_text(json.dumps({
    "input_file": str(INPUT_PATH),
    "book_txt": str(BOOK_TXT_PATH),
    "model": MODEL_NAME,
    "device": device,
    "chapters_detected": len(chapters),
    "chapter_files_dir": str(chapters_dir),
    "outputs": {
        "chapter_summaries": str(chapter_summaries_path),
        "big_book_summary_parts": str(big_summary_path),
        "full_concat": str(full_concat_path),
    }
}, ensure_ascii=False, indent=2), encoding="utf-8")

print("\nSaved outputs:")
print(" - Chapter summaries:", chapter_summaries_path)
print(" - BIG organized parts:", big_summary_path)
print(" - FULL concat:", full_concat_path)
print(" - Meta:", meta_path)

print("\nPreview BIG summary part 1:\n")
print(big_parts[0][:1500] if big_parts else "N/A")
# =========================
# Cell 9 — Save model + zip outputs + download
# =========================
saved_model_dir = OUTPUT_DIR / "saved_model_bart_large_cnn"
saved_model_dir.mkdir(parents=True, exist_ok=True)

model.save_pretrained(saved_model_dir)
tokenizer.save_pretrained(saved_model_dir)

print("Model saved to:", saved_model_dir)

zip_path = Path("/content/litvision_output.zip")
!zip -qr "{zip_path}" "{OUTPUT_DIR}"
print("Zipped to:", zip_path)

from google.colab import files
files.download(str(zip_path))
