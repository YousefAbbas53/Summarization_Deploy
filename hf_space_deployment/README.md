---
title: LITVISION Summarization API
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# LITVISION Book Summarization API

A production-ready FastAPI endpoint for the LITVISION Book Summarization Feature. This service accepts PDF or TXT files, extracts text (using native extraction with OCR fallback for scanned pages), chunks the text smartly, and generates both per-chapter summaries and a final organized summary using `facebook/bart-large-cnn`.

It is fully configured for deployment on Hugging Face Spaces (Docker).

## Features

- **Text Extraction:** Native PDF text extraction using `PyMuPDF`.
- **OCR Fallback:** Scans unextractable PDF pages using `pytesseract` (supports English and Arabic).
- **Smart Chunking:** Token-aware sentence grouping to prevent cutting mid-sentence.
- **Generative AI:** Uses `BART-large-CNN` on GPU (or CPU fallback) with FP16 optimization.
- **FastAPI Backend:** Fully async HTTP endpoint for file uploads.
- **Hugging Face Ready:** Pre-configured `Dockerfile` with non-root user and correct port mappings.

## API Endpoints

### `GET /`
Returns basic API information.

### `GET /health`
Returns health status.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### `POST /summarize`
Accepts a PDF or TXT file via `multipart/form-data`.

**Request:**
```bash
curl -X POST -F "file=@book.pdf" http://localhost:7860/summarize
```

**Response Format:**
```json
{
  "success": true,
  "file_name": "book.pdf",
  "num_chapters": 1,
  "chapter_summaries": [
    {
      "chapter": "BOOK",
      "summary": "..."
    }
  ],
  "final_summary": "..."
}
```

## Folder Structure

```
.
├── app.py                # FastAPI endpoints and startup events
├── summarizer.py         # AI generation logic (BART model)
├── utils.py              # PDF extraction, OCR, and chunking tools
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container configuration
├── .dockerignore
├── .gitignore
└── README.md
```

## Local Development

### 1. Install System Dependencies (Linux/macOS)
Make sure you have Tesseract and Poppler installed:
- **Ubuntu:** `sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara`
- **Mac:** `brew install poppler tesseract tesseract-lang`

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Server
```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

## Docker Build & Run (Local)

```bash
docker build -t litvision-summarizer .
docker run -p 7860:7860 --gpus all litvision-summarizer
```
*(Remove `--gpus all` if running on CPU)*

## Deployment to Hugging Face Spaces

1. Go to Hugging Face and create a new Space.
2. Select **Docker** as the Space SDK.
3. Upload all the files in this directory directly to the repository.
4. The space will automatically build the container and start the Uvicorn server on port 7860.

## Troubleshooting

- **CUDA OOM Errors:** Ensure the uploaded book is not excessively long, or adjust the `BATCH_SIZE` in `summarizer.py`.
- **OCR Not Working:** Verify Tesseract language packs (`tesseract-ocr-ara` and `tesseract-ocr-eng`) are correctly installed in your environment.
