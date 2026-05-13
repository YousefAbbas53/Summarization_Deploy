import os
import tempfile
import logging
import asyncio
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

from utils import extract_text_from_file, split_into_chapters
from summarizer import summarizer, CHAPTER_MIN_NEW_TOKENS_FLOOR, CHAPTER_MAX_NEW_TOKENS_CAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LITVISION Book Summarization API",
    description="Extracts text from PDFs/TXTs, chunks, and generates chapter/final summaries.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def root():
    return {
        "api": "LITVISION Book Summarization API",
        "status": "online",
        "endpoints": ["/health", "/summarize"]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": summarizer.model is not None,
        "device": summarizer.device
    }

class ChapterSummary(BaseModel):
    chapter: str
    summary: str

class SummarizationResponse(BaseModel):
    success: bool
    file_name: str
    num_chapters: int
    chapter_summaries: List[ChapterSummary]
    final_summary: str

def remove_temp_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted temp file: {path}")
    except Exception as e:
        logger.error(f"Error deleting temp file {path}: {e}")

@app.post("/summarize", response_model=SummarizationResponse)
async def summarize_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .pdf and .txt are supported.")

    temp_file_path = ""
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty file provided.")
            if len(content) > 50 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="File too large. Max size is 50MB.")
            temp_file.write(content)
            temp_file_path = temp_file.name

        logger.info(f"Extracting text from {file.filename}...")
        text = await asyncio.to_thread(extract_text_from_file, temp_file_path)
        
        if not text.strip():
            raise HTTPException(status_code=422, detail="Could not extract any text from the file.")

        logger.info("Splitting into chapters...")
        chapters = split_into_chapters(text)
        
        chapter_summaries_result = []
        raw_chapter_summaries = []

        logger.info(f"Generating summaries for {len(chapters)} chapters...")
        for title, body in chapters:
            if not body.strip():
                continue
            
            summ = await asyncio.to_thread(
                summarizer.summarize_long_text,
                body,
                CHAPTER_MIN_NEW_TOKENS_FLOOR,
                CHAPTER_MAX_NEW_TOKENS_CAP
            )
            raw_chapter_summaries.append(summ)
            chapter_summaries_result.append(ChapterSummary(chapter=title, summary=summ))

        logger.info("Generating final organized summary...")
        final_summary = await asyncio.to_thread(summarizer.make_big_book_summary, raw_chapter_summaries)

        return SummarizationResponse(
            success=True,
            file_name=file.filename,
            num_chapters=len(chapter_summaries_result),
            chapter_summaries=chapter_summaries_result,
            final_summary=final_summary
        )

    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        error_msg = str(e).lower()
        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in error_msg:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise HTTPException(status_code=500, detail="CUDA out of memory. Try a smaller file.")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path:
            background_tasks.add_task(remove_temp_file, temp_file_path)
