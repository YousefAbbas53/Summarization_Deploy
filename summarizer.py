import math
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import iter_paragraphs, split_sentences, normalize_text

logger = logging.getLogger(__name__)

# Model config
MODEL_NAME = "facebook/bart-large-cnn"
BATCH_SIZE = 4
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 3
EARLY_STOPPING = True

# Chunking config
MAX_INPUT_TOKENS = 1024
HEADROOM_TOKENS = 16
EFFECTIVE_MAX_INPUT = MAX_INPUT_TOKENS - HEADROOM_TOKENS
OVERLAP_SENTENCES = 2

# Output size caps
CHAPTER_MAX_NEW_TOKENS_CAP = 320
CHAPTER_MIN_NEW_TOKENS_FLOOR = 120
BOOK_PARTS = 8

class BookSummarizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Loads the tokenizer and model into memory."""
        if self.model is not None:
            return
            
        logger.info(f"Loading model {MODEL_NAME} onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(self.device)
        
        if self.device == "cuda":
            try:
                self.model.half()
            except Exception as e:
                logger.warning(f"Could not convert model to fp16: {e}")
                
        self.model.eval()
        logger.info("Model loaded successfully.")

    def tok_len(self, s: str) -> int:
        if not self.tokenizer:
            self.load_model()
        return len(self.tokenizer.encode(s, add_special_tokens=False))

    def split_by_tokens(self, s: str, max_len: int, overlap_tokens: int = 64):
        if not self.tokenizer:
            self.load_model()
        ids = self.tokenizer.encode(s, add_special_tokens=False)
        if len(ids) <= max_len:
            return [s.strip()]
        overlap_tokens = max(0, min(overlap_tokens, max_len // 3))
        step = max(1, max_len - overlap_tokens)
        parts = []
        for i in range(0, len(ids), step):
            chunk_ids = ids[i:i+max_len]
            if not chunk_ids:
                continue
            t = self.tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            if t:
                parts.append(t)
        return parts

    def chunk_text(self, text: str, max_input_tokens: int = EFFECTIVE_MAX_INPUT, overlap_sentences: int = OVERLAP_SENTENCES):
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
                st_tok = self.tok_len(st)

                if st_tok > max_input_tokens:
                    flush()
                    chunks.extend(self.split_by_tokens(st, max_len=max_input_tokens, overlap_tokens=64))
                    continue

                if cur_tok + st_tok <= max_input_tokens:
                    cur_sents.append(st)
                    cur_tok += st_tok
                else:
                    prev = cur_sents[:]
                    flush()
                    overlap = prev[-overlap_sentences:] if overlap_sentences and prev else []
                    cur_sents = overlap + [st]
                    cur_tok = self.tok_len(" ".join(cur_sents))

        flush()
        return chunks

    @torch.no_grad()
    def generate_summaries(self, texts, min_new_tokens, max_new_tokens, batch_size=BATCH_SIZE):
        if not self.model:
            self.load_model()
            
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt",
                truncation=True, padding=True,
                max_length=EFFECTIVE_MAX_INPUT
            ).to(self.device)

            try:
                gen = self.model.generate(
                    **enc,
                    num_beams=NUM_BEAMS,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                    early_stopping=EARLY_STOPPING,
                )
            except TypeError:
                gen = self.model.generate(
                    **enc,
                    num_beams=NUM_BEAMS,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                    min_length=min_new_tokens,
                    max_length=max_new_tokens,
                    early_stopping=EARLY_STOPPING,
                )

            decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            outs.extend([d.strip() for d in decoded])
        return outs

    def summarize_long_text(self, text: str, min_new: int, max_new: int):
        chunks = self.chunk_text(text)
        if not chunks:
            return ""

        chunk_summaries = []
        for ch in chunks:
            tlen = self.tok_len(ch)
            dyn_max = int(min(max_new, max(min_new, round(tlen * 0.18))))
            dyn_min = max(30, min(min_new, dyn_max - 10))
            chunk_summaries.append(self.generate_summaries([ch], dyn_min, dyn_max, batch_size=1)[0])

        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        current = chunk_summaries
        for _ in range(6):
            combined = "\n".join([f"Part {i+1}: {t}" for i, t in enumerate(current)])
            if self.tok_len(combined) <= EFFECTIVE_MAX_INPUT:
                return self.generate_summaries([combined], min_new, max_new, batch_size=1)[0]

            sub_chunks = self.chunk_text(combined, overlap_sentences=1)
            current = self.generate_summaries(
                sub_chunks,
                min_new_tokens=max(60, min_new // 2),
                max_new_tokens=max(180, max_new // 2),
                batch_size=BATCH_SIZE
            )
        return "\n".join(current).strip()

    def make_big_book_summary(self, chapter_summaries, parts=BOOK_PARTS):
        chap_summaries = [s for s in chapter_summaries if s.strip()]
        if not chap_summaries:
            return ""

        n = len(chap_summaries)
        group_size = max(1, math.ceil(n / parts))
        groups = [chap_summaries[i:i+group_size] for i in range(0, n, group_size)]

        part_summaries = []
        for gi, g in enumerate(groups):
            combined = "\n".join([f"ChapterSummary {gi+1}.{i+1}: {t}" for i, t in enumerate(g)])
            ps = self.summarize_long_text(combined, min_new=220, max_new=520)
            part_summaries.append(ps.strip())
        return "\n\n".join(part_summaries)

summarizer = BookSummarizer()
