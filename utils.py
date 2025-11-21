from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np
from docx import Document
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline

# -------------------------
# Data structures
# -------------------------

@dataclass
class IndexedChunk:
    text: str
    doc_id: str
    chunk_id: int


# -------------------------
# Extraction
# -------------------------

def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        doc = Document(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"Extension non supportée: {suffix}")


# -------------------------
# Chunking
# -------------------------

def build_tokenizer(model_name: str = "bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)


def chunk_text(text: str, tokenizer=None, chunk_size: int = 250, overlap: int = 30) -> List[str]:
    tokenizer = tokenizer or build_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(tokens), step):
        piece = tokens[start : start + chunk_size]
        if not piece:
            continue
        decoded = tokenizer.decode(piece, skip_special_tokens=True)
        if decoded.strip():
            chunks.append(decoded.strip())
    return chunks


# -------------------------
# Embeddings
# -------------------------

def build_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name, device="cpu")


def encode_chunks(chunks: Sequence[str], embedder=None, batch_size: int = 4) -> np.ndarray:
    embedder = embedder or build_embedder()
    embeddings = embedder.encode(
        list(chunks),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.astype("float32")


# -------------------------
# Index
# -------------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    if embeddings.ndim != 2:
        raise ValueError("Les embeddings doivent être de forme (n, dim)")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatL2, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.IndexFlatL2:
    return faiss.read_index(str(path))


# -------------------------
# Recherche
# -------------------------

def search(embeddings: np.ndarray, index: faiss.IndexFlatL2, top_k: int = 5):
    scores, idxs = index.search(embeddings, top_k)
    return scores, idxs


# -------------------------
# Résumé
# -------------------------

def build_summarizer(model_name: str = "t5-small"):
    return pipeline(
        "summarization",
        model=model_name,
        device=-1,
        truncation=True,
    )


def summarize_chunks(chunks: Sequence[IndexedChunk], summarizer=None, max_length: int = 200, min_length: int = 30) -> str:
    summarizer = summarizer or build_summarizer()
    merged = "\n".join(chunk.text for chunk in chunks)
    summary = summarizer(
        merged,
        max_length=max_length,
        min_length=min_length,
        no_repeat_ngram_size=3,
        truncation=True,
    )
    return summary[0]["summary_text"].strip()


# -------------------------
# Évaluation (optionnelle)
# -------------------------

def precision_recall_at_k(relevant_ids: Sequence[int], retrieved_ids: Sequence[int], k: int = 5):
    relevant_set = set(relevant_ids)
    retrieved_top_k = retrieved_ids[:k]
    true_positive = len([rid for rid in retrieved_top_k if rid in relevant_set])
    precision = true_positive / max(k, 1)
    recall = true_positive / max(len(relevant_set), 1)
    return precision, recall


# -------------------------
# Persist metadata
# -------------------------

def save_metadata(chunks: Sequence[IndexedChunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([chunk.__dict__ for chunk in chunks], f, ensure_ascii=False, indent=2)


def load_metadata(path: Path) -> List[IndexedChunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [IndexedChunk(**item) for item in data]
