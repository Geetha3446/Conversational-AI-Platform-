"""
The RAG layer: turn PDFs into searchable vectors.

Why this talks to FAISS directly
--------------------------------
The `langchain-community` FAISS wrapper is being sunset upstream, and it
persists its docstore as a pickle that has to be loaded back with
`allow_dangerous_deserialization=True`. Driving FAISS ourselves is about eighty
lines, drops a deprecated dependency, and stores chunk metadata as plain JSON
you can open and read.

Layout on disk, per user:

    data/vectorstores/user_<id>/index.faiss   the vectors
    data/vectorstores/user_<id>/meta.json     chunk text, source file, page

Design notes
------------
* One index per user. A user can never retrieve someone else's documents,
  because the index is chosen from their authenticated id, never from anything
  they send in a request.
* Embeddings are L2-normalised and the index is inner-product, which makes the
  returned score exactly cosine similarity.
* The embedding model and the loaded indexes are cached in memory behind locks,
  because FastAPI serves requests from several threads.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from backend.config import settings


@dataclass
class Chunk:
    """One retrievable piece of a PDF."""

    text: str
    source: str  # original filename
    page: int
    score: float = 0.0


# ---------------------------------------------------------------------------
# Embedding model (loaded lazily, exactly once, thread-safely)
# ---------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()

# In-memory cache of loaded indexes: {user_id: (faiss.Index, [Chunk, ...])}
_index_cache: dict = {}
_index_lock = threading.Lock()


def get_model():
    """Return the shared SentenceTransformer, downloading it on first use."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # double-checked locking
                from sentence_transformers import SentenceTransformer

                print(f"[vectorstore] Loading embedding model {settings.EMBEDDING_MODEL}")
                _model = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")
    return _model


def embed(texts: List[str]) -> np.ndarray:
    """Encode texts into a float32 matrix of unit-length vectors."""
    vectors = get_model().encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,  # makes inner product identical to cosine
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


# ---------------------------------------------------------------------------
# PDF -> chunks
# ---------------------------------------------------------------------------
def pdf_to_chunks(pdf_path: Path, source_name: str) -> Tuple[List[Chunk], int]:
    """
    Read a PDF with pypdf and split it into overlapping chunks.

    Returns (chunks, page_count). Pages with no extractable text, for example
    pure scans, are skipped rather than producing empty chunks.
    """
    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Chunk] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""  # one unreadable page should not fail the whole upload
        if not text:
            continue
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(text=piece, source=source_name, page=page_number))

    return chunks, page_count


# ---------------------------------------------------------------------------
# Index lifecycle
# ---------------------------------------------------------------------------
def _paths(user_id: int) -> Tuple[Path, Path]:
    directory = settings.user_index_dir(user_id)
    return directory / "index.faiss", directory / "meta.json"


def has_documents(user_id: int) -> bool:
    index_file, meta_file = _paths(user_id)
    return index_file.exists() and meta_file.exists()


def _load(user_id: int):
    """Load a user's index from disk into the cache. None if they have no docs."""
    with _index_lock:
        if user_id in _index_cache:
            return _index_cache[user_id]

    if not has_documents(user_id):
        return None

    import faiss

    index_file, meta_file = _paths(user_id)
    try:
        index = faiss.read_index(str(index_file))
        rows = json.loads(meta_file.read_text(encoding="utf-8"))
        chunks = [Chunk(text=r["text"], source=r["source"], page=r["page"]) for r in rows]
    except Exception as exc:  # a corrupt index must not take the server down
        print(f"[vectorstore] Could not load index for user {user_id}: {exc}")
        return None

    with _index_lock:
        _index_cache[user_id] = (index, chunks)
    return index, chunks


def _save(user_id: int, index, chunks: List[Chunk]) -> None:
    import faiss

    index_file, meta_file = _paths(user_id)
    faiss.write_index(index, str(index_file))
    meta_file.write_text(
        json.dumps(
            [{"text": c.text, "source": c.source, "page": c.page} for c in chunks],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with _index_lock:
        _index_cache[user_id] = (index, chunks)


def add_chunks(user_id: int, new_chunks: List[Chunk]) -> int:
    """
    Append chunks to a user's index, creating it if this is their first upload.
    Returns the number of chunks written.
    """
    if not new_chunks:
        return 0

    import faiss

    vectors = embed([c.text for c in new_chunks])
    existing = _load(user_id)

    if existing is None:
        index = faiss.IndexFlatIP(vectors.shape[1])
        chunks: List[Chunk] = []
    else:
        index, chunks = existing

    index.add(vectors)
    chunks = chunks + new_chunks
    _save(user_id, index, chunks)
    return len(new_chunks)


def rebuild_index_excluding(user_id: int, filename: str) -> None:
    """
    FAISS has no cheap "delete by metadata", so when a document is removed we
    rebuild from the PDFs that remain on disk. Fine at this scale.
    """
    with _index_lock:
        _index_cache.pop(user_id, None)
    for path in _paths(user_id):
        path.unlink(missing_ok=True)

    remaining = [
        p for p in settings.user_upload_dir(user_id).glob("*.pdf") if p.name != filename
    ]
    all_chunks: List[Chunk] = []
    for pdf in remaining:
        chunks, _ = pdf_to_chunks(pdf, pdf.name)
        all_chunks.extend(chunks)

    if all_chunks:
        add_chunks(user_id, all_chunks)


def search(user_id: int, query: str, k: Optional[int] = None) -> List[Chunk]:
    """Cosine-similarity search over the user's own documents."""
    loaded = _load(user_id)
    if loaded is None:
        return []

    index, chunks = loaded
    if not chunks:
        return []

    top_k = min(k or settings.RETRIEVER_K, len(chunks))
    scores, ids = index.search(embed([query]), top_k)

    results: List[Chunk] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:  # FAISS pads with -1 when fewer than k results exist
            continue
        hit = chunks[int(idx)]
        results.append(
            Chunk(text=hit.text, source=hit.source, page=hit.page, score=float(score))
        )
    return results
