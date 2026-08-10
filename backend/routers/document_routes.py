"""
/documents endpoints: upload a PDF, list what has been uploaded, delete one.

Uploading is the only heavy request in the app: it reads the PDF, chunks it,
embeds every chunk locally and writes a FAISS index. For a 30 page document
that is a couple of seconds on CPU.
"""

from __future__ import annotations

import re
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.agent.graph import invalidate_graph_cache
from backend.config import settings
from backend.database import get_db
from backend.models import Document, User
from backend.rag import vectorstore
from backend.schemas import DocumentOut, SimpleMessage
from backend.security import get_current_user

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitise(filename: str) -> str:
    """Strip any path components and unusual characters before touching disk."""
    base = filename.replace("\\", "/").split("/")[-1]
    cleaned = _SAFE_NAME.sub("_", base).strip("._") or "document.pdf"
    return cleaned if cleaned.lower().endswith(".pdf") else cleaned + ".pdf"


@router.get("", response_model=List[DocumentOut])
def list_documents(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> List[Document]:
    return (
        db.query(Document)
        .filter(Document.user_id == user.id)
        .order_by(Document.created_at.desc())
        .all()
    )


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Document:
    """Accept a PDF, embed it, and make it searchable by the agent."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    raw = file.file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="That file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    safe_name = _sanitise(file.filename)
    already = (
        db.query(Document)
        .filter(Document.user_id == user.id, Document.filename == safe_name)
        .first()
    )
    if already:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"'{safe_name}' has already been uploaded.",
        )

    destination = settings.user_upload_dir(user.id) / safe_name
    destination.write_bytes(raw)

    try:
        chunks, pages = vectorstore.pdf_to_chunks(destination, safe_name)
        if not chunks:
            destination.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "No text could be extracted. This looks like a scanned PDF; "
                    "run OCR on it first."
                ),
            )
        written = vectorstore.add_chunks(user.id, chunks)
    except HTTPException:
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {exc}")

    record = Document(
        user_id=user.id,
        filename=safe_name,
        num_pages=pages,
        num_chunks=written,
        size_bytes=len(raw),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    # Force the agent to rebuild its retrieval tool against the new index.
    invalidate_graph_cache(user.id)
    return record


@router.delete("/{document_id}", response_model=SimpleMessage)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SimpleMessage:
    """Remove the file, drop its rows, and rebuild the index without it."""
    doc = db.get(Document, document_id)
    if doc is None or doc.user_id != user.id:
        raise HTTPException(status_code=404, detail="Document not found.")

    filename = doc.filename
    (settings.user_upload_dir(user.id) / filename).unlink(missing_ok=True)
    db.delete(doc)
    db.commit()

    try:
        vectorstore.rebuild_index_excluding(user.id, filename)
    except Exception as exc:
        print(f"[documents] Index rebuild failed: {exc}")

    invalidate_graph_cache(user.id)
    return SimpleMessage(detail=f"Deleted '{filename}'.")
