"""
Offline bulk ingestion.

Most of the time PDFs arrive through the Streamlit uploader, but it is handy to
be able to seed a user's knowledge base from a folder, for example when
demonstrating the app.

Usage (from the project root, with the venv active):

    python -m backend.rag.ingest --username srinath --folder ./my_pdfs

Every PDF in the folder is copied into that user's private upload directory,
chunked, embedded and registered in the documents table.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from backend.config import settings
from backend.database import SessionLocal, init_db
from backend.models import Document, User
from backend.rag import vectorstore


def ingest_folder(username: str, folder: Path) -> None:
    init_db()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            print(f"No user named '{username}'. Register in the app first.")
            sys.exit(1)

        pdfs = sorted(folder.glob("*.pdf"))
        if not pdfs:
            print(f"No PDF files found in {folder}")
            sys.exit(1)

        target_dir = settings.user_upload_dir(user.id)

        for pdf in pdfs:
            existing = (
                db.query(Document)
                .filter(Document.user_id == user.id, Document.filename == pdf.name)
                .first()
            )
            if existing:
                print(f"  skip   {pdf.name} (already ingested)")
                continue

            destination = target_dir / pdf.name
            shutil.copy2(pdf, destination)

            chunks, pages = vectorstore.pdf_to_chunks(destination, pdf.name)
            written = vectorstore.add_chunks(user.id, chunks)

            db.add(
                Document(
                    user_id=user.id,
                    filename=pdf.name,
                    num_pages=pages,
                    num_chunks=written,
                    size_bytes=destination.stat().st_size,
                )
            )
            db.commit()
            print(f"  ok     {pdf.name}  pages={pages}  chunks={written}")

        print("\nIngestion complete.")
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-ingest PDFs for a user.")
    parser.add_argument("--username", required=True, help="Existing account username")
    parser.add_argument("--folder", required=True, type=Path, help="Folder of PDFs")
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"{args.folder} is not a directory")
        sys.exit(1)

    ingest_folder(args.username, args.folder)


if __name__ == "__main__":
    main()
