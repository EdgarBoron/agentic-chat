"""Rerunnable ingestion of the prompt-engineering reference library.

Usage (inside the backend container / image):
    python -m app.ingest.ingest_reference

Chunk IDs are derived from (relative path, chunk index, chunk content) so
re-running after editing a file upserts changed chunks instead of
duplicating unchanged ones.
"""

import hashlib
import pathlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.memory.chroma_client import get_reference_collection


def chunk_id(rel_path: pathlib.Path, idx: int, text: str) -> str:
    digest = hashlib.sha256(f"{rel_path}:{idx}:{text}".encode()).hexdigest()[:16]
    return f"{rel_path.as_posix()}-{idx}-{digest}"


def main() -> None:
    root = pathlib.Path(settings.reference_data_dir)
    if not root.exists():
        print(f"Reference directory {root} does not exist, nothing to ingest.")
        return

    coll = get_reference_collection(settings.chroma_persist_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    for path in sorted(list(root.rglob("*.md")) + list(root.rglob("*.txt"))):
        rel_path = path.relative_to(root)
        text = path.read_text(encoding="utf-8", errors="ignore")
        for i, chunk in enumerate(splitter.split_text(text)):
            ids.append(chunk_id(rel_path, i, chunk))
            docs.append(chunk)
            metas.append({"source": rel_path.as_posix()})

    if ids:
        coll.upsert(ids=ids, documents=docs, metadatas=metas)

    print(f"Ingested {len(ids)} chunks from {root}")


if __name__ == "__main__":
    main()
