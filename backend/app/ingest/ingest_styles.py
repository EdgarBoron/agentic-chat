"""Rerunnable ingestion of the artist/photographer style catalog into its
own Chroma collection, kept separate from the general reference library so
`search_artist_styles` returns named-style entries only.

Usage (inside the backend container / image):
    python -m app.ingest.ingest_styles

Chunk IDs are derived from (relative path, chunk index, chunk content) so
re-running after editing the file upserts changed chunks instead of
duplicating unchanged ones.
"""

import pathlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.ingest.chunking import chunk_id, split_document
from app.memory.chroma_client import get_style_collection


def main() -> None:
    path = pathlib.Path(settings.artist_styles_file)
    if not path.exists():
        print(f"Styles file {path} does not exist, nothing to ingest.")
        return

    coll = get_style_collection(settings.chroma_persist_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    rel_path = path.name
    text = path.read_text(encoding="utf-8", errors="ignore")
    for i, chunk in enumerate(split_document(text, splitter)):
        ids.append(chunk_id(pathlib.Path(rel_path), i, chunk))
        docs.append(chunk)
        metas.append({"source": rel_path})

    if ids:
        coll.upsert(ids=ids, documents=docs, metadatas=metas)

    print(f"Ingested {len(ids)} chunks from {path}")


if __name__ == "__main__":
    main()
