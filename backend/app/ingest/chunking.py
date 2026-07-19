"""Shared markdown chunking helpers for the reference-library ingestion
scripts (`ingest_reference.py`, `ingest_styles.py`).
"""

import hashlib
import pathlib
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800


def chunk_id(rel_path: pathlib.Path, idx: int, text: str) -> str:
    digest = hashlib.sha256(f"{rel_path}:{idx}:{text}".encode()).hexdigest()[:16]
    return f"{rel_path.as_posix()}-{idx}-{digest}"


def split_document(text: str, splitter: RecursiveCharacterTextSplitter) -> list[str]:
    """Split first on markdown "## " section headers, so e.g. each named
    style/technique in a reference doc becomes its own retrievable chunk
    instead of getting diluted by 2-3 unrelated neighbors sharing one
    embedding. Falls back to plain character splitting for any section
    (or whole doc, if it has no "## " headers at all) that's still too big.

    The document's own title/preamble (everything before the first "## "
    header) is dropped rather than indexed as its own chunk: it's
    generic "what this file contains" framing, and its generic wording
    tends to out-rank specific sections for equally-generic queries
    (e.g. "style reference"), burying the section actually being asked
    about. It's only dropped when the doc has real "## " sections to
    fall back on — a header-less doc still gets indexed in full.
    """
    sections = re.split(r"(?m)^(?=## )", text)
    if len(sections) > 1:
        sections = sections[1:]  # drop the pre-first-header preamble
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= CHUNK_SIZE:
            chunks.append(section)
        else:
            chunks.extend(splitter.split_text(section))
    return chunks
