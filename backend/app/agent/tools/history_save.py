import datetime
import hashlib

from langchain_core.tools import tool

from app.config import settings
from app.memory.chroma_client import get_history_collection


@tool
async def save_prompt_to_history(prompt_text: str, note: str = "") -> str:
    """Save a finalized, user-facing image-generation prompt to episodic
    history. Always call this exactly once, immediately before presenting
    the final crafted prompt to the user."""
    coll = get_history_collection(settings.chroma_persist_dir)
    # Content-derived id makes repeated saves of the identical prompt text
    # idempotent (upsert overwrites in place) instead of creating
    # duplicate history entries, e.g. if the model calls this tool twice
    # for the same final prompt.
    doc_id = hashlib.sha256(prompt_text.encode()).hexdigest()[:24]
    coll.upsert(
        documents=[prompt_text],
        metadatas=[
            {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "note": note,
            }
        ],
        ids=[doc_id],
    )
    return "Saved."
