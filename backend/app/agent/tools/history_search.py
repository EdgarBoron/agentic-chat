from langchain_core.tools import tool

from app.config import settings
from app.memory.chroma_client import get_history_collection


@tool
async def search_prompt_history(query: str) -> str:
    """Search previously crafted image prompts (episodic history) for
    similar past requests that can be reused or adapted for the current
    request."""
    coll = get_history_collection(settings.chroma_persist_dir)
    res = coll.query(query_texts=[query], n_results=3)

    documents = res["documents"][0] if res["documents"] else []
    if not documents:
        return "No similar past prompts found."

    metadatas = res["metadatas"][0]
    return "\n\n---\n\n".join(
        f"[{meta.get('timestamp', 'unknown time')}]\n{doc}"
        for doc, meta in zip(documents, metadatas)
    )
