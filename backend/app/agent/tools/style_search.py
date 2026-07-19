from langchain_core.tools import tool

from app.config import settings
from app.memory.chroma_client import get_style_collection


@tool
async def search_artist_styles(query: str) -> str:
    """Search the local artist/photographer style catalog for a named
    visual style (e.g. a specific photographer or artist) to reference in
    an image prompt. Prefer this over web_search for known style names."""
    coll = get_style_collection(settings.chroma_persist_dir)
    res = coll.query(query_texts=[query], n_results=5)

    documents = res["documents"][0] if res["documents"] else []
    if not documents:
        return "No matching style found."

    metadatas = res["metadatas"][0]
    return "\n\n---\n\n".join(
        f"[Source: {meta.get('source', 'unknown')}]\n{doc}"
        for doc, meta in zip(documents, metadatas)
    )
