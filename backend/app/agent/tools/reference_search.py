from langchain_core.tools import tool

from app.config import settings
from app.memory.chroma_client import get_reference_collection


@tool
async def search_prompt_reference(query: str) -> str:
    """Search the local prompt-engineering reference library (style guides,
    technique cheatsheets, Flux/SD3 prompting guides) for relevant guidance.
    Prefer this over web_search for established techniques and vocabulary."""
    coll = get_reference_collection(settings.chroma_persist_dir)
    res = coll.query(query_texts=[query], n_results=5)

    documents = res["documents"][0] if res["documents"] else []
    if not documents:
        return "No matching reference material found."

    metadatas = res["metadatas"][0]
    return "\n\n---\n\n".join(
        f"[Source: {meta.get('source', 'unknown')}]\n{doc}"
        for doc, meta in zip(documents, metadatas)
    )
