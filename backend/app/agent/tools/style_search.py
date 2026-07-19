from typing import Annotated

from langchain_core.messages import AnyMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState

from app.agent.tools._dedup import DUPLICATE_QUERY_MESSAGE, is_repeat_query
from app.config import settings
from app.memory.chroma_client import get_style_collection


@tool
async def search_artist_styles(
    query: str,
    messages: Annotated[list[AnyMessage], InjectedState("messages")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Search the local artist/photographer style catalog for a named
    visual style (e.g. a specific photographer or artist) to reference in
    an image prompt. Prefer this over web_search for known style names."""
    if is_repeat_query(messages, "search_artist_styles", query, tool_call_id):
        return DUPLICATE_QUERY_MESSAGE

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
