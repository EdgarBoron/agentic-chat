from typing import Annotated

from langchain_core.messages import AnyMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState

from app.agent.tools._dedup import DUPLICATE_QUERY_MESSAGE, is_repeat_query
from app.config import settings
from app.memory.chroma_client import get_history_collection


@tool
async def search_prompt_history(
    query: str,
    messages: Annotated[list[AnyMessage], InjectedState("messages")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Search previously crafted image prompts (episodic history) for
    similar past requests that can be reused or adapted for the current
    request."""
    if is_repeat_query(messages, "search_prompt_history", query, tool_call_id):
        return DUPLICATE_QUERY_MESSAGE

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
