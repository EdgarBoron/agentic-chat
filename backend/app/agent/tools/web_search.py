from typing import Annotated

import httpx
from langchain_core.messages import AnyMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState

from app.agent.tools._dedup import DUPLICATE_QUERY_MESSAGE, is_repeat_query
from app.config import settings


@tool
async def web_search(
    query: str,
    messages: Annotated[list[AnyMessage], InjectedState("messages")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Search the public web for art styles, techniques, lighting/camera
    terminology, or current trends relevant to crafting an image prompt.
    Use this when you need information not already in the local reference
    library, or when the user references something current/trending."""
    if is_repeat_query(messages, "web_search", query, tool_call_id):
        return DUPLICATE_QUERY_MESSAGE

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{settings.searxng_url}/search",
            params={"q": query, "format": "json"},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])[:5]

    if not results:
        return "No web results found."

    return "\n\n".join(
        f"[{r.get('title', 'untitled')}]({r.get('url', '')})\n{r.get('content', '')[:400]}"
        for r in results
    )
