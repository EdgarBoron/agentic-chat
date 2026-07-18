import httpx
from langchain_core.tools import tool

from app.config import settings


@tool
async def web_search(query: str) -> str:
    """Search the public web for art styles, techniques, lighting/camera
    terminology, or current trends relevant to crafting an image prompt.
    Use this when you need information not already in the local reference
    library, or when the user references something current/trending."""
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
