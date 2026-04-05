from __future__ import annotations

import httpx

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current, real-time, or factual information you do not already know. "
            "Only use this for explicit questions about news, current events, prices, weather, or facts. "
            "Do NOT use for greetings, casual conversation, opinions, or anything you can answer yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
}


class SearXNGClient:
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=15.0)

    async def search(self, query: str, num_results: int = 5) -> list[dict]:
        response = await self._client.get(
            f"{self.url}/search",
            params={"q": query, "format": "json", "categories": "general"},
        )
        response.raise_for_status()
        results = response.json().get("results", [])[:num_results]
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in results
        ]

    async def aclose(self) -> None:
        await self._client.aclose()
