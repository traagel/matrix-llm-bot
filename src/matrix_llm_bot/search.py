from __future__ import annotations

import httpx


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
