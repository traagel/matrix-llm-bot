from __future__ import annotations

import httpx


class OllamaClient:
    def __init__(self, url: str, model: str) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=120.0)

    async def chat(self, messages: list[dict]) -> str:
        response = await self._client.post(
            f"{self.url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def aclose(self) -> None:
        await self._client.aclose()
