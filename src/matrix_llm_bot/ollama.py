from __future__ import annotations

import httpx


class OllamaClient:
    def __init__(self, url: str, model: str) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=120.0)

    async def chat(self, messages: list[dict], model: str | None = None) -> str:
        response = await self._client.post(
            f"{self.url}/api/chat",
            json={"model": model or self.model, "messages": messages, "stream": False},
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def chat_with_image(self, messages: list[dict], image_b64: str, model: str | None = None) -> str:
        msgs = list(messages)
        if msgs and msgs[-1]["role"] == "user":
            msgs[-1] = {**msgs[-1], "images": [image_b64]}
        response = await self._client.post(
            f"{self.url}/api/chat",
            json={"model": model or self.model, "messages": msgs, "stream": False},
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """Returns the raw assistant message dict (may contain tool_calls)."""
        response = await self._client.post(
            f"{self.url}/api/chat",
            json={"model": self.model, "messages": messages, "tools": tools, "stream": False},
        )
        response.raise_for_status()
        return response.json()["message"]

    async def aclose(self) -> None:
        await self._client.aclose()
