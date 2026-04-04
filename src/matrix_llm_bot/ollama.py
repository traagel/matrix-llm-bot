from __future__ import annotations

import httpx


class OllamaClient:
    def __init__(self, url: str, model: str, routing_model: str = "") -> None:
        self.url = url.rstrip("/")
        self.model = model
        self._routing_model = routing_model or model
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

    async def should_search(self, message: str) -> bool:
        """Decide if a message requires a web search to answer."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You decide if a message requires a web search to answer. "
                    "Answer YES only if the message explicitly asks for: current news, live prices, "
                    "today's weather, recent events, or time-sensitive facts you cannot know. "
                    "Answer NO for: greetings, small talk, opinions, jokes, questions about what was "
                    "previously said in the conversation, memory questions ('what did I tell you', "
                    "'do you remember'), personal questions, or anything answerable from general knowledge. "
                    "If the message asks about current events, prices, or news — answer YES. "
                    "Reply with a single word: YES or NO. Nothing else."
                ),
            },
            {"role": "user", "content": message},
        ]
        response = await self._client.post(
            f"{self.url}/api/chat",
            json={"model": self._routing_model, "messages": messages, "stream": False},
        )
        response.raise_for_status()
        answer = response.json()["message"]["content"].strip().upper()
        return answer.startswith("Y")

    async def should_respond(self, bot_name: str, message: str) -> bool:
        """Ask the model if this message is actually directed at bot_name."""
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a routing assistant deciding if a chat message is directly addressed "
                    f"to '{bot_name}' (match the name case-insensitively). "
                    f"Answer YES if '{bot_name}' is the one being spoken to or asked to do something. "
                    f"Answer NO if the message is addressed to someone else, if '{bot_name}' is only "
                    f"mentioned in passing (e.g. 'tell {bot_name} to leave'), or if another name "
                    f"appears at the start of the message as the primary addressee. "
                    f"Reply with a single word: YES or NO. Nothing else."
                ),
            },
            {"role": "user", "content": message},
        ]
        response = await self._client.post(
            f"{self.url}/api/chat",
            json={"model": self._routing_model, "messages": messages, "stream": False},
        )
        response.raise_for_status()
        answer = response.json()["message"]["content"].strip().upper()
        return answer.startswith("Y")

    async def acknowledgment(self, system_prompt: str, query: str) -> str:
        """Generate a brief in-character acknowledgment before a web search."""
        messages = [
            {
                "role": "system",
                "content": (
                    system_prompt + "\n\n" if system_prompt else ""
                ) + (
                    "Reply with ONLY a single short sentence acknowledging that you are about to "
                    "look something up. Stay in character. No additional commentary."
                ),
            },
            {"role": "user", "content": f'I need you to search for: "{query}"'},
        ]
        response = await self._client.post(
            f"{self.url}/api/chat",
            json={"model": self._routing_model, "messages": messages, "stream": False},
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
