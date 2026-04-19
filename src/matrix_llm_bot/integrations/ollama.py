from __future__ import annotations

import json
import logging

import httpx

logger = logging.getLogger(__name__)


def _parse_yes_no(answer: str, default: bool) -> bool:
    answer = answer.strip().upper()
    yes_pos = answer.find("YES")
    no_pos = answer.find("NO")
    if yes_pos == -1 and no_pos == -1:
        return default
    if yes_pos == -1:
        return False
    if no_pos == -1:
        return True
    return yes_pos < no_pos


class OllamaClient:
    def __init__(
        self,
        url: str,
        model: str,
        routing_model: str = "",
        api_kind: str = "ollama",
        api_key: str = "",
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self._routing_model = routing_model or model
        self._api_kind = api_kind
        self._chat_path = "/v1/chat/completions" if api_kind == "openai" else "/api/chat"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(timeout=120.0, headers=headers)

    def _build_payload(
        self,
        model: str,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        image_b64: str | None = None,
    ) -> dict:
        if self._api_kind == "openai":
            out_messages = _to_openai_messages(messages, image_b64=image_b64)
        else:
            out_messages = list(messages)
            if image_b64 and out_messages and out_messages[-1].get("role") == "user":
                out_messages[-1] = {**out_messages[-1], "images": [image_b64]}
        payload: dict = {"model": model, "messages": out_messages, "stream": False}
        if tools:
            payload["tools"] = tools
        return payload

    def _parse_response(self, data: dict) -> dict:
        if self._api_kind == "openai":
            return _from_openai_message(data["choices"][0]["message"])
        return data["message"]

    async def _post(self, payload: dict) -> dict:
        response = await self._client.post(f"{self.url}{self._chat_path}", json=payload)
        response.raise_for_status()
        return self._parse_response(response.json())

    async def chat(self, messages: list[dict], model: str | None = None) -> str:
        msg = await self._post(self._build_payload(model or self.model, messages))
        return msg.get("content", "")

    async def chat_with_image(self, messages: list[dict], image_b64: str, model: str | None = None) -> str:
        msg = await self._post(self._build_payload(model or self.model, messages, image_b64=image_b64))
        return msg.get("content", "")

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
        raw = await self.chat(messages, model=self._routing_model)
        result = _parse_yes_no(raw, default=False)
        logger.debug("Search gate: %r -> %s (raw: %r)", message[:80], "YES" if result else "NO", raw[:40])
        return result

    async def should_respond(self, bot_name: str, message: str, peer_bots: list[str] | None = None) -> bool:
        """Ask the model if this message is directed at bot_name and not at a peer bot."""
        others = peer_bots or []
        others_line = (
            f" Other bots in this room: {', '.join(others)}."
            f" Answer NO if the message is clearly directed at one of them instead."
            if others
            else ""
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a routing assistant deciding if a chat message is directly addressed "
                    f"to '{bot_name}' (match case-insensitively, including nicknames or partial matches).{others_line} "
                    f"Answer YES if '{bot_name}' is mentioned, named, or appears to be the one being spoken to. "
                    f"When in doubt, answer YES. "
                    f"Answer NO only if the message is clearly addressed to a specific other person or bot and '{bot_name}' is not involved. "
                    f"Reply with a single word: YES or NO. Nothing else."
                ),
            },
            {"role": "user", "content": message},
        ]
        raw = await self.chat(messages, model=self._routing_model)
        result = _parse_yes_no(raw, default=True)
        logger.debug("Respond gate: %r -> %s (raw: %r)", message[:80], "YES" if result else "NO", raw[:40])
        return result

    async def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """Returns the raw assistant message dict (may contain tool_calls)."""
        tool_names = [t["function"]["name"] for t in tools]
        logger.debug("chat_with_tools: offering %s", tool_names)
        msg = await self._post(self._build_payload(self.model, messages, tools=tools))
        calls = [c.get("function", {}).get("name") for c in (msg.get("tool_calls") or [])]
        logger.debug("chat_with_tools: model called %s", calls if calls else "no tools")
        return msg

    async def aclose(self) -> None:
        await self._client.aclose()


def _to_openai_messages(messages: list[dict], *, image_b64: str | None = None) -> list[dict]:
    """Translate the internal (Ollama-shaped) message list to OpenAI format."""
    out: list[dict] = []
    last_user_idx = _last_user_index(messages) if image_b64 else -1

    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            preserved = m.get("__openai_tool_calls__")
            tool_calls = preserved if preserved else [
                {
                    "id": tc.get("_id") or f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": json.dumps(tc.get("function", {}).get("arguments", {}) or {}),
                    },
                }
                for idx, tc in enumerate(m["tool_calls"])
            ]
            entry: dict = {"role": "assistant", "tool_calls": tool_calls}
            content = m.get("content")
            if content:
                entry["content"] = content
            out.append(entry)
        elif role == "tool":
            tool_call_id = m.get("tool_call_id") or _find_tool_call_id(out, m.get("tool_name"))
            entry = {"role": "tool", "content": m.get("content", "")}
            if tool_call_id:
                entry["tool_call_id"] = tool_call_id
            out.append(entry)
        elif role == "user" and i == last_user_idx and image_b64:
            out.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": m.get("content", "")},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            )
        else:
            out.append({k: v for k, v in m.items() if not k.startswith("__")})
    return out


def _from_openai_message(msg: dict) -> dict:
    """Translate an OpenAI assistant message back into the internal Ollama-shape."""
    result: dict = {"role": "assistant", "content": msg.get("content") or ""}
    raw_tool_calls = msg.get("tool_calls") or []
    if raw_tool_calls:
        normalized = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            args_raw = fn.get("arguments", "")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw) if args_raw else {}
                except json.JSONDecodeError:
                    args = {}
            else:
                args = args_raw or {}
            normalized.append(
                {
                    "_id": tc.get("id", ""),
                    "function": {"name": fn.get("name", ""), "arguments": args},
                }
            )
        result["tool_calls"] = normalized
        result["__openai_tool_calls__"] = raw_tool_calls
    return result


def _last_user_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


def _find_tool_call_id(translated: list[dict], tool_name: str | None) -> str | None:
    for m in reversed(translated):
        calls = m.get("tool_calls") if m.get("role") == "assistant" else None
        if not calls:
            continue
        if tool_name:
            for tc in calls:
                if tc.get("function", {}).get("name") == tool_name:
                    return tc.get("id")
        return calls[0].get("id")
    return None
