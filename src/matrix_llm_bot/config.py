from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MatrixConfig:
    server: str
    username: str
    password: str


@dataclass
class OllamaConfig:
    url: str
    model: str


@dataclass
class Config:
    matrix: MatrixConfig
    ollama: OllamaConfig
    rooms: list[str]
    admins: list[str]
    history_size: int
    bot_name: str

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = json.loads(Path(path).read_text())

        matrix = data.get("matrix", {})
        ollama = data.get("ollama", {})

        missing = []
        for key in ("server", "username", "password"):
            if not matrix.get(key):
                missing.append(f"matrix.{key}")
        for key in ("url", "model"):
            if not ollama.get(key):
                missing.append(f"ollama.{key}")
        if missing:
            raise ValueError(f"Missing required config fields: {', '.join(missing)}")

        return cls(
            matrix=MatrixConfig(
                server=matrix["server"],
                username=matrix["username"],
                password=matrix["password"],
            ),
            ollama=OllamaConfig(
                url=ollama["url"],
                model=ollama["model"],
            ),
            rooms=data.get("rooms", []),
            admins=data.get("admins", []),
            history_size=int(data.get("history_size", 20)),
            bot_name=data.get("bot_name", "llm-bot"),
        )
