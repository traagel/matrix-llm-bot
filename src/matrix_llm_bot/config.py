from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_K8S_KEYWORDS: list[str] = [
    # Kubernetes terms
    "k8s", "k3s", "kubernetes", "cluster", "pod", "pods", "deploy", "deployment",
    "namespace", "service", "node", "container", "replica", "restart",
    "healthy", "health", "running", "status", "version", "logs", "log",
    "up", "down", "crashed", "evicted", "pending", "oomkilled",
]


@dataclass
class MatrixConfig:
    server: str
    username: str
    password: str


@dataclass
class OllamaConfig:
    url: str
    model: str
    vision_model: str = ""
    routing_model: str = ""


@dataclass
class Config:
    matrix: MatrixConfig
    ollama: OllamaConfig
    rooms: list[str]
    admins: list[str]
    history_size: int
    bot_name: str
    system_prompt: str = ""
    searxng_url: str = ""
    k8s_enabled: bool = False
    k8s_keywords: list[str] = field(default_factory=lambda: list(DEFAULT_K8S_KEYWORDS))
    k8s_services: list[str] = field(default_factory=list)

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
                vision_model=ollama.get("vision_model", ""),
                routing_model=ollama.get("routing_model", ""),
            ),
            rooms=data.get("rooms", []),
            admins=data.get("admins", []),
            history_size=int(data.get("history_size", 20)),
            bot_name=data.get("bot_name", "llm-bot"),
            system_prompt=data.get("system_prompt", ""),
            searxng_url=data.get("searxng_url", ""),
            k8s_enabled=bool(data.get("k8s_enabled", False)),
            k8s_keywords=data.get("k8s_keywords", list(DEFAULT_K8S_KEYWORDS)),
            k8s_services=data.get("k8s_services", []),
        )
