import json

import pytest

from matrix_llm_bot.config import Config


def write_config(tmp_path, data):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    return str(p)


MINIMAL = {
    "matrix": {"server": "https://matrix.example.com", "username": "@bot:example.com", "password": "secret"},
    "ollama": {"url": "http://localhost:11434", "model": "llama3"},
}


def test_minimal_config_loads(tmp_path):
    cfg = Config.load(write_config(tmp_path, MINIMAL))
    assert cfg.matrix.server == "https://matrix.example.com"
    assert cfg.ollama.model == "llama3"


def test_defaults(tmp_path):
    cfg = Config.load(write_config(tmp_path, MINIMAL))
    assert cfg.history_size == 0
    assert cfg.system_prompt == ""
    assert cfg.k8s_enabled is False
    assert cfg.peer_bots == []
    assert cfg.searxng_url == ""


def test_full_config_loads(tmp_path):
    data = {
        **MINIMAL,
        "bot_name": "mybot",
        "history_size": 20,
        "system_prompt": "You are helpful.",
        "k8s_enabled": True,
        "k8s_services": ["nginx", "redis"],
        "k8s_aliases": {"cache": "redis"},
        "peer_bots": ["@other:example.com"],
        "searxng_url": "http://searxng:8080",
        "rooms": ["!abc:example.com"],
        "admins": ["@admin:example.com"],
    }
    cfg = Config.load(write_config(tmp_path, data))
    assert cfg.bot_name == "mybot"
    assert cfg.history_size == 20
    assert cfg.k8s_enabled is True
    assert cfg.k8s_services == ["nginx", "redis"]
    assert cfg.k8s_aliases == {"cache": "redis"}
    assert cfg.ollama.routing_model == ""  # not set → empty


def test_missing_matrix_server_raises(tmp_path):
    data = {"matrix": {"username": "@bot:example.com", "password": "x"}, "ollama": MINIMAL["ollama"]}
    with pytest.raises(ValueError, match=r"matrix\.server"):
        Config.load(write_config(tmp_path, data))


def test_missing_ollama_model_raises(tmp_path):
    data = {**MINIMAL, "ollama": {"url": "http://localhost:11434"}}
    with pytest.raises(ValueError, match=r"ollama\.model"):
        Config.load(write_config(tmp_path, data))


def test_multiple_missing_fields_reported(tmp_path):
    data = {"matrix": {}, "ollama": {}}
    with pytest.raises(ValueError) as exc_info:
        Config.load(write_config(tmp_path, data))
    msg = str(exc_info.value)
    assert "matrix.server" in msg
    assert "ollama.url" in msg
