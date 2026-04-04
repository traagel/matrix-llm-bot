# matrix-llm-bot

A simple Matrix chat bot that forwards messages to a local [Ollama](https://ollama.com) instance for LLM inference. Responds to messages in configured rooms, maintains per-user conversation history, and posts replies back to the room.

## Features

- Logs into a Matrix homeserver with username/password
- Joins configured rooms on startup
- Responds to `.ai <message>` or `<bot_name>: <message>` prefixes
- Sends messages to Ollama's `/api/chat` endpoint
- Maintains per-user, per-room conversation history (in-memory, configurable size)
- Skips backlog — only processes messages arriving after startup
- Unencrypted rooms only (no E2EE)

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (or pip)
- A running Ollama instance
- A Matrix account for the bot

## Installation

```bash
git clone https://github.com/youruser/matrix-llm-bot
cd matrix-llm-bot
uv sync
```

## Configuration

Copy the example config and fill in your values:

```bash
cp config.example.json config.json
```

```json
{
  "matrix": {
    "server": "https://matrix.example.com",
    "username": "llm-bot",
    "password": "secretpassword"
  },
  "ollama": {
    "url": "http://localhost:11434",
    "model": "llama3.1:8b"
  },
  "rooms": ["!roomid:example.com"],
  "admins": ["@admin:example.com"],
  "history_size": 20,
  "bot_name": "llm-bot"
}
```

| Field | Description |
|---|---|
| `matrix.server` | URL of your Matrix homeserver |
| `matrix.username` | Bot's Matrix localpart (without `@` and homeserver) |
| `matrix.password` | Bot's Matrix password |
| `ollama.url` | Base URL of your Ollama instance |
| `ollama.model` | Model name to use (e.g. `llama3.1:8b`, `mistral`) |
| `rooms` | List of room IDs the bot should join |
| `admins` | List of Matrix user IDs with admin privileges |
| `history_size` | Number of messages to keep per user per room |
| `bot_name` | Display name prefix used to address the bot |

## Running

```bash
uv run python -m matrix_llm_bot --config config.json
```

Optional flag:

```
--log-level DEBUG   # default: INFO
```

## Usage

In any configured room, address the bot with:

```
.ai What is the capital of France?
```

or

```
llm-bot: explain async/await in Python
```

The bot replies in the same room and remembers the conversation history per user.

## Docker

Build and run with Docker:

```bash
docker build -t matrix-llm-bot .
docker run -v /path/to/your/config.json:/data/config.json matrix-llm-bot
```

## Kubernetes / k3s

Example deployment — mount your `config.json` as a Secret:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: matrix-llm-bot-config
  namespace: matrix
stringData:
  config.json: |
    {
      "matrix": {
        "server": "http://tuwunel.matrix.svc.cluster.local:8008",
        "username": "llm-bot",
        "password": "secretpassword"
      },
      "ollama": {
        "url": "http://ollama.ollama.svc.cluster.local:11434",
        "model": "llama3.1:8b"
      },
      "rooms": ["!roomid:example.com"],
      "admins": ["@admin:example.com"],
      "history_size": 20,
      "bot_name": "llm-bot"
    }
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: matrix-llm-bot
  namespace: matrix
spec:
  replicas: 1
  selector:
    matchLabels:
      app: matrix-llm-bot
  template:
    metadata:
      labels:
        app: matrix-llm-bot
    spec:
      containers:
        - name: bot
          image: youruser/matrix-llm-bot:latest
          volumeMounts:
            - name: config
              mountPath: /data
              readOnly: true
      volumes:
        - name: config
          secret:
            secretName: matrix-llm-bot-config
```

> Keep `replicas: 1` — the bot holds conversation history in memory, multiple replicas would each have an independent state.

## Project structure

```
matrix-llm-bot/
├── pyproject.toml
├── Dockerfile
├── config.example.json
└── src/matrix_llm_bot/
    ├── __init__.py
    ├── __main__.py   # CLI entrypoint
    ├── bot.py        # Matrix client, sync loop, message handler
    ├── ollama.py     # Async Ollama API client
    └── config.py     # Config loading and validation
```

## License

MIT
# matrix-llm-bot
