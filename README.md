# matrix-llm-bot

A Matrix chat bot that forwards messages to a local [Ollama](https://ollama.com) instance for LLM inference. Supports web search via SearXNG, image vision, Kubernetes cluster monitoring, and per-user conversation history.

## Features

- Responds when mentioned by name or @mentioned in a room
- LLM gate — decides if the message is actually addressed to the bot before responding
- Per-user, per-room conversation history (in-memory, configurable size)
- **Web search** via SearXNG with Ollama tool calling — model decides when to search
- **Image vision** — send an image, then ask the bot about it
- **Kubernetes monitoring** — query pod status, deployments, logs, and versions directly from chat
- Persona via `system_prompt` — give the bot a character
- `reset` command clears your conversation history
- Admin commands: `avatar <url>` sets the bot's profile picture
- Skips message backlog on startup
- @mentions sender in replies for clear attribution
- Unencrypted rooms only (no E2EE)

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (or pip)
- A running Ollama instance
- A Matrix account for the bot

## Installation

```bash
git clone https://github.com/traagel/matrix-llm-bot
cd matrix-llm-bot
uv sync
```

## Configuration

```bash
cp config.example.json config.json
```

```json
{
  "matrix": {
    "server": "http://localhost:8008",
    "username": "@bot:example.com",
    "password": "secretpassword"
  },
  "ollama": {
    "url": "http://localhost:11434",
    "model": "llama3.1:8b",
    "vision_model": "llava:7b",
    "routing_model": "llama3.2:1b"
  },
  "rooms": ["!roomid:example.com"],
  "admins": ["@admin:example.com"],
  "history_size": 20,
  "bot_name": "llm-bot",
  "system_prompt": "",
  "searxng_url": "http://searxng:8080",
  "k8s_enabled": false,
  "k8s_services": ["myapp", "mydb", "myproxy"],
  "k8s_keywords": [],
  "k8s_aliases": {
    "custom-name": "actual-deployment-name"
  }
}
```

### Config reference

| Field | Required | Description |
|---|---|---|
| `matrix.server` | Yes | URL of your Matrix homeserver |
| `matrix.username` | Yes | Bot's full Matrix user ID (`@bot:example.com`) |
| `matrix.password` | Yes | Bot's Matrix password |
| `ollama.url` | Yes | Base URL of your Ollama instance |
| `ollama.model` | Yes | Main chat model (e.g. `llama3.1:8b`) |
| `ollama.vision_model` | No | Vision model for image analysis (e.g. `llava:7b`). Omit to disable. |
| `ollama.routing_model` | No | Small fast model for yes/no gate calls (e.g. `llama3.2:1b`). Falls back to main model. |
| `rooms` | Yes | List of room IDs the bot should join |
| `admins` | No | Matrix user IDs with admin privileges (avatar command, k8s logs) |
| `history_size` | No | Messages to keep per user per room (default: 20) |
| `bot_name` | No | Display name set on login, also used as mention trigger |
| `system_prompt` | No | Persona/instructions prepended to every LLM call |
| `searxng_url` | No | SearXNG base URL. Omit to disable web search. |
| `k8s_enabled` | No | Enable Kubernetes monitoring tools (default: false) |
| `k8s_services` | No | Service names that trigger k8s context injection (e.g. `["jellyfin", "postgres"]`) |
| `k8s_keywords` | No | Trigger words for k8s context injection. Omit to use built-in defaults; set to `[]` to disable keyword matching entirely. |
| `k8s_aliases` | No | Map custom names to real deployment names (e.g. `{"myflix": "jellyfin"}`) |

## Running

```bash
uv run python -m matrix_llm_bot --config config.json
# or
matrix-llm-bot --config config.json --log-level DEBUG
```

## Usage

Address the bot by name or @mention:

```
llm-bot what's the weather in Tokyo?
@llm-bot explain how transformers work
llm-bot reset
```

**Web search** — triggered automatically when the model needs current info, or explicitly:
```
llm-bot search for the latest Rust release notes
```

**Images** — send an image to the room, then mention the bot:
```
[upload image]
llm-bot what's in this image?
```

**Kubernetes** (admins and non-admins can check status; only admins can view logs):
```
llm-bot is jellyfin healthy?
llm-bot show me pods in the media namespace
llm-bot get logs for the ollama pod       ← admins only
```

## Docker

```bash
docker build -t traagel/matrix-llm-bot:latest .
docker run -v /path/to/config.json:/data/config.json traagel/matrix-llm-bot:latest
```

## Kubernetes / k3s

Mount `config.json` as a Secret. The bot auto-detects in-cluster credentials when `k8s_enabled: true`.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: matrix-llm-bot-config
  namespace: myns
stringData:
  config.json: |
    {
      "matrix": { "server": "http://homeserver:8008", "username": "@bot:example.com", "password": "..." },
      "ollama": { "url": "http://ollama:11434", "model": "llama3.1:8b" },
      "rooms": ["!roomid:example.com"],
      "admins": ["@admin:example.com"],
      "bot_name": "llm-bot",
      "k8s_enabled": true
    }
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: matrix-llm-bot
  namespace: myns
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
      serviceAccountName: matrix-llm-bot
      containers:
        - name: bot
          image: traagel/matrix-llm-bot:latest
          volumeMounts:
            - name: config
              mountPath: /data
              readOnly: true
      volumes:
        - name: config
          secret:
            secretName: matrix-llm-bot-config
```

> Keep `replicas: 1` — history is in-memory, multiple replicas would each have independent state.

**RBAC** required when `k8s_enabled: true`:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: matrix-llm-bot
rules:
  - apiGroups: [""]
    resources: [pods, pods/log, services, namespaces]
    verbs: [get, list]
  - apiGroups: [apps]
    resources: [deployments]
    verbs: [get, list]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: matrix-llm-bot
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: matrix-llm-bot
subjects:
  - kind: ServiceAccount
    name: matrix-llm-bot
    namespace: myns
```

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
    ├── search.py     # SearXNG client
    ├── k8s.py        # Kubernetes API client
    └── config.py     # Config loading and validation
```

## License

MIT
