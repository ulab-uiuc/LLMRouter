# OpenClaw Router

OpenClaw Router is an OpenAI-compatible API server that routes each request to the most suitable backend LLM (Together, NVIDIA, OpenAI-compatible endpoints, etc.). It is designed to integrate cleanly with OpenClaw so you can use it from Slack (and other channels supported by OpenClaw).

## Why OpenClaw + OpenClaw Router

Pairing OpenClaw (channels + agents) with OpenClaw Router (OpenAI-compatible routing layer) gives you:

- A Slack-native UX: talk to your bot from mobile/desktop without building a custom UI.
- One stable endpoint: OpenClaw calls a single `baseUrl`, while the Router selects the best backend model per request.
- Provider flexibility: swap Together/NVIDIA/OpenAI-compatible backends in `openclaw_router/config.yaml` without touching Slack/OpenClaw.
- Key isolation: upstream LLM API keys stay on the server (Router), not on clients.
- Network simplicity: with Slack Socket Mode, you typically only need outbound internet from the server; no public inbound webhook URL.
- Better ops/debuggability: central logs, health checks, and optional `[model]` prefixes to verify routing decisions.

## Features

- OpenAI-compatible API: drop-in replacement for OpenAI-style clients (`/v1/chat/completions`).
- HTTP tool calling pass-through: forwards OpenAI-compatible `tools` / `tool_choice` fields and preserves `tool_calls` in `/v1/chat/completions` responses. The Router does not execute tools itself.
- Multiple routing strategies: built-in strategies plus the original LLMRouter ML-based routers.
- Streaming support: end-to-end streaming responses.
- Optional model prefix: add `[model_name]` to responses for debugging routing decisions.
- Multi-API key support: rotate keys (for example NVIDIA) for basic load balancing.
- Optional routing memory (Contriever): persist (query -> model) history and retrieve top-k similar past routes to help `router.strategy: llm`.

## Overview

High level flow (Slack + OpenClaw + OpenClaw Router):

```text
Slack (mobile/desktop)
        |
        | (Slack Cloud)
        v
OpenClaw (Slack channel, Socket Mode)
        |
        | HTTP (OpenAI-compatible): POST /v1/chat/completions  model="auto"
        v
OpenClaw Router (FastAPI, default :8000)
        |
        | (routing: built-in strategies or original LLMRouter ML routers)
        |
        | HTTP to provider (Together/NVIDIA/OpenAI-compatible)
        v
Upstream LLM Provider(s)
```

Key idea:
- OpenClaw behaves like an OpenAI client.
- OpenClaw Router behaves like an OpenAI-compatible server.
- OpenClaw Router then calls your real model providers using the API keys you configure in `openclaw_router/config.yaml`.

## Installation

Python dependencies are managed by the main project packaging. Follow the repository root installation instructions:
- See `../README.md`

If you want Slack integration, you also need OpenClaw (Node.js). OpenClaw has strict Node.js version requirements; if you see syntax errors or version errors, upgrade Node (commonly via `nvm`).

### Install OpenClaw (Slack / Discord gateway)

1) Install a recent Node.js (recommended: via `nvm`)

OpenClaw's minimum Node version may change over time. If you run `openclaw` and it says "requires Node >= X", install that (or newer). In our setup, OpenClaw required Node 22+.

```bash
# Install nvm (Linux/macOS)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# Load nvm for the current shell
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

# Install and use a modern Node.js (example: 22)
nvm install 22
nvm use 22

node -v
npm -v
```

If `nvm use` warns about `.npmrc` `prefix`/`globalconfig` conflicts, remove the npm prefix settings (or run the suggested `nvm use --delete-prefix ...`). The simplest goal is: your `node` and `npm` should come from the `~/.nvm/...` path, and global installs should not try to write into `/usr/local/...`.

2) Install OpenClaw

```bash
npm install -g openclaw
openclaw --version
openclaw doctor
```

Common install/runtime issues:
- `npm ERR! EACCES ... /usr/local/lib/node_modules`: you are trying to install globally into a root-owned directory. Use `nvm` (recommended) so global installs go into your user directory, or fix npm permissions/prefix.
- `openclaw requires Node >= ...`: upgrade Node to the required version and retry.
- `openclaw: command not found`: your npm global bin is not on `PATH` (or you are in a new shell/session where `nvm` is not loaded).

## Configuration Summary

You typically configure 2 files:

1. `openclaw_router/config.yaml`
- Controls which backend models exist, how routing works, and which API keys to use when calling upstream providers.

2. `~/.openclaw/openclaw.json`
- Tells OpenClaw how to call OpenClaw Router (as a custom model provider).
- Configures Slack tokens and Socket Mode for receiving events.

Important: `~/.openclaw/openclaw.json` is a full OpenClaw config. You should edit/merge specific sections, not replace the whole file.

## Step-by-Step (Recommended: One Script Starts Router + Gateway)

### 1) Configure OpenClaw Router backends (API keys live here)

Edit `openclaw_router/config.yaml`.

Notes:
- The `api_keys` section is used when OpenClaw Router calls upstream providers. This is where you put Together/NVIDIA/OpenAI keys.
- OpenClaw does not need the upstream provider keys. OpenClaw only calls your Router.

Example: Together (OpenAI-compatible)

```yaml
serve:
  host: "0.0.0.0"
  port: 8000
  show_model_prefix: true

router:
  strategy: llm
  provider: together
  base_url: https://api.together.xyz/v1
  model: meta-llama/Llama-3.1-8B-Instruct-Turbo

api_keys:
  together: ${TOGETHER_API_KEY}

llms:
  llama-3.1-8b:
    description: "Fast chat"
    provider: together
    model: meta-llama/Llama-3.1-8B-Instruct-Turbo
    base_url: https://api.together.xyz/v1
    max_tokens: 1024
    context_limit: 128000

  qwen2.5-72b:
    description: "Stronger reasoning"
    provider: together
    model: Qwen/Qwen2.5-72B-Instruct-Turbo
    base_url: https://api.together.xyz/v1
    max_tokens: 1024
    context_limit: 32768
```

### 2) Configure OpenClaw to call your Router (provider: `openclaw`)

In `~/.openclaw/openclaw.json`, ensure these keys exist under `models.providers.openclaw`.

This is a fragment (merge into your existing JSON):

```json
{
  "models": {
    "providers": {
      "openclaw": {
        "api": "openai-completions",
        "baseUrl": "http://127.0.0.1:8000/v1",
        "apiKey": "not-needed",
        "models": [{"id": "auto", "name": "OpenClaw Router"}]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {"primary": "openclaw/auto"}
    }
  }
}
```

Meaning of the important fields:
- `baseUrl`: the Router address OpenClaw will send requests to. The Router listens on the `serve.port` you set in `openclaw_router/config.yaml`.
  - If OpenClaw and Router run on the same machine, use `http://127.0.0.1:<port>/v1`.
  - If they run on different machines/containers, `localhost` will be wrong. Use a reachable IP/hostname.
- `api`: the protocol OpenClaw uses to talk to your `baseUrl`. OpenClaw Router is OpenAI-compatible, so use `openai-completions`.
- `apiKey`: can be a placeholder if your Router does not enforce auth. It is not the key for Together/NVIDIA/OpenAI.

If you change the Router port (for example with `./scripts/start-openclaw.sh -p 9000`), you must also update `baseUrl` to match.

Safe patch (recommended): this updates only the relevant keys and keeps the rest of your OpenClaw config intact.

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path.home() / ".openclaw" / "openclaw.json"
cfg = json.loads(p.read_text())

models = cfg.setdefault("models", {})
models.setdefault("mode", "merge")
providers = models.setdefault("providers", {})
claw = providers.setdefault("openclaw", {})

claw["api"] = "openai-completions"
claw["baseUrl"] = "http://127.0.0.1:8000/v1"
claw.setdefault("apiKey", "not-needed")
claw.setdefault("models", [{"id": "auto", "name": "OpenClaw Router"}])

agents = cfg.setdefault("agents", {})
defaults = agents.setdefault("defaults", {})
defaults.setdefault("model", {})["primary"] = "openclaw/auto"

p.write_text(json.dumps(cfg, indent=2))
print("updated", p)
PY
```

Optional but recommended for first-time setup:
- Ensure `gateway.mode` is set to `local`, otherwise `openclaw gateway run` may be blocked.

```json
{
  "gateway": { "mode": "local" }
}
```

### 3) Configure Slack (Socket Mode) in OpenClaw

OpenClaw's Slack config is under `channels.slack` (newer versions migrated away from `slack.*`).

In `~/.openclaw/openclaw.json`, ensure you have:

```json
{
  "channels": {
    "slack": {
      "enabled": true,
      "mode": "socket",
      "botToken": "xoxb-your-bot-token",
      "appToken": "xapp-your-app-token"
    }
  }
}
```

#### Slack App setup checklist (Socket Mode)

1. Create a Slack App (from scratch) in your workspace.
2. Add a Bot user.
3. Enable Socket Mode.
4. Create an App-Level Token (this becomes your `xapp-...` token):
   - Scope: `connections:write`
   - Slack only shows this token once. If you lose it, create a new one.
5. OAuth & Permissions:
   - Minimal scopes for mention-only bots: `app_mentions:read`, `chat:write`
   - If you want DM support and message events, you usually also need: `im:history` (and optionally `im:read`)
6. Event Subscriptions (bot events):
   - `app_mention` (recommended)
   - `message.im` (if you want DMs)
   - Optionally `message.channels` (if you want channel messages)
7. Install (or re-install) the app to your workspace after changing scopes/events.
8. In Slack, invite the bot to the channel you want it to respond in (for public channels).

Where to find tokens in the Slack UI:
- Bot token (`xoxb-...`): "OAuth & Permissions" -> "Bot User OAuth Token"
- App token (`xapp-...`): "Socket Mode" -> "App-Level Tokens" (Slack shows it once)

Where to configure event subscriptions in the Slack UI:
- "Event Subscriptions" (left sidebar under Features)
  - In Socket Mode, events are delivered over the WebSocket; you typically do not need a public Request URL.
  - After adding events, click **Save Changes** at the bottom of the page (easy to miss).

### 4) Start everything (Router + OpenClaw Gateway)

From the repository root:

```bash
./scripts/start-openclaw.sh
```

This script requires `bash` (Linux/macOS). On Windows, run it under WSL, or start the Router and OpenClaw in separate terminals.

Common options:

```bash
./scripts/start-openclaw.sh -c openclaw_router/config.yaml
./scripts/start-openclaw.sh -p 9000
./scripts/start-openclaw.sh -r knnrouter --router-config configs/model_config_test/knnrouter.yaml
./scripts/start-openclaw.sh --no-gateway
```

Logs:
- Router: `/tmp/openclaw.log`
- Gateway: `/tmp/openclaw-gateway.log`

Stop everything:

```bash
./scripts/stop-openclaw.sh
```

### 5) Pairing / access approval (required by default in many setups)

If Slack says something like "access not configured" and shows a pairing code, approve it on the server:

```bash
openclaw pairing approve slack <pairing-code>
```

After approval, try sending a DM to the bot or mentioning it in a channel again.

## Configuration Reference

### Main Router Config (`openclaw_router/config.yaml`)

Minimal shape:

```yaml
serve:
  host: "0.0.0.0"
  port: 8000
  show_model_prefix: true

router:
  strategy: random   # random | round_robin | rules | llm | llmrouter

api_keys:
  together: ${TOGETHER_API_KEY}    # or a literal key string
  nvidia:
    - nvapi-...                    # list = basic key rotation

llms:
  some-model:
    provider: together
    model: meta-llama/Llama-3.1-8B-Instruct-Turbo
    base_url: https://api.together.xyz/v1
    description: "..."
    max_tokens: 1024
    context_limit: 128000
```

Key fields:
- `serve.host` / `serve.port`: where OpenClaw Router listens.
- `router.strategy`:
  - `random` / `round_robin` / `rules`: deterministic/simple routing.
  - `llm`: uses a "router LLM" to pick the backend model.
    - If `auth_mode` resolves to `bearer`, an API key is required.
    - If `auth_mode` resolves to `none` (for local OpenAI-compatible backends), key is optional.
  - `llmrouter`: uses the original LLMRouter ML-based routers.
- `api_keys`: keys used by OpenClaw Router when calling upstream providers.
  - Supports environment variables like `${TOGETHER_API_KEY}`.
- `llms`: your backend model pool (the Router chooses one of these for each request).
  - You can mix frameworks in one pool (for example `sglang`, `vllm`, `llama_cpp`, `lmstudio`, `huggingface_cli`, cloud providers).

OpenAI-compatible adapter fields (supported in both `router` and each `llms.<name>`):
- `provider_type`: currently `openai_compatible` (reserved for future extension).
- `auth_mode`: `auto | bearer | none`.
- `chat_path`: defaults to `/chat/completions`.
- `local`: optional boolean override for local detection in `auto` mode.

Example: local multi-framework pool

```yaml
llms:
  sglang_qwen:
    provider: sglang
    model: Qwen/Qwen2.5-7B-Instruct
    base_url: http://127.0.0.1:30000/v1
    auth_mode: none

  vllm_llama:
    provider: vllm
    model: meta-llama/Llama-3.1-8B-Instruct
    base_url: http://127.0.0.1:8001/v1
    auth_mode: none

  lmstudio_local:
    provider: lmstudio
    model: local-model
    base_url: http://127.0.0.1:1234/v1
    auth_mode: none
```

Optional routing memory (retrieval-augmented routing):

```yaml
memory:
  enabled: true
  # If omitted/empty, defaults to: ~/.llmrouter/openclaw_memory.jsonl
  path: "${HOME}/.llmrouter/openclaw_memory.jsonl"
  top_k: 10
  retriever_model: "facebook/contriever-msmarco"
  device: "cpu"          # "cpu" or "cuda"
  max_length: 256
  max_query_chars: 500
  max_prompt_chars: 200
  per_user: false
```

Notes:
- Memory persists (query -> selected model) pairs to a JSONL file and retrieves top-k similar past queries.
- Currently, memory is only used to augment `router.strategy: llm` (the router LLM prompt gets the retrieved pairs).
- The first run will download the retriever model if it is not present (requires outbound internet).

Optional media understanding (converts images/audio/video to text descriptions):

```yaml
media:
  enabled: true
  # Uses Together AI API (same key as api_keys.together)
  api_key_env: "TOGETHER_API_KEY"
  base_url: "https://api.together.xyz/v1"
  # Vision model for image/video understanding (Qwen3-VL recommended for serverless)
  vision_model: "Qwen/Qwen3-VL-8B-Instruct"
  # Audio transcription model (Whisper)
  audio_model: "openai/whisper-large-v3"
  # Prompts
  image_prompt: "Describe this image concisely in 2-3 sentences."
  video_prompt: "Describe what you see in these video frames."
  # Video processing
  video_max_frames: 4
  # Max description length
  max_description_chars: 500
```

**Media Understanding Pipeline:**

When media understanding is enabled, the router automatically:

1. **Detects media** in incoming messages (supports two formats):
   - OpenAI multimodal format: `{"type": "image_url", "image_url": {"url": "data:image/..."}}`
   - OpenClaw format: `[media attached: /path/to/file (mime/type) | optional_url]`

2. **Converts media to text**:
   - Images → Vision API (Qwen3-VL) → text description
   - Audio → Whisper API → transcript
   - Video → Frame extraction + Vision API → description

3. **Replaces media placeholders** in the message content with the generated text description, so the LLM can understand what's in the image/audio/video.

4. **Uses the processed text** for routing decisions (which model to select).

**Example flow:**
```
User sends: "[media attached: photo.png (image/png)]"
           ↓
Vision API: "The image shows a cat sitting on a couch."
           ↓
LLM receives: "[Image: The image shows a cat sitting on a couch.]"
           ↓
LLM responds based on the image description
```

Notes:
- Image descriptions use Together AI's Qwen3-VL-8B-Instruct vision model (serverless compatible).
- Audio transcription uses Together AI's Whisper Large v3.
- Video processing extracts key frames and describes them (requires `opencv-python`).
- If memory is also enabled, the combined text (original + media description) is stored for retrieval.

LLMRouter strategy config (two equivalent forms are supported):

```yaml
router:
  strategy: llmrouter
  llmrouter:
    name: knnrouter
    config_path: configs/model_config_test/knnrouter.yaml
    model_path: saved_models/knnrouter.pt  # optional
```

or:

```yaml
router:
  strategy: llmrouter
  name: knnrouter
  config_path: configs/model_config_test/knnrouter.yaml
  model_path: saved_models/knnrouter.pt  # optional
```

### Model Fallback

If the model chosen by the router fails to respond (the provider is down, rate-limited, or returns an error), the server can retry the request against other configured models before giving up. Set an ordered list under `router.fallback_models`; the server tries each one in turn, skipping the model that already failed:

```yaml
router:
  strategy: llm
  fallback_models:
    - llama-3.1-8b
    - qwen-2.5-7b
```

Only names present in `llms` are used. If `fallback_models` is empty or omitted, behavior is unchanged: a failed call returns an error immediately.

## Routing Strategies (Built-in + Original LLMRouter)

OpenClaw Router supports two routing families:

1) Built-in strategies (configure `router.strategy` in `openclaw_router/config.yaml`)
- `random`: pick a backend model randomly (optionally weighted)
- `round_robin`: cycle through backend models
- `rules`: keyword rules (map keywords to specific backend models)
- `llm`: use a small "router LLM" to choose the backend model
  - Uses `router.provider`, `router.base_url`, `router.model`, and (if needed) your `api_keys` to call the router LLM.

Routing granularity note:
- Routing is request-level, not agent-identity-level by default.
- If OpenClaw sends `model: "auto"`, the router decides per request content.
- If you need strict per-agent binding, configure explicit model names in OpenClaw per agent, or add deterministic `rules`.

2) Original LLMRouter ML-based routers (learned routers)
- Set `router.strategy: llmrouter` and choose a router name (for example `knnrouter`, `mlprouter`, `svmrouter`, etc.).
- With the startup script, you can pass the router name via `-r` (and optionally `--router-config`):

```bash
./scripts/start-openclaw.sh -r knnrouter
./scripts/start-openclaw.sh -r knnrouter --router-config configs/model_config_test/knnrouter.yaml
```

Router config auto-detection:
- If you do not pass `--router-config`, OpenClaw Router will try (in order):
  - `configs/model_config_test/<router>.yaml`
  - `custom_routers/<router>/config.yaml`
  - `configs/model_config_train/<router>.yaml`

List routers:
- `llmrouter list-routers`
- Or: `./scripts/start-openclaw.sh --list-routers`

For training ML routers and preparing router configs, follow the main project docs:
- See `../README.md`

## Command Line Options

### Startup Script (`./scripts/start-openclaw.sh`)

This is the recommended entry point for Slack: it starts both OpenClaw Router and the OpenClaw Gateway.

| Option | Description |
|--------|-------------|
| `-c, --config FILE` | Config file path (default: `openclaw_router/config.yaml`) |
| `-p, --port PORT` | Router port (default: `8000`) |
| `-r, --router NAME` | Router name or built-in strategy (e.g. `random`, `llm`, `knnrouter`) |
| `--router-config FILE` | Router-specific config file path (optional; auto-detected if omitted) |
| `--no-gateway` | Don't start OpenClaw Gateway |
| `--no-prefix` | Don't add model name prefix to responses |
| `--list-routers` | List available original LLMRouter routers |
| `-h, --help` | Show help message |

### Router-Only Alternatives (No Bash Script)

If you prefer not to use the bash script:

```bash
# Start Router (OpenAI-compatible API)
llmrouter serve --config openclaw_router/config.yaml

# Optional: use an original LLMRouter ML router
llmrouter serve --config openclaw_router/config.yaml \
  --router knnrouter \
  --router-config configs/model_config_test/knnrouter.yaml

# Start OpenClaw Gateway in another terminal (requires gateway.mode=local)
openclaw gateway run --bind loopback --port 18789 --force
```

## API Endpoints

Once running, the following endpoints are available:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/chat/completions` | POST | Chat completions (OpenAI-compatible) |
| `/v1/models` | GET | List available models |
| `/v1/chat/ws` | WS | Real-time streaming chat (WebSocket) |
| `/routers` | GET | List available routing strategies |

Quick checks:

```bash
curl -s http://127.0.0.1:8000/health

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello from OpenAI-compatible client"}]
  }'

# Streaming
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello (streaming)"}],
    "stream": true
  }'

# WebSocket Streaming
# See tests/test_websocket.py for a Python client example
```

## Usage Examples

```bash
# Use random built-in strategy
./scripts/start-openclaw.sh -r random

# Use round-robin built-in strategy
./scripts/start-openclaw.sh -r round_robin

# Use LLM-based routing (uses a small model to decide)
./scripts/start-openclaw.sh -r llm

# Use an original LLMRouter ML router
./scripts/start-openclaw.sh -r knnrouter

# Use an ML router with an explicit router config file
./scripts/start-openclaw.sh -r knnrouter --router-config configs/model_config_test/knnrouter.yaml

# Custom port
./scripts/start-openclaw.sh -p 9000

# Start without OpenClaw Gateway (API only)
./scripts/start-openclaw.sh --no-gateway
```

## Architecture

### Full Integration (Slack via OpenClaw)

```text
+----------------+     +------------------------+     +------------------+
| Slack User     | --> | OpenClaw Gateway       | --> | OpenClaw Router    |
| (mobile/desktop)|     | (port 18789, socket)  |     | (port 8000, /v1)  |
+----------------+     +------------------------+     +--------+---------+
                                                          |
                               +--------------------------+--------------------+
                               |                          |                    |
                               v                          v                    v
                         +-----------+              +-----------+        +-----------+
                         | LLaMA     |              | Qwen      |        | Mistral   |
                         +-----------+              +-----------+        +-----------+
```

### Standalone Mode (API only)

```text
+-----------------------------------------------+
| Client (curl / OpenAI SDK / OpenClaw, etc.)   |
+---------------------------+-------------------+
                            |
                            v
+-----------------------------------------------------------+
| OpenClaw Router (port 8000)                                |
| - built-in: random / round_robin / rules / llm            |
| - original LLMRouter routers: knnrouter / mlprouter / ...  |
+---------------------------+-------------------------------+
                            |
        +-------------------+-------------------+
        v                   v                   v
  +-----------+       +-----------+       +-----------+
  | LLaMA     |       | Qwen      |       | Mistral   |
  +-----------+       +-----------+       +-----------+
```

## Directory Structure

```text
LLMRouter/
  scripts/
    start-openclaw.sh        # Start OpenClaw Router + OpenClaw Gateway
    stop-openclaw.sh         # Stop all services
  configs/
    openclaw_example.yaml
  openclaw_router/
    __init__.py             # Module exports
    __main__.py             # CLI entry point
    server.py               # FastAPI server
    config.py               # Configuration classes
    config.yaml             # Main configuration file
    routers.py              # Routing strategies
    README.md               # This file
  custom_routers/
    randomrouter/           # Example custom router
      config.yaml
```

## Logs

- Router log: `/tmp/openclaw.log`
- Gateway log: `/tmp/openclaw-gateway.log`

View logs in real time:

```bash
tail -f /tmp/openclaw.log
```

## Troubleshooting

### "No API provider registered for api: undefined" (OpenClaw)

Your `models.providers.<name>.api` is missing. Add:
- `models.providers.openclaw.api = "openai-completions"`

### Slack receives nothing / bot never replies

Checklist:
- Is OpenClaw running and connected (Socket Mode)?
  - Check `/tmp/openclaw-gateway.log`
- Did you install (or reinstall) the Slack app after updating scopes?
- Did you add the right bot events (e.g. `app_mention`, `message.im`)?
- Is the bot in the channel (invite it)?
- Did OpenClaw print a pairing code that still needs approval?

### "OpenClaw: access not configured" (pairing)

If OpenClaw prints a pairing code, approve it on the server:

```bash
openclaw pairing approve slack <pairing-code>
```

### Port already in use

If the Router port is already taken, the startup script will try to stop the previous Router process. You can also stop services manually:

```bash
./scripts/stop-openclaw.sh
```

### Router not loading (ML router)

If you start with `-r <router>` and it does not load:
- Try passing `--router-config` explicitly (see examples in this README).
- Or check that one of the auto-detected paths exists:
  - `configs/model_config_test/<router>.yaml`
  - `custom_routers/<router>/config.yaml`
  - `configs/model_config_train/<router>.yaml`

### "No module named openclaw_router"

This usually means you are using a Python environment that does not have this repo installed, or you are running from a different directory/session.

Fix:
- Activate the correct virtual environment, and install the repo in editable mode:

```bash
pip install -e .
```

### Slow responses with `router.strategy: llm`

The `llm` strategy makes two upstream calls:
1) Call the router LLM to decide which backend model to use
2) Call the selected backend model to generate the final answer

If you want lower latency, use `random` or `round_robin`.

### Internal network / VPN

Your phone network does not need direct access to your server.

Slack integration works as long as the server running OpenClaw can reach Slack over the internet (outbound). Socket Mode does not require a public inbound webhook URL.

### "openclaw requires Node >= ..."

Upgrade Node.js, then reinstall OpenClaw if needed:
- Prefer `nvm` and use a modern Node version that OpenClaw requires.
