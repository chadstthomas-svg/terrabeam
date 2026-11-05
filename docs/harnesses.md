# Using terrabeam with your apps

terrabeam gives you an OpenAI-style endpoint with the Tablinum library (and optional memory) attached. Any app that can
talk to "an OpenAI-compatible server" can use it, and agent apps that support MCP can use the library as a tool.

| Setup | Command | Endpoint |
|---|---|---|
| Run a model with terrabeam | `terrabeam run tablinum/Moon-Beam-8B-A1B-GGUF-Preview` | `http://127.0.0.1:8080/v1` |
| Keep your current server, add the library | `terrabeam proxy --upstream <your server>` | `http://127.0.0.1:8090/v1` |
| Library as an MCP tool | `terrabeam mcp` | stdio (configured in the app) |

Add `--memory long-term` to any of them to turn on persistent memory. Add `--host 0.0.0.0` if the app runs in Docker or
on another machine.

---

## Open WebUI

1. Start terrabeam:
   - For OpenWebUI running on the same machine, natively:
     `terrabeam run tablinum/Moon-Beam-8B-A1B-GGUF-Preview --memory long-term`.
   - For OpenWebUI in Docker, add `--host 0.0.0.0`.
2. In Open WebUI, open **Admin Panel → Settings → Connections**. Under **OpenAI API**, click **+** and enter:
   - **URL:** `http://localhost:8080/v1`. From Docker, use `http://host.docker.internal:8080/v1`.
   - **API key:** anything, for example `terrabeam`.
3. Pick the model in a new chat. Answers now come with library passages. In the browser's network tab, the
   `x-engram-sources` response header shows which passages were used.
4. For memory per chat and per user, start Open WebUI with `ENABLE_FORWARD_USER_INFO_HEADERS=true`. terrabeam reads the
   forwarded `X-OpenWebUI-User-Id` and `X-OpenWebUI-Chat-Id` headers.

Already running models in Ollama behind Open WebUI? Use the proxy instead:
`terrabeam proxy --upstream http://localhost:11434`. Then add `http://localhost:8090/v1` as the connection.

## Ollama

```sh
terrabeam proxy --upstream http://localhost:11434
```

Point your app at `http://localhost:8090/v1` and use your Ollama model names as usual. terrabeam adds the passages and
forwards everything else to Ollama.

## LM Studio

1. In LM Studio, start the local server from the **Developer** tab. It defaults to port 1234.
2. Run `terrabeam proxy --upstream http://localhost:1234`.
3. Point your app at `http://localhost:8090/v1`.

## vLLM, SGLang, llama.cpp, or a cloud API

```sh
terrabeam proxy --upstream http://localhost:8000          # vLLM / SGLang
terrabeam proxy --upstream http://localhost:8081          # llama-server
terrabeam proxy --upstream https://api.example.com        # any OpenAI-compatible API
```

The client's `Authorization` header is passed through to the upstream server.

## Claude Code

```sh
claude mcp add tablinum -- terrabeam mcp --pack lite
```

Claude gets a `search_library` tool. With `--memory long-term` it also gets `remember` and `recall`. The first start
downloads the pack in the background.

## Claude Desktop

In `claude_desktop_config.json` (**Settings → Developer → Edit Config**):

```json
{
  "mcpServers": {
    "tablinum": { "command": "terrabeam", "args": ["mcp", "--pack", "lite"] }
  }
}
```

## Cursor

In `~/.cursor/mcp.json`, or `.cursor/mcp.json` in a project:

```json
{
  "mcpServers": {
    "tablinum": { "command": "terrabeam", "args": ["mcp", "--pack", "lite"] }
  }
}
```

## Cline / Roo Code

- **As a model:**
  - API Provider: **OpenAI Compatible**;
  - Base URL: `http://localhost:8080/v1`;
  - API key: anything;
  - Model ID: `Moon-Beam-8B-A1B`.
- **As a tool:** add an MCP server with command `terrabeam` and args `mcp --pack lite`.

## Continue

In `~/.continue/config.yaml`:

```yaml
models:
  - name: Moon-Beam 8B (Tablinum)
    provider: openai
    model: Moon-Beam-8B-A1B
    apiBase: http://localhost:8080/v1
    apiKey: terrabeam
```

## opencode

In `opencode.json`:

```json
{
  "provider": {
    "terrabeam": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:8080/v1" },
      "models": { "Moon-Beam-8B-A1B": {} }
    }
  },
  "mcp": { "tablinum": { "type": "local", "command": ["terrabeam", "mcp", "--pack", "lite"] } }
}
```

## Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="terrabeam")
r = client.chat.completions.with_raw_response.create(
    model="Moon-Beam-8B-A1B",
    messages=[{"role": "user", "content": "Who was the first Chaplain of the US Senate?"}],
)
print(r.parse().choices[0].message.content)
print("sources:", r.headers.get("x-engram-sources"))
```

## Per-request switches

| Switch | Body field | Header |
|---|---|---|
| Library off for this request | `"engram": false` | `X-Engram: off` |
| Memory off for this request | `"memory": false` | `X-Memory: off` |
| Name the conversation (memory) | `"user": "<id>"` | `X-Session-Id`, `X-User-Id` |

Debugging:
- `GET /engram/search?q=...` shows what the library returns.
- `GET /memory` lists the saved notes.
