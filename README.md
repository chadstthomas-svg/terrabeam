<p align="center"><img src="assets/hero.svg" alt="terrabeam: any model, with the library attached" width="100%"></p>
<p align="center"><img src="assets/tiles.svg" alt="1 command · any GGUF · open library · MIT" width="100%"></p>

**terrabeam** runs a language model on your own computer and connects it to the
[Tablinum](https://huggingface.co/tablinum) knowledge library and a persistent memory. Everything sits behind an ordinary
OpenAI-style endpoint, so it works with the apps you already use.

When you ask a question, terrabeam searches the library (Wikipedia, case law, software documentation, Stack Exchange,
regulations, open textbooks), passes the three best passages to the model, and the model answers from them. Tablinum
models are trained to read those passages and to say so when they don't contain the answer. Any other model gets the
same passages as clearly marked sources.

> **Preview.** The Moon-Beam 8B release and these tools are `-Preview`. The library is now
> [Tablinum-3](https://huggingface.co/datasets/tablinum/tablinum-3): 34 tables, 210M passages, with new agent tables
> (app manuals, web and MCP specs, sysadmin and developer Q&A, scientific code) and home repair.

## Quick start

**1. Install**

```sh
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/chadstthomas-svg/terrabeam/master/install.sh | sh

# Windows (PowerShell)
irm https://raw.githubusercontent.com/chadstthomas-svg/terrabeam/master/install.ps1 | iex

# or from source (Rust 1.85+; on Linux older than glibc 2.38 add --no-default-features)
cargo install --git https://github.com/chadstthomas-svg/terrabeam terrabeam
```

**2. Run a model**

```sh
terrabeam run tablinum/Moon-Beam-8B-A1B-GGUF-Preview
```

That one command:
- picks the right quant for your GPU or CPU;
- downloads the model, the matching [llama.cpp](https://github.com/ggml-org/llama.cpp) build and the `lite` library pack;
- serves **http://127.0.0.1:8080/v1**.

Downloads are checked against their SHA256 and resume if interrupted. Later starts take seconds.

**3. Use it** with any OpenAI client:

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What does PHP json_encode return when it fails?"}]}'
```

The response header `x-engram-sources` lists the passages the model was given.

## Any model, any app

| You have… | Run | Then point your app at |
|---|---|---|
| A Tablinum model | `terrabeam run tablinum/<model>-GGUF-Preview` | `http://127.0.0.1:8080/v1` |
| Any GGUF on Hugging Face | `terrabeam run unsloth/Qwen3-8B-GGUF` (or a local `.gguf`) | `http://127.0.0.1:8080/v1` |
| Ollama, LM Studio, vLLM, llama.cpp, a cloud API | `terrabeam proxy --upstream http://127.0.0.1:11434` | `http://127.0.0.1:8090/v1` |
| Claude Code, Claude Desktop, Cursor, Cline, opencode | `terrabeam mcp` (an MCP tool: `search_library`) | see the guide |

Step-by-step setup for **Open WebUI**, **LM Studio**, **Ollama**, **Claude Code**, **Claude Desktop**, **Cursor**,
**Continue**, **Cline**, **opencode** and **Python** is in **[docs/harnesses.md](docs/harnesses.md)**.

## The library (engram packs)

| Pack | Size | Tables |
|---|---|---|
| `lite` (default) | ~6.6 GB | software docs, app manuals (VS Code, PowerShell, Godot, Blender, LibreOffice, GIMP), web + MCP specs, scientific code, home repair, Stack Exchange (STEM, humanities, finance, law), Britannica 1911, US Code, eCFR |
| `agent` | ~12.4 GB | software docs, app manuals, web + MCP specs, scientific code, developer and sysadmin Q&A (Stack Overflow tooling, Super User, Ask Ubuntu, Server Fault, Unix & Linux …) |
| `standard` | ~31 GB | lite + case law, EUR-Lex, CRS reports, developer and sysadmin Q&A |
| `full` | ~76 GB | standard + English Wikipedia + open textbooks |
| `wiki` | ~34 GB | English Wikipedia only |

The packs nest (`lite` ⊂ `standard` ⊂ `full`), so moving up downloads only the difference. Choose a pack with
`--pack standard`, or pick tables with `--tables software,law`. `terrabeam pack agent` downloads a pack ahead of time;
`terrabeam mcp --pack agent` is the one to give a coding agent. The pack list is read from the published `packs.json`,
so new tables arrive without a new terrabeam release.

Every table keeps its own license (CC BY-SA, CC BY, public domain and others); see
[Tablinum-3 SOURCES](https://huggingface.co/datasets/tablinum/tablinum-3/blob/main/SOURCES.md). Health text is never
included in a pack, whatever its license.

To turn the library off for one request, send `"engram": false` or the header `X-Engram: off`. To run without it, use
`--no-engram`.

## Memory

```sh
terrabeam run tablinum/Moon-Beam-8B-A1B-GGUF-Preview --memory long-term
```

With memory on, terrabeam saves short notes of your questions and the answers. When a later question relates to them, it
adds the matching notes for the model: in front of the library passages for Tablinum models (as the user's own earlier
words, never as a library source), otherwise in the system prompt. A note matches when it shares a distinctive word with
the question, such as a name or place; common words like "what" or "my" don't count.

- **Modes:**
  - `session`: this conversation only, useful when an app trims long chats;
  - `long-term`: across all your conversations.
- **Safety:** API keys, passwords and bearer tokens are redacted before anything is saved. Retrieved passages are never
  saved. Notes are always marked as untrusted reference material, never as instructions.
- **Managing it:**
  - notes are plain JSONL under `~/.cache/terrabeam/memory/`;
  - `GET /memory` lists them, `DELETE /memory` clears them, and `DELETE /memory/<id>` removes one;
  - to skip memory for one request, send `"memory": false` or the header `X-Memory: off`.

## Built for agents

- **The model can search for itself.** With `--library-tool`, the model gets a `search_library` tool (plus `remember` and
  `recall` with `--memory`). terrabeam runs those calls and keeps going until the model answers. Your app only sees
  its own tools; a call to one of them comes back as usual, and each library search shows up as a line in the
  reasoning stream. Turn it off for one request with `"library_tool": false` or the header `X-Library-Tool: off`.
- **Reasoning is kept across turns.** Many apps drop the model's earlier reasoning when they send the next message.
  Tablinum models are trained to see it, and multi-turn agent work falls apart without it: in tau2 airline tasks
  Moon-Beam scored 0.71 with reasoning kept and 0.05 without. `terrabeam run` stores each answer's reasoning (and its
  hidden library steps) for the conversation and puts it back when the app leaves it out (`--client-reasoning` turns
  this off; for `proxy`, turn it on with `--keep-reasoning`).
- **Adapters per app.** A model's `tablinum.json` can list LoRA adapters, for example one trained for a particular
  agent app. `terrabeam run <model> --adapter <name>` loads it on top of the same GGUF.
- **The harness has a version.** Every response carries `x-terrabeam-harness`, for example
  `tb1;inject=tablinum;k=3;tool=none;reasoning=keep`: the injection layout, passages per question, library tool and
  reasoning handling. When you run a model differently from how it was trained, terrabeam says so at startup.

## Training data and evaluation in the same harness

- **`--record DIR`** writes every exchange to `DIR/exchanges-<day>.jsonl`: the request as sent, every message the model
  saw (passages, restored reasoning, library steps), its full answer with reasoning and tool calls, and the sources used.
  It is opt-in and local; terrabeam never uploads it. User and conversation ids are stored as hashes, and exchanges that
  used a health table are not written.
- **`terrabeam env run`** runs a task file through the same proxy users get, grades each answer and writes a summary
  with per-task solve rates (for calibrating a model before training it). Each task can be run in four variants:

  | Variant | The model gets |
  |---|---|
  | `live` | the library as served (plus the library tool when on) |
  | `oracle` | passages that contain the answer |
  | `distract` | related passages that don't contain it |
  | `closed` | no library |

  Grading is correct +1, "I don't know" 0, wrong −1. Every proxy also serves `POST /env/render` and `POST /env/grade`,
  so a trainer can use the exact prompts and rewards. Task format and the builder are in [envs/](envs/).

## Commands

| Command | What it does |
|---|---|
| `terrabeam run <model>` | Runs a model with the library and memory. Flags: `--quant`, `--pack`, `--tables`, `--memory`, `--no-engram`, `--library-tool`, `--client-reasoning`, `--adapter`, `--record`, `--cpu`, `--ctx`, `--port`, `-- <llama-server args>` |
| `terrabeam proxy --upstream <url>` | Puts the library and memory in front of a server you already run (same flags; `--keep-reasoning` to restore reasoning) |
| `terrabeam env run --tasks <file> --upstream <url>` | Runs tasks through the harness, grades them, and writes per-task solve rates |
| `terrabeam mcp` | Runs the library (and `--memory`) as an MCP server on stdio |
| `terrabeam pack <name>` | Downloads a pack ahead of time |
| `terrabeam serve` | Runs the library as an HTTP service (`/search`, `/inject`); this is how Rig uses it |

## How it works

- **Retrieval:** pure Rust. It reads the published tables directly:
  - Faiss IVF-SQ8 indexes;
  - Tantivy BM25 for the keyword half of each search;
  - parquet passages;
  - static embeddings for the query;
  - a MiniLM cross-encoder through ONNX Runtime.

  Results match the reference Python implementation (rerank top-1 on 60 of 60 test queries), and it runs about 3× faster.
- **Candidates across many tables:** each search pools the best vector hits from every table (they share one embedding
  model, so their scores compare) and adds each table's top keyword hit, then re-ranks. Earlier versions divided a fixed
  budget evenly between tables, which with a 29-table library left one candidate per table. On 200 library questions the
  answer reached the top 3 passages 82% of the time, up from 54%. `--pool split` restores the old behavior.
- **Models:** run by the pinned upstream llama.cpp release. The MTP draft head is used when the model ships one.
- **Proxy:** adds passages to the last user message:
  - **Tablinum models:** in the exact layout they were trained on;
  - **other models:** as clearly marked source blocks.

  With `--library-tool`, the proxy also runs the model's library calls in a loop. Everything else is passed through
  unchanged, streaming included.

See [DESIGN.md](DESIGN.md) for details.

## Credit and license

MIT. Use it, change it, ship it. Keep the copyright notice, and if terrabeam or the Tablinum research helps you, a
mention or a thank-you is appreciated. See [CITATION.cff](CITATION.cff).

Built by [Tablinum](https://huggingface.co/tablinum).
