# terrabeam - one-command runner + engram service (Rust, 11-04)

Goal (Chad 11-02, memory release-out-of-the-box): a user or an agent gets a Beam model *as it is meant to be run* with one command:
right GGUF for the hardware, engram library attached, OpenAI endpoint on localhost. No card reading, no redirects.
11-04 additions (Chad): **written in Rust**, and **built into Rig (the Omnigent install)** so the engram is a checkbox for every model.

    terrabeam run tablinum/Moon-Beam-8B-A1B-GGUF     # -> http://127.0.0.1:8080/v1 (OpenAI-compatible, engram on)
    terrabeam mcp tablinum/Moon-Beam-8B-A1B-GGUF     # MCP stdio server: search_library tool for agent hosts
    terrabeam serve --tables /mnt/optane/engram/...  # engram service only (Rig: replaces router/engram_serve.py on :18395)
    uvx terrabeam ...                                # same binary from PyPI (maturin bin wheel); also cargo install / GitHub releases

**Name (Chad 11-03): `terrabeam`** (PyPI + crates.io free 11-03/11-04; GitHub user `terrabeam` is someone else's; trademark check
"Terabeam" before launch). Lite-Beam / Moon-Beam / Sun-Beam models, TerraBeam = the ground they run on.

## Why Rust
One static binary per platform (Linux x86_64/aarch64, macOS arm64, Windows x86_64): no Python, no torch, no faiss wheels, no CUDA
wheels - the GPU work stays in the pinned upstream llama.cpp binary it downloads. Distribution: GitHub release assets, `cargo install`,
and a maturin `bindings = "bin"` wheel so `uvx terrabeam` keeps working.

## Workspace (~/optane-llm/terrabeam)
- `crates/engram` (lib) - retrieval, pure Rust, reads the EXISTING tables as built (no re-export):
  - `ivf.rs`: faiss `IVF{n}_HNSW32,SQ8` + OnDiskInvertedLists reader (IP metric). Coarse step = brute-force IP over the centroids
    (16k x 256 = ~1 ms; the HNSW graph is skipped), SQ8 decode with the trained vmin/vdiff, list data mmapped from lists.ivfdata.
  - `bm25.rs`: tantivy (same major as tantivy-py 0.26 that built the indexes), en_stem, one term per token, pid stored field.
  - `passages.rs`: parquet (pid = file_idx << 32 | row; key files by stored pid too - 11-03 papers_humft fix), row-group LRU.
  - `embed.rs`: static-retrieval-mrl-en-v1 (HF `tokenizers` crate + safetensors, mean of token rows, 256-d MRL, L2 norm).
  - `rerank.rs`: cross-encoder ms-marco-MiniLM-L6-v2 in candle (pure Rust BERT); fall back to `ort` only if parity fails.
    Length bucketing + top-N cap (11-03: 2.8 s/query at 100 x 256 tok on 4 threads in Python).
  - `lib.rs`: `Engram::retrieve(q, k, tables)` = hybrid candidates (half vector, half BM25 per table) -> union -> rerank; confident
    fast lane (top vector IP >= 0.85 skips rerank, as the Rig sidecar).
- `crates/terrabeam` (bin) - `run` / `serve` / `mcp` / `pack`:
  - **serve**: HTTP (axum) `GET /health`, `GET /search?q=&k=&tables=` (drop-in for engram_serve.py's API), `POST /inject`
    (messages + format -> rewritten messages + the sources used).
  - **run**: manifest (`tablinum.json`) -> hardware detect (nvidia-smi / Metal / Vulkan / CPU) -> pick quant -> fetch pinned ggml-org
    llama.cpp release + GGUF + MTP + engram pack (SHA256SUMS) into the user cache -> start llama-server on an internal port -> OpenAI
    proxy that injects on the last user turn. `X-Engram: off` / `"engram": false` per request; `/engram/search` for debugging.
  - **mcp**: stdio MCP server, tool `search_library(query, k)`.
- Parity tests vs the Python stack on EngramBench items: first-stage ids (IVF + BM25) identical, rerank top-1 = 60/60, top-3 overlap 1.0.

## Injection formats
- `tablinum` (models trained on it: Moon-Beam-8B, Moonlight R-rounds; gen_r1 / engram_eval v1): the LAST user turn becomes
  `Reference passages (retrieved automatically; they may be incomplete or irrelevant):\n[1] ...\n\n[2] ...\n\nQuestion: {q}`.
  Must match engram_eval.py byte for byte (verify against its builder, not from memory).
- `sources` (every other model): Rig's existing nonce-spotlighted `<source n="nonce">` blocks + preface (rigKnows.ts), moved server-side.
- Only a user turn is rewritten (never a tool result); earlier turns stay as sent; streaming / tools / reasoning pass through.

## Rig (Omnigent install) integration - "engram as a checkbox for all models"
Today: Rig Chat's "Rig knows" chip injects client-side in one composer only (MainAgentSurface; NewChatDialog deferred), via the Python
sidecar `rig-engram.service` (:18395, engram_serve.py, 5 tables, 2-3.5 s warm).
Target:
1. `rig-engram.service` runs `terrabeam serve` (same port, same /health + /search API -> dashboard engram tile + rigKnows keep working),
   all tablinum-3 tables available, per-request table routing.
2. **router.py does the injection** for any backend: request flag (`"engram": true|false` body field, or `X-Engram` header) overrides
   the per-model default in registry.json (`engram: {default: off|on, format: tablinum|sources, k: 3}`). Router strips the flag
   before forwarding, calls `/inject` with a latency budget (on timeout: send unmodified, as rigKnows does), and logs which sources were
   used (trace row + response header `X-Engram-Sources`).
3. **Checkbox**: Rig Chat chip sends the flag instead of building blocks client-side (both composers get it for free); dashboard Models
   view gets a per-model "Engram" default checkbox (writes registry.json through the existing registry API). Agents/API clients use the
   body field or header.
4. Rollout: shadow first (terrabeam serve on a spare port, compare /search vs the Python sidecar), then swap the unit, then router flag.

## Repo files (every release repo)
`tablinum.json` (manifest), `AGENTS.md` (the one command, endpoint, MCP, what the bare GGUF lacks), same pointers in GGUF metadata.

## Tests before release
Linux here (CUDA A2000, CPU-only, Vulkan on the 2060); Windows + macOS on tailnet devices if available, else clean VMs. Parity: runner
endpoint vs the eval harness on 100 EB items; Rust retrieval vs Python (above).

## Harness for training (0.3.0, 11-18)
Idea taken from harness adapters (RL'd LoRAs per agent harness, HF harness-hub 11-16): train a model in the exact harness it is
served in. terrabeam is that harness for Beam models, so it gains what training needs:
- **canonical history** (history.rs): per user + conversation, the model's real messages (reasoning, hidden library-tool steps) are
  stored under keys from what the client received (tool-call ids, hash of the visible content) and put back when a client trims
  them. On for `run`, opt-in for `proxy`. Motivation: Moon-Beam tau2 airline 0.711 with reasoning kept vs 0.045 as clients send it.
- **library tool loop** (proxy.rs, libtools.rs): `search_library` (+ `remember` / `recall` with memory) offered as OpenAI tools;
  terrabeam executes them and calls the model again (max `--max-steps`, the last call without its tools). Streaming forwards
  content/reasoning deltas live and holds tool-call deltas; the client sees only its own tools (terrabeam's are dropped when the
  model mixes both in one turn). The MCP server shares the same tool definitions and result layout.
- **retrieval cache**: identical query + k returns the same passages (agent loops re-send the user turn on every call).
- **harness id** `tb1;inject=<tablinum|sources|off>;k=<k>;tool=<library|none>;reasoning=<keep|client>`: response header, record
  field, manifest `harness.id` (startup note when served differently). Manifest `adapters` + `--adapter` -> llama-server
  `--lora-scaled file:scale`; an adapter may carry its own harness id.
- **record** (record.rs): `--record DIR` -> one JSON line per exchange (request, trajectory incl. reasoning and tool steps, tools,
  visible response, sources, usage). Local, opt-in, hashed ids, health-table exchanges skipped.
- **env** (env.rs): `terrabeam env run` drives tasks through a private proxy (same code path) in four controls (live / oracle /
  distract / closed), rule-graded (+1 / 0 / -1), summary with retrieval recall@3/@10 and per-task solve rates; `/env/render` and
  `/env/grade` on every proxy for trainers. Task builder: envs/build_tasks.py.

## Candidate pool (0.3.0, 11-19)
engram.py's first stage gives each table `ncand / n_tables` candidates (half vector, half BM25). Fine for 3-5 routed tables,
but terrabeam serves whole packs without a router: 29 tables x ncand 48 -> one vector hit and NO BM25 per table. Recall of
the gold answer on 200 tablinum-envs tasks (29 tables, crates/engram/examples/recall.rs, same cross-encoder for all):
split48 r@3 53.5 / r@10 56.5 (29 cand/query); split192 84.0 / 87.5 (174); global g8-32-l1 82.0 / 85.5 (61); g16-64-l2 84.5 / 88.0 (122).
Default is now `Pool::Global` (8 vector hits per table pooled, best 2/3 * ncand kept by inner product - one embedding model,
so comparable - plus each table's top BM25 hit). End to end, Moon-Beam preview (Q4_K_M, tablinum-envs eval split, 2 attempts):
live index +38.6 -> +46.1, and with --library-tool +34.1 -> +71.4 (correct 53.2% -> 83.4%).
