# terrabeam

terrabeam runs a Beam model the way it is meant to be run, with one command. It picks the right GGUF for the machine,
starts a pinned upstream llama.cpp build, attaches the model's engram knowledge library, and serves an OpenAI-compatible
endpoint.

```sh
terrabeam run tablinum/Moon-Beam-8B-A1B-GGUF      # -> http://127.0.0.1:8080/v1
```

> Private preview. Not published yet: no PyPI, crates.io or release binaries. The license is a placeholder (Apache-2.0
> in Cargo.toml) until it is decided.

## Commands

- **`run <hf-repo | local folder>`**
  - Reads the repo's `tablinum.json` (quants with SHA256, the MTP draft head, the llama.cpp release tag, sampling
    defaults, engram packs).
  - Detects CUDA, Metal, Vulkan or CPU.
  - Downloads only what is missing. Downloads resume and are checked.
  - Runs llama-server on an internal port, behind a proxy that adds retrieved passages to the last user message in the
    layout the model was trained on.
  - Per request: `"engram": false` in the body, or the header `X-Engram: off`.
  - Debugging: `GET /engram/search?q=`; the response header `x-engram-sources` lists the passages used.
  - Flags: `--quant`, `--pack`, `--engram-root`, `--tables`, `--no-engram`, `--no-mtp`, `--cpu`, `--ctx`,
    `--llama-server`. Anything after `--` is passed to llama-server.
- **`serve`**: the engram as a service.
  - `GET /health` and `GET /search?q=&k=&tables=`.
  - `POST /inject` with `{messages, format: tablinum|sources, k, tables}`, which returns the rewritten messages and the
    sources used.
  - Runs as Rig's `rig-engram.service`, where the router adds passages for every model.

## Crates

- **`crates/engram`**: retrieval in pure Rust. It reads the existing tables as they were built, with no re-export:
  - a faiss `IVF*_HNSW32,SQ8` + OnDiskInvertedLists reader. Centroids are searched exactly. The ids match faiss when
    faiss is also given exact centroid search;
  - tantivy 0.26 BM25;
  - parquet passages;
  - static-retrieval-mrl-en-v1 query keys;
  - an ms-marco-MiniLM-L6-v2 cross-encoder through onnxruntime fp32, statically linked, with a candle fallback.
    Top-1 matches torch on 60/60 queries.
- **`crates/terrabeam`**: the CLI.

## Build

```sh
cargo build --release          # one binary; links onnxruntime statically
cargo test
```

The parity probes in `crates/engram/examples` compare each stage with the Python stack. See `DESIGN.md` for the design
and for how terrabeam fits into Rig.
