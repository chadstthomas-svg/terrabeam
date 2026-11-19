# tablinum-envs: tasks for training and scoring models in the terrabeam harness

A model trained to use the library should be trained in the same setup it is served in: the same injection layout,
tools, restored reasoning and memory. `terrabeam env run` sends tasks through that exact proxy and grades the answers.
This folder holds the task format and the builder for **tablinum-envs**, the task set made from Tablinum passages.

## Task format (one JSON object per line)

```json
{"id": "fa-law-1a2b3c4d5e6f", "kind": "fact", "split": "train", "table": "law",
 "question": "…", "gold": "…", "aliases": ["…"],
 "passages": {"oracle": ["…", "…", "…"], "distract": ["…", "…", "…"]},
 "source": {"pid": 123, "title": "…"}, "local_only": false}
```

| Kind | What it tests |
|---|---|
| `fact` | One fact stated in one library passage. |
| `multihop` | Two linked wiki passages. The question names the first hop only by its relation, so a single search on the question rarely finds the answer. |
| `memory` | A personal fact told in an earlier conversation (`prior`, sent first with memory on) and asked about in a new one. |

Each task can be run in four variants (`--controls`):

| Variant | The model gets | Answers the question |
|---|---|---|
| `live` | the library as served (and the library tool when on) | how good is the whole setup? |
| `oracle` | the source passage plus two related ones, shuffled | can the model read the answer when it's there? |
| `distract` | three related passages that don't contain the answer | does it say "I don't know", or guess? |
| `closed` | no library | what does it know on its own? |

Grading: the gold answer or an alias appearing in the answer (normalized: case, accents, punctuation, whole words) is
correct (+1). "I don't know" and similar is 0, and anything else is −1. The summary reports each variant's index
(correct − incorrect) and, for `live`, how often the gold answer was in the top 3 and top 10 retrieved passages. That
separates "retrieval missed it" from "the model didn't use it". Per-task solve rates go under `calibration`, so you can
pick the tasks a model solves sometimes (0 < solved < attempts) for training.

## Building it

`build_tasks.py` runs these steps (each can be resumed; `all` runs them in order):

1. **sample:** passages from each non-health table.
2. **write:** a writer model (Claude Haiku via the `claude` CLI, which only sees passages) writes one question per
   passage. A question is kept only when its answer appears word for word in the passage and not in the question.
3. **passages:** oracle and distract sets, from a running `terrabeam serve` over the same tables.
4. **multihop:** two-hop questions over wiki pairs.
5. **memory:** synthetic memory tasks.
6. **assemble:** `tasks.jsonl` and `stats.json`.

```sh
terrabeam serve --root /mnt/optane/engram --tables wiki,law,… --port 18396 --text-cap 0 --embed-dir … --ce-dir … &
python3 envs/build_tasks.py all --root /mnt/optane/engram --out tablinum-envs-v1 --search http://127.0.0.1:18396
```

- **Health:** health tables (`health`, `healthp`, `psych`, `fdalabel`, `cdcj`) are never sampled.
- **Held out for `eval`:** whole tables (`sehum`, `scicode`, `ecfr`, `papers_bio`, `homerepair`, `philo`), every fourth
  multihop task and every fifth memory task.
- **Local only:** tasks from recipe-only tables are marked `local_only`. They are fine for local training but shouldn't
  be redistributed.

## Scoring a model

```sh
llama-server -m model.gguf --jinja -c 65536 -np 4 --kv-unified --port 18380 … &
terrabeam env run --tasks tablinum-envs-v1/tasks.jsonl --split eval --upstream http://127.0.0.1:18380 \
  --format tablinum --engram-root /mnt/optane/engram --attempts 2 --out eval.jsonl --request '{"max_tokens": 4096}'
```

Add `--library-tool` to score the model searching for itself.
