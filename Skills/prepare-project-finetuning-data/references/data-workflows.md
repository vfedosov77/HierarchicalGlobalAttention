# Dataset workflow and existing code

## Format and supervision

`Experiments/hga_qlora/src/data/agentic_dataset.py` accepts one JSON object per line:

- Chat: `{"messages": [...], "tools": [...], "meta": {...}}`. The Qwen3.5 loader labels assistant content and terminating assistant marker; system/user/tool messages are context with `labels=-100`. Tool call `function.arguments` strings are normalized to objects. Use complete, correctly ordered calls and responses and the same tool names/schemas as serving.
- Text: `{"text": "..."}`. All tokens plus the added end marker are supervised. This is suitable for repository text but does not teach tool interactions.

Do not invent chain-of-thought or tool outputs from a diff. The current project traces do not yet contain tool calls. Once trace collection supports them, retain real sessions as originally observed, remove secrets, check licenses and access rights, and keep provenance in `meta`.

## Existing commands

Run these from `Experiments/hga_qlora` after `make setup`; substitute private paths outside the repository. Inspect each script's `--help` and the selected model before running because the default split helpers use Qwen3.5-4B.

```bash
make dataset-check DATASET=/private/sessions.jsonl
PYTHONPATH=src .venv/bin/python scripts/split_train_val.py \
  --input /private/sessions.jsonl \
  --train-out /private/train.jsonl --val-out /private/val.jsonl \
  --model Qwen/Qwen3.5-4B --max-tokens 4096 --n-train 500 --n-val 50
```

`validate_dataset.py` is for full tool-calling records and has additional checklist requirements; its tool checks may flag legitimate single-turn PR edit records. `split_train_val.py` renders via `build_example`, rejects overlength/no-target rows, and chooses the newest 50 validation records and the 500 immediately before them. Its strict time-shift printout must be checked; equal or missing `meta.date` makes the desired guarantee fail. It splits **rows**, so PR-derived rows from one PR may cross the boundary. Group by PR first or use one record per PR when that matters.

For the Rdiff commits corpus, `scripts/build_rdiff_split.py` sorts training records oldest first and reserves the 50 newest C/C++/Bazel examples. The `train_qlora_rdiff.yaml` config uses `shuffle: false` to preserve that order. This script has Rdiff-specific language and metadata filters, so do not point it at arbitrary projects without adapting them.

For the two-host Qwen3.8 path, use its own `src/data/agentic_dataset.py` and tokenizer. The dual controller copies the declared local data to both ranks; it can train full trajectories up to its configured `seq_len` and drops overlength records as whole records. Its `enable_thinking` setting changes prompt rendering and should match the desired deployment mode.

## Holdout and audit

Check source and output SHA-256, record IDs/PR IDs, earliest/latest dates, exact rendered token lengths, valid assistant target count, duplicates, and overlap with any benchmark. Keep all examples from a PR together. For a private project, use a time-shifted validation window and independent executable tasks from later commits. Avoid training on the expected patch or hidden tests of benchmark tasks.

The checked-out `Experiments/hga_qlora/scripts` contains splitting, validation, analysis, and training helpers. It does not contain a PR API downloader or an agent-session extractor. The optional `op_distillation` project under `Qwen3_8_27B/hga_qlora-master-dual` operates on **already-built** trajectories; it is not a PR extractor.
