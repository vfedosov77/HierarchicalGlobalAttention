# Local PR commit ranges to supervised edit examples

Use this when a Git checkout contains the commits for merged PRs, but actual agent tool transcripts are unavailable. It creates `messages` records accepted by the Qwen3.5 chat loader: user gives the PR title and pre-change file; assistant gives the complete post-change file. It does **not** reconstruct tool calls, reviews, reasoning, intermediate edits, or a full multi-file agent session.

## Input

Create a private JSONL manifest with one object per merged PR:

```json
{"number": 123, "title": "Fix iterator boundary", "base_sha": "<full pre-PR commit>", "head_sha": "<full merged-result commit>", "merged_at": "2026-08-01T12:00:00+00:00"}
```

The two SHAs must be in the local repository and `base_sha` must be an ancestor of `head_sha`. For squash merges, `head_sha` can be the squash commit in the mainline and `base_sha` its first parent. Confirm that the range contains **only the desired PR change**; an interval containing unrelated merges contaminates examples. The script selects added/modified UTF-8 files, skips binaries and oversized files, and writes one example per file. No network or hosted API is used.

```bash
python3 Experiments/ModelsFineTuningSkills/prepare-project-finetuning-data/scripts/pr_commits_to_edits.py \
  --repo /private/project-checkout \
  --manifest /private/merged-prs.jsonl \
  --output /private/pr-edits.jsonl \
  --max-file-bytes 131072
```

The output is created with owner-only permissions. If it already exists, choose a new path; the script refuses to replace it. It prints selected/skipped counts and a SHA-256. Keep the manifest and output private. Inspect representative records for task ambiguity, secrets, generated files, and giant unchanged content. A title like “fix stuff” is insufficient supervision; improve the manifest title or use a human-authored task statement before training.

The exporter retains `meta.pr_number`, `meta.date`, both commits, path, and status. Split its output by **whole PR** with the companion helper:

```bash
python3 Experiments/ModelsFineTuningSkills/prepare-project-finetuning-data/scripts/split_pr_groups.py \
  --input /private/pr-edits.jsonl \
  --train-out /private/pr-train.jsonl --val-out /private/pr-val.jsonl \
  --n-val-prs 20
```

It refuses an overlapping date boundary and refuses to overwrite outputs. Then check rendered token lengths with the model-specific tokenizer; the byte cap is only an early filter. A PR with several files remains one validation unit. Do not run the row-based `split_train_val.py` on these outputs again.

Tool-call collection is future work for this project's traces. When implemented, collect real sessions from the authorized tool runner, convert them to the OpenAI message/tool schema, and validate call/result pairing. Do not synthesize a tool trace from final PR files.
