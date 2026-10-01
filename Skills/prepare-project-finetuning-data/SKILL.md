---
name: prepare-project-finetuning-data
description: Build and validate private repository fine-tuning data from recorded agent sessions or merged PR commit ranges, then make a leakage-resistant train/validation split. Use before QLoRA or HGA training on a software project.
---

# Prepare project fine-tuning data

Produce a local JSONL corpus that the chosen trainer can render with the **same tokenizer settings used for serving**. Keep source provenance, hashes, dates, and a separate validation set. A Git diff supplies a known final edit, but does not contain an agent's tool calls or reasoning; never present PR-derived examples as recorded agent trajectories. The current project traces do not yet support tool calls; collecting them is future work.

## Choose the source

- When a tool-call trace collector becomes available, retain ordered `system`/`user`/`assistant`/`tool` messages and matching tool schemas as `{"messages": [...], "tools": [...], "meta": {...}}`, one full session per line. Preserve the tool registry the deployment will use. Read [references/data-workflows.md](references/data-workflows.md) for validation and splitting.
- If there are merged PRs and a local checkout, use [scripts/pr_commits_to_edits.py](scripts/pr_commits_to_edits.py) to turn verified base/head commit ranges into **single-turn whole-file edit examples**. Read [references/pr-edits.md](references/pr-edits.md) first. This path teaches project edits or facts; it does not teach tool use.
- If the goal is repository knowledge rather than edits, the Qwen3.5 trainer also accepts `{"text":"..."}` records and trains on all tokens. Keep factual content and provenance separate from conversational/tool trajectories.

The source `Experiments/hga_qlora` checkout has a dataset validator, splitters, and trainer loader. It has **no checked-in PR-to-agent-trajectory extractor**. Do not guess a command for one.

## Validate and split

Run the relevant trainer's dataset renderer to check trainable target counts and actual token lengths. For Qwen3.5, use `Experiments/hga_qlora/scripts/validate_dataset.py` for full tool sessions and `scripts/split_train_val.py` or `scripts/build_rdiff_split.py` for dated holdouts. The former validator's full-session checks intentionally do not fit the single-turn edit records produced by this skill; validate those via `data.agentic_dataset.build_example` and the split script.

Split by PR/commit or task, never by near-duplicate rows from the same PR. Use [scripts/split_pr_groups.py](scripts/split_pr_groups.py) for multi-file PR exports. Prefer a time-shifted holdout and check that no post-change content, tests, or answers leak into prompts or retrieval data. Record which source records were filtered for length or absent assistant targets. Keep private source, generated datasets, model outputs, and credentials out of public commits.

Before training, provide paths and a small audit: source count, eligible count, train/validation counts, rendered-token distribution, supervised-token counts, date ranges, source hashes, and duplicate/overlap checks. Then read [the low-VRAM training skill](../train-long-context-small-gpus/SKILL.md) for the hardware path and [the agentic behavior skill](../preserve-agent-tools-finetuning/SKILL.md) for adaptation strategy.
