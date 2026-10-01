# Train on a matching base, serve on its post-trained counterpart

## Why try this

The project's hypothesis is that updating LoRA against a base checkpoint can inject repository knowledge while leaving the post-trained checkpoint's learned tool-use behavior intact when the adapter is applied at serving time. The base weights of the served model are never replaced by the training base; only the adapter delta is transferred. This is a **model-pair-specific empirical strategy**, not a guarantee that RL-trained parameters are untouched: the transferred LoRA still changes the post-trained model's effective weights during inference.

## Compatibility gate

Confirm that both checkpoints are the corresponding base and post-trained releases of the same model size and architecture. Compare tokenizer IDs and special tokens, chat template requirements, layer count, hidden size, target projection names and shapes, GQA/RoPE configuration, and adapter scaling. Use the **base weights** for training with the **same tokenizer settings used for the served model**, as in the team's Qwen3.5-4B run. A mismatched release can load partially or silently produce poor behavior. Refuse a transfer with missing/unexpected adapter targets, NaN weights, or an all-zero LoRA B path.

Freeze the base model; train LoRA on project data. Keep the adapter separate from both checkpoints. For HGA training, inspect the export path because routed attention can rename projections to `.orig.`; a stock PEFT merge onto the untouched model may miss those keys. The Qwen3.5 `Experiments/hga_qlora/src/eval/merge_adapter.py` handles its own mapping for a **merge into the training base**. Serving a LoRA on the post-trained model is a separate operation; do not merge the base model over it.

For a new Qwen3.5-4B run, start from a reviewed `Experiments/hga_qlora` YAML and private `dataset_full`/`dataset_val` files. Set `model=Qwen/Qwen3.5-4B-Base` and `tokenizer_model=Qwen/Qwen3.5-4B` in the YAML or with repeated `--set` options; the trainer uses the second value only for tokenization and chat rendering. Use the served model tokenizer for the split and validation scripts as well. Pin both revisions, verify token IDs and special tokens, render sample records with assistant targets, then run the trainer's `--smoke` before a full run. The current PR-derived records are single-turn edits; tool-call trace collection is future work.

From `Experiments/hga_qlora`, one smoke command is:

```bash
PYTHONPATH=src .venv/bin/python -m train.run_train \
  --config configs/train_qlora_codeedit.yaml --smoke \
  --set model=Qwen/Qwen3.5-4B-Base \
  --set tokenizer_model=Qwen/Qwen3.5-4B \
  --set dataset_full=/private/train.jsonl \
  --set dataset_val=/private/val.jsonl
```

Export the PEFT checkpoint with `Experiments/Qwen4BHosting/scripts/create_gguf.py --checkpoint /private/adapter/final --output /private/adapter-from-base.gguf`. It accepts metadata naming either `Qwen/Qwen3.5-4B-Base` or `Qwen/Qwen3.5-4B` as the training model, maps the LoRA against the pinned served-model config, and validates tensor pairs, F16 type, count, and LoRA alpha. Load the result with `Experiments/Qwen4BHosting/host_4b_model.sh /private/adapter-from-base.gguf`. Confirm adapter targets and run a serving smoke test before the agent benchmark.

## Qwen3.5-4B example

`Experiments/Qwen4BHosting/README.md` documents a local F16 `Qwen/Qwen3.5-4B` server and an adapter whose metadata names `Qwen/Qwen3.5-4B-Base` as the training base. The server loads the adapter at scale 1. The copied [comparison report](qwen35-agentic-example.md) compares the post-trained model alone with that same served model plus `adapter-f16_final.gguf` under matched prompts and limits. The report does not by itself prove how the adapter was trained; its metadata and hosting README provide that provenance.

In the reported 10 executable repository tasks, baseline finished 0/10 and the adapter 2/10. In 50 separate whole-file edit records, strict successes were 11 versus 14. Both variants still had many loops/turn limits. There was one attempt per task at one seed; training/validation overlap was not independently checked, and the standard external judge score was unavailable. Describe this as no observed tool-use regression in **that test**, not “works perfectly” or preserved tool behavior in general.

## Decision test for a new project

Run three variants where feasible: post-trained model alone; post-trained model with the base-trained LoRA; and post-trained model with a directly trained LoRA at matched project-data budget. Evaluate tool-call schema validity, task completion, regression tests, repeated loops, and held-out project edits. Keep generation settings, tool registry, context, output budget, and harness identical. Repeat or vary seeds if the observed difference is small. Prefer the transfer only if it improves the project objective without a material agentic regression.
