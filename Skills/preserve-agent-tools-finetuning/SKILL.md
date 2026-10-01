---
name: preserve-agent-tools-finetuning
description: Choose and evaluate LoRA fine-tuning methods that aim to retain an agent model's tool use while adapting it to private project data. Covers base-to-instruct adapter transfer, implemented entropy/KL masking, and a proposed future-attention masking extension.
---

# Preserve agent tool behavior during project fine-tuning

Project-only SFT can change tool selection, argument formatting, reasoning, and finish behavior even when the data is well formed. Treat this as a measured regression risk, not a theorem that every SFT run breaks tools or that RL is the only possible repair. Evaluate the unchanged deployed model and every candidate under the same tool schema, prompts, budget, seed policy, and executable tasks.

## Choose a method

1. **Matching base and instruct checkpoints available:** train a LoRA on the **base** checkpoint; serve the resulting adapter on the matching **post-trained/instruct** checkpoint. The tested Qwen3.5-4B example did this. Verify architecture, tensor names/shapes, tokenizer, RoPE, and adapter target coverage before transfer; do not assume adapters transfer across model sizes or releases. The Qwen3.5 trainer accepts `tokenizer_model` separately from `model`, and the GGUF exporter accepts Base-trained adapters. Read [references/base-to-instruct.md](references/base-to-instruct.md).
2. **No matching base or transfer fails:** compare ordinary SFT with entropy/KL selective masking on the post-trained model, plus a random-mask control at the same effective mask rate. The local trainer currently masks selected tokens from CE only; it does not implement the paper's regularization losses. Read [references/eksft-and-attention.md](references/eksft-and-attention.md).
3. **High future-attention fact tokens must be retained:** the proposed top-20%-attention exception is an **unmerged extension**. Follow the design and tests in that reference before using it. Do not claim it has measured accuracy or tool-preservation results.

The approaches can be combined only after individual baselines establish what each contributes. HGA is compatible with either training path when its model-specific implementation exists; it solves a different resource constraint.

## Acceptance evidence

Use an executable tool harness, including read/edit/test/finish gates, on tasks held out by project history. Check malformed tool calls, repeated tool loops, turn limits, tests passing after the final edit, and final completion. Include a project knowledge/edit metric and a long-context task if those are goals. Compare with matched budgets and more than one seed or repeat before making a reliability claim. Record contamination and source overlap checks.

The copied [Qwen3.5-4B comparison report](references/qwen35-agentic-example.md) is a worked example with its own limitations. It is evidence for that run, not a universal guarantee. The cited EKSFT [paper](https://arxiv.org/abs/2605.29303) studies selective SFT and subsequent RL, not this project's tool-use outcomes.
