---
name: train-long-context-small-gpus
description: Plan and operate HGA plus QLoRA long-context fine-tuning on limited VRAM, including the tested two-host 16 GB V100 Qwen3.8-27B trainer and model-specific porting. Use for training memory, context, rank placement, smoke runs, and model adaptation.
---

# Train long context on small GPUs

Hierarchical Global Attention (HGA), developed by this team, routes from chunk and group summaries to **exact token K/V** in selected history. It retains the pretrained attention projections and can keep cold KV in host RAM or NVMe. With segment-wise backpropagation and QLoRA, it enables long-context fine-tuning while bounding the active GPU working set. Read [the HGA papers and evidence](references/hga-papers.md) for the algorithm, reported experiments, and scope of each result.

For the team's Qwen3.8-27B setup, **two separate 16 GB V100 hosts** hold disjoint resident model stages, with CPU activation replay and a 1024-token live segment. The current dual config allows 262,144-token trajectories; the team's long-context work also records a 251K-token training trajectory on the Qwen3.5 path. Read [references/qwen38-two-hosts.md](references/qwen38-two-hosts.md) to run or alter the dual setup.

The HGA method can be adapted to other transformer models, including long contexts, after model-specific integration. Hardware capacity still depends on weights, architecture, RAM/NVMe, and the training implementation; two 16 GB cards are the tested Qwen3.8 deployment, not a universal hardware requirement. For another model, follow [references/porting-hga.md](references/porting-hga.md). HGA addresses attention/KV memory; read [the agentic behavior skill](../preserve-agent-tools-finetuning/SKILL.md) to select a method for retaining tool behavior after SFT.

## Working sequence

1. Identify the exact model and checkpoint revision, tokenizer/chat template, GPU type and number, RAM, inter-host bandwidth, intended sequence length, dataset format, and desired output adapter.
2. Start from an implementation already covering that architecture. For Qwen3.8-27B, use the existing project-local dual training skill as the operational authority. For Qwen3.5-4B on one 16 GB Turing GPU, use `Experiments/hga_qlora` and its README/configs. Do not interchange those commands or assume their checkpoints are compatible.
3. Quantify frozen-weight placement, embeddings, final head, optimizer/LoRA, logits, activations, HGA hot banks, NCCL buffers, and host RAM. Choose the rank split and segment size from measured **peak** memory, not parameter count alone.
4. Validate data with the training tokenizer and train-only target mask. Smoke the longest useful sample and at least two segments; for the tested dual V100 profile, run its four-segment 4096-token smoke. Check both ranks, finite main and MTP losses, global gradient norm, memory margin, and synchronized checkpoints.
5. Train only after the smoke succeeds. Preserve the original base and LoRA separately. Record effective config, model/data hashes, output location, checkpoint step, validation, and agentic regression results.

Do not silently weaken the tested dual profile's RAM KV semantics, resident stage weights, MTP requirement, or rank synchronization. Any model port must establish its own contracts instead of copying Qwen-specific constants.
