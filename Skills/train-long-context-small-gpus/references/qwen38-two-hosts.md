# Tested Qwen3.8-27B dual V100 path

The operational source of truth is `Qwen3_8_27B/hga_qlora-master-dual/skills/qwen38-v100-dual-training/SKILL.md` and its four references. Resolve commands relative to `Qwen3_8_27B/hga_qlora-master-dual`; inspect current YAML and inventory before use.

## Measured architecture

- Two Linux PCs, one V100 16 GB GPU each, PyTorch distributed world size 2. This is training with inter-rank autograd boundary gradients, not llama.cpp inference RPC.
- Rank 0 owns token embedding and decoder `[0,33)`; rank 1 owns `[33,64)`, final norm/head, and MTP. The older 36/28 split OOMed when rank 0 installed its approximately 2.37 GiB embedding.
- Both ranks load a prequantized 4-bit base, retain only their stage's weights in GPU memory, and train LoRA. Inputs saved in host RAM are replayed one decoder layer at a time during backward. Only activation/gradient tensors cross the network per segment.
- HGA wraps the 16 full-attention layers; native GDN/linear-attention state remains on its own path. Cold routed KV is held in host RAM with a bounded VRAM hot cache.
- The current `configs/train_qlora_qwen38_27b_dual.yaml` sets `seq_len: 262144`, `train_seg_len: 1024`, `split_layer: 33`, `cache_location: ram`, NCCL, `loss_chunk_size: 128`, and positive MTP loss. A 4096-token smoke spans four segments. The sequence setting is an eligibility cap; inspect run metadata and actual rendered lengths to report what a given run exercised. These values belong to this checkpoint/hardware, not every model.

## Reproduce the existing workflow

Inspect `deployment/cluster.json`, `configs/train_qlora_qwen38_27b_dual.yaml`, and the smoke YAML. Supply matching model, official MTP safetensors, exact dataset, both hosts, and output directories through the inventory. The controller supports read-only preflight and deployment dry run:

```bash
python3 deployment/deploy.py --cluster deployment/cluster.json preflight
python3 deployment/deploy.py --cluster deployment/cluster.json --dry-run deploy --setup
```

For an authorized run, use the controller's `deploy --setup`, then `start` or `launch`; choose the smoke YAML and a separate smoke output first. It starts rank 1 before rank 0. After the smoke, use `status`, `fetch`, and `scripts/merge_dual_checkpoint.py` as documented in the existing skill. Inspect both stage checkpoints and nonzero finite LoRA weights. Pin model revision and hash model/index, MTP file, dataset, YAML/inventory, and source version. Never silently disable MTP or stream layer weights in the loop.

If training on an instruct checkpoint damages tools, HGA alone will not resolve that. The base-to-instruct transfer demonstrated for Qwen3.5-4B is a separate model-specific experiment; no matching Qwen3.8-27B base-to-instruct transfer is established here. Use agentic evaluation to decide whether the resulting Qwen3.8 adapter is deployable.
