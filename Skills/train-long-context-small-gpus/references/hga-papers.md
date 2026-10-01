# HGA papers and evidence

These papers describe a method developed by this team. Keep their published experiments separate from later local deployments and team tests.

## HGA mechanism

[Hierarchical Global Attention: Drop-In Exact-Token Routing for Pretrained Long-Context Transformers](https://arxiv.org/abs/2606.30709) introduces a replacement for dense causal attention that retains pretrained Q/K/V/O projections and norms without adding learned router weights. RoPE-aware key summaries select relevant historical chunks and, where configured, groups inside them. The final softmax uses the original exact token K/V from selected regions; summaries do not become attention output tokens. First/sink and recent chunks stay visible, while middle history is selected by content. Full cold K/V can live in RAM or NVMe; GPU memory holds model weights, active tokens, summaries, and a bounded routed working set.

The paper demonstrates a pretrained Qwen3-30B-A3B FP8 model with HGA on one 32 GB RTX 5090 without retraining, including a 64K needle test at three depths. It reports small dense-versus-routed loss gaps for tested routing budgets. These are sparse-attention compatibility and retrieval measurements, not a promise of exact equivalence for every prompt or of production serving latency.

## Fine-tuning evidence

[Long-Context Fine-Tuning with Limited VRAM](https://arxiv.org/abs/2607.15105) combines HGA, 4-bit QLoRA, 2,048-token truncated backpropagation segments, and RAM-backed historical KV. On Qwen3-8B and one 16 GB Quadro RTX 5000, dense training fits 2,048 tokens and OOMs at 4,096; HGA completes a 16,384-token training run with a 15.28 GB peak. Its separate feasibility sweep also completes 32,768 tokens with bounded caches. The article evaluates an HGA-trained adapter through 131,072 tokens. At the matched 2K training length, held-out dense-readout loss was 2.7405 nats for HGA training and 2.7383 for dense training, with matched data order, seed, and hyperparameters. The adapter may be served with ordinary dense attention where memory allows; HGA serving is a separate path.

The team's later local records describe a successful approximately 251K-token training trajectory on Qwen3.5-4B (`Experiments/hga_qlora/memory-bank/progress.md`). Its `docs/benchmarks.en.html` reports a PR corpus with records up to about 253K tokens and a 396-step run using `seq_len: 262144`. Current Qwen3.5 and dual Qwen3.8 configs allow 262,144-token records. The team has tested working 256K contexts. For any particular run, use its rendered sample lengths and logs to distinguish a configured context cap from a completed long example.

The fine-tuning paper reports a causal side channel from chunk-shared routing decisions becoming measurable after roughly 100–200M training tokens, and stable fine-tuning through about 100M in its experiments. The team's tests did not find a significant effect through 100M tokens. This matters when adapting the method for much longer runs or pretraining; it is not evidence that a few-million-token proprietary fine-tune is compromised.
