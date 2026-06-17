# Qwen 3 0.6B Long-Context Fine-tuning

*Work in progress — scripts and results will be added here.*

This directory will contain experiments applying Hierarchical Global Attention to [Qwen 3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) for extended context (target: 128K tokens).

The approach extends the base model's 32K context by slowing down the low-frequency RoPE rotation rates by 4×.
The HA module already separates high-frequency and low-frequency RoPE pairs in the chunk/group summary computation,
so adding a frequency-scaling knob to those low-frequency pairs is the main required change.
The hierarchical router then handles retrieval across the full 128K context while full attention remains local.

Benchmarks will compare inference speed and memory usage against the standard Qwen 3 0.6B at long contexts.
