# Qwen 3 0.6B Long-Context Fine-tuning

*Work in progress — scripts and results will be added here.*

This directory will contain experiments applying Hierarchical Global Attention to [Qwen 3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) for extended context (target: 128K tokens).

The approach extends the base model's 32K context by slowing down the low-frequency RoPE rotation rates by 4×.
The HA module already separates high-frequency and low-frequency RoPE pairs in the chunk/group summary computation,
so adding a frequency-scaling knob to those low-frequency pairs is the main required change.
The hierarchical router then handles retrieval across the full 128K context while full attention remains local.

Benchmarks will compare inference speed and memory usage against the standard Qwen 3 0.6B at long contexts.

pip install -U "torch>=2.3" "transformers>=4.51.0" datasets accelerate safetensors

python -m ExistingModelFineTuning.Qwen3LongContext.replace_qwen_attention_finetune \
  --output-dir ./qwen3_06b_global_attention_ft \
  --context-len 32768 \
  --loss-chunk-len 2048 \
  --stage1-steps 1000 \
  --stage2-steps 1000
