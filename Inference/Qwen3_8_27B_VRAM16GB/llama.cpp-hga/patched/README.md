# Patched llama.cpp sources

Copies of llama.cpp files changed for `--offload-rs` / hybrid memory, taken
from a checkout after `scripts/apply_hga.py`.

The patcher is the source of truth: `scripts/apply_hga.py` reapplies these
edits onto a llama.cpp tree. These files are here so the diff can be
reviewed without the gitignored `third_party/` checkout.

| File | Change |
|---|---|
| `llama-memory-hybrid.h/.cpp` | `offload` → `offload_attn` + `offload_recr` |
| `llama-memory-hybrid-iswa.h/.cpp` | same |
| `llama-cparams.h` | `offload_rs` (default true) |
| `llama-memory-recurrent.cpp` | R/S device log is INFO |
| `qwen35.cpp` | HGA hook; mode-dependent pins |
| `llama-context.cpp` `graph_get_cb` | do not force `norm` onto the layer GPU when `--no-kv-offload` |
