#pragma once

/* Decode-only host K/V GEMV: hidden-dim shard across physical cores.
 * Declared here; qwen35.cpp reaches it via llama-hga.h.
 */

struct llm_graph_context;
struct ggml_tensor;

#ifdef __cplusplus
ggml_tensor * hga_build_kv_proj(
        llm_graph_context * gctx,
        ggml_tensor * x,
        ggml_tensor * wk,
        ggml_tensor * wv,
        int il);
#endif
