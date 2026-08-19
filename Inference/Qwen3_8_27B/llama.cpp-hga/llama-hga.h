#pragma once

/* Glue between llama.cpp Qwen3.5/3.8 gated attention and the HGA CPU runtime.
 * Copied into llama.cpp/src by scripts/apply_hga.py.
 */

struct llm_graph_context;
struct llm_graph_input_attn_kv;
struct ggml_tensor;
struct llama_cparams;
struct llama_hparams;
struct llama_context_params;
struct llama_context;

void hga_cparams_from_ctx_params(llama_cparams & cparams, const llama_context_params & params);
void hga_runtime_init(llama_cparams & cparams, const llama_hparams & hparams);
void hga_runtime_free(llama_cparams & cparams);

/* Replace build_attn for a full-attention Qwen3.5/3.8 layer. Returns [n_embd, n_tokens] F32. */
ggml_tensor * hga_build_full_attn(
        llm_graph_context * gctx,
        llm_graph_input_attn_kv * inp,
        ggml_tensor * Q,      /* post-RoPE  [head_dim, n_head,    n_tokens] */
        ggml_tensor * K_rope, /* post-RoPE  [head_dim, n_head_kv, n_tokens] */
        ggml_tensor * V,      /*            [head_dim, n_head_kv, n_tokens] */
        ggml_tensor * K_raw,  /* pre-RoPE   [head_dim, n_head_kv, n_tokens] */
        float kq_scale,
        int il);
