#pragma once

/* Glue between llama.cpp Qwen3.5/3.8 gated attention and the HGA CPU runtime.
 * Copied into llama.cpp/src by scripts/apply_hga.py.
 */

#include "hga-weight-swap.h"
#include "llama.h"

#include <vector>

struct llm_graph_context;
struct llm_graph_input_attn_kv;
struct ggml_tensor;
struct llama_cparams;
struct llama_hparams;
struct llama_context_params;
struct llama_context;
struct ggml_backend_sched;
typedef struct ggml_backend_sched * ggml_backend_sched_t;
struct hga_session;

/* First GPU backend on the scheduler, or nullptr. */
ggml_backend_t hga_sched_gpu_backend(ggml_backend_sched_t sched);
/* Force a tensor (and its op) onto that GPU. No-op if there is no GPU. */
void hga_pin_gpu(ggml_backend_sched_t sched, ggml_tensor * t);
/* Pin when either packed pipeline is active (Q/K/V GEMM, gate, o_proj). */
void hga_pin_gpu_pack(ggml_backend_sched_t sched, ggml_tensor * t, int32_t phase);
/* Prefill-only (q-norm stays CUDA; decode leaves q/k-norm on CPU for HGA). */
void hga_pin_gpu_prefill(ggml_backend_sched_t sched, ggml_tensor * t, int32_t phase);
/* Decode-only extras that stay on CUDA (attn/post/output norms). Not q/k-norm. */
void hga_pin_gpu_decode(ggml_backend_sched_t sched, ggml_tensor * t, int32_t phase);
/* Diagnostic equivalent of the old "pin every module" experiment. With
 * HGA_PREFILL_PIN_ALL=1 every named Qwen graph tensor passed through cb() is
 * assigned to CUDA during PREFILL. HGA CPU staging keeps its explicit backend
 * assignments inside hga_build_full_attn(). */
void hga_pin_gpu_prefill_probe(ggml_backend_sched_t sched, ggml_tensor * t, int32_t phase);
/* Explicit H2D of a host tensor onto CUDA as a contiguous 2D F32 copy.
 * Pinning descendants is not enough: with --no-op-offload the scheduler
 * expands the CPU-pinned HGA chain through gate and o_proj. */
ggml_tensor * hga_copy_to_gpu(llm_graph_context * gctx, ggml_tensor * src, const char * name);
/* Materialize a contiguous host copy. Pinning a GPU view to CPU aliases the
 * device buffer as a host pointer and segfaults in the HGA kernel. */
ggml_tensor * hga_copy_to_cpu(llm_graph_context * gctx, ggml_tensor * src, const char * name);

void hga_cparams_from_ctx_params(llama_cparams & cparams, const llama_context_params & params);
void hga_runtime_init(llama_cparams & cparams, const llama_hparams & hparams);
void hga_runtime_free(llama_cparams & cparams);

struct llama_ubatch;
/* Grow a short ubatch to n_pad_to tokens so the ggml graph is always
 * n_ubatch-wide (prefill dummy n=2 → configured width). Pad tokens repeat the last real
 * token. Returns true if padding was applied. */
bool hga_ubatch_pad_to(llama_ubatch & ub, uint32_t n_pad_to);
/* Prefill: one logits row (last real token) so n_outputs stays 1 and the
 * fixed-width graph can be reused for every chunk. */
void hga_ubatch_prefill_one_output(llama_ubatch & ub);

/* Owned storage for a padded llama_batch (generate n=1/2 → K+1). */
struct hga_llama_batch_pad {
    std::vector<llama_token>     token;
    std::vector<float>           embd;
    std::vector<llama_pos>       pos;
    std::vector<int32_t>         n_seq_id;
    std::vector<llama_seq_id>    seq_id_data;
    std::vector<llama_seq_id *>  seq_id;
    std::vector<int8_t>          logits;
    llama_batch                  batch{};
};

/* After hga_swap_ensure has set generate n_ubatch=K+1, pad a short batch so
 * VERIFY and MTP share one n_tokens=n_outputs graph. */
bool hga_maybe_pad_decode_batch(const llama_cparams & cparams,
                                const llama_batch & in,
                                uint32_t n_embd,
                                hga_llama_batch_pad & mem);

uint32_t hga_ubatch_padded_n_real();
void     hga_ubatch_pad_reset();

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

/* Decode: hidden-dim-sharded host K/V GEMV. nullptr → caller uses ggml mul_mat. */
ggml_tensor * hga_build_kv_proj(
        llm_graph_context * gctx,
        ggml_tensor * x,
        ggml_tensor * wk,
        ggml_tensor * wv,
        int il);

/* True if Kcur/Vcur were filled from the sharded GEMV (decode, n_tokens=1). */
bool hga_build_kv_gemv_pair(
        llm_graph_context * gctx,
        ggml_tensor * wk,
        ggml_tensor * wv,
        ggml_tensor * x,
        ggml_tensor ** Kcur,
        ggml_tensor ** Vcur,
        int il);

void hga_l2_after_hga(hga_session * sess, int hga_il);
void hga_l3_prefetch_layer(hga_session * worker_sess,
                           const hga_session * data_sess, int hga_il);
