#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Hierarchical Global Attention (HGA) CPU runtime for Qwen3.8-27B inference.
 *
 * Exact-token path: chunk/group *summaries* only *select*; softmax attends real
 * token K/V (sinks + local window + routed/opened spans + the active chunk).
 *
 * One-level vs two-level use the *same* routed chunk set. One-level opens every
 * group in those chunks (~8 % tokens). Two-level opens half of those groups
 * (~4 % tokens). Groups can be faster because the attended working set is
 * smaller, at the cost of extra gather.
 */

typedef enum hga_dtype {
    HGA_F32 = 0,
    HGA_F16 = 1,
} hga_dtype;

/* Token KV + QK inner product. Softmax / routing stay F32.
 * I8 is the default on Xeon Gold 6148 (AVX-512BW integer MAC, no VNNI). */
typedef enum hga_prec {
    HGA_PREC_F16 = 0, /* F16 KV, F32 dots (reference) */
    HGA_PREC_I8  = 1, /* INT8 KV + INT8 QK dots, per-vector scale */
} hga_prec;

typedef struct hga_config {
    int n_q_heads;     /* 24 for Qwen3.8-27B gated attention */
    int n_kv_heads;    /* 4 */
    int head_dim;      /* 256 */
    int rotary_dim;    /* 64 (partial RoPE) */
    int chunk_size;    /* 64 */
    int group_size;    /* 16; must divide chunk_size */
    int keep_first;    /* sink chunks, always token-level */
    int keep_last;     /* local chunks, always token-level */
    int levels;        /* 1 or 2 */
    float frac_l1;     /* target attended-token fraction for 1-level, ~0.08 */
    float frac_l2;     /* target attended-token fraction for 2-level, ~0.04 */
    float theta;       /* RoPE theta (1e6) */
    float mixed_rope_threshold; /* 0.5, matches KvRouter */
    int n_threads;
    int max_seq;
    hga_prec prec;     /* HGA_PREC_I8 by default */
} hga_config;

typedef struct hga_stats {
    int n_kv;
    int n_closed_chunks;
    int n_selected_chunks;   /* same by construction for 1-level and 2-level */
    int n_opened_groups;
    int n_attended_tokens;   /* per last query, including windows + current */
    float sparsity;          /* n_attended / max(n_kv,1)  (1 = dense) */
    double ms_route;
    double ms_attn;
} hga_stats;

typedef struct hga_session hga_session;

hga_config hga_config_qwen38_27b(int levels, int max_seq, int n_threads);

hga_session * hga_session_create(const hga_config * cfg, int n_layers);
void          hga_session_free(hga_session * s);
void          hga_session_reset(hga_session * s);

const hga_config * hga_session_config(const hga_session * s);
int                hga_session_n_layers(const hga_session * s);
int                hga_session_n_kv(const hga_session * s, int layer);

/* llama.cpp graph-input hook: start_pos of the current ubatch. */
void hga_set_ubatch(hga_session * s, int start_pos, int n_tokens);
int  hga_ubatch_start(const hga_session * s);
int  hga_ubatch_n(const hga_session * s);

/* Append n_new tokens at start_pos. Layout of k_rope/k_raw/v:
 *   [n_kv_heads, n_new, head_dim]  (head-major, token-major, dim-contiguous)
 * k_raw is pre-RoPE (after k-norm); k_rope is post-RoPE. */
void hga_append(hga_session * s, int layer, int start_pos, int n_new,
                const void * k_rope, const void * k_raw, const void * v,
                hga_dtype dtype);

/* Close any chunk that is now full. Call *after* hga_attend so the active chunk
 * is still visible to its own tokens (matches KvRouter.decode_block). */
void hga_close_full_chunks(hga_session * s, int layer);

/* Attend n_q queries at start_pos. q layout [n_q_heads, n_q, head_dim].
 * out is always f32, same layout. Causal: query i only sees keys <= start_pos+i. */
void hga_attend(hga_session * s, int layer, int start_pos, int n_q,
                const void * q, hga_dtype q_dtype, float * out, hga_stats * stats);

/* Convenience: append + attend in one call (typical decode / prefill block). */
void hga_forward(hga_session * s, int layer, int start_pos, int n_q,
                 const void * q, const void * k_rope, const void * k_raw, const void * v,
                 hga_dtype dtype, float * out, hga_stats * stats);

/* Compute top-k chunk count used at a given closed-chunk count (identical for both levels). */
int hga_topk_chunks(const hga_config * cfg, int n_closed);
int hga_topk_groups(const hga_config * cfg, int n_closed, int topk_chunks);

#ifdef __cplusplus
} /* extern "C" */
#endif
