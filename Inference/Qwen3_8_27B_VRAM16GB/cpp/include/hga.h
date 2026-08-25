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
 * Default is two-level (HGA-2). Both levels route the *same* chunk set.
 * One-level opens every group in those chunks (~8 % tokens). Two-level opens
 * groups of 16 inside them (~4 % tokens). The smaller working set is why HGA-2
 * is the default for Qwen3.8-27B on 16 GB GPUs.
 */

typedef enum hga_dtype {
    HGA_F32 = 0,
    HGA_F16 = 1,
} hga_dtype;

/* Token KV + QK inner product. Softmax / routing stay F32.
 * I8 is the default on AVX-512 hosts (integer MAC, no VNNI required). */
typedef enum hga_prec {
    HGA_PREC_F16 = 0, /* F16 KV, F32 dots (reference) */
    HGA_PREC_I8  = 1, /* INT8 KV + INT8 QK dots, per-vector scale */
} hga_prec;

/* Decode-time KV selection. Prefill routes each 64-token query chunk and each
 * query head independently, matching KvRouter's vectorized reference. Spec
 * verify (n_q = 2..8) routes once with mean-pooled Q and attends per query.
 * WAVE is the RetroInfer algorithm: segmented k-means on the
 * mid-context keys, retrieve a small cluster set, estimate the rest via
 * centroid / value-sum. Official RetroInfer targets Sm80 kernels, head_dim=128,
 * and full BF16 weights; this is that retrieval on Qwen3.8-27B dims
 * (head_dim=256) in the existing CPU path. */
typedef enum hga_router {
    HGA_ROUTER_HIER = 0,
    HGA_ROUTER_WAVE = 1,
} hga_router;

typedef struct hga_config {
    int n_q_heads;     /* 24 for Qwen3.8-27B gated attention */
    int n_kv_heads;    /* 4 */
    int head_dim;      /* 256 */
    int rotary_dim;    /* 64 (partial RoPE) */
    int chunk_size;    /* 64 */
    int group_size;    /* 16; must divide chunk_size */
    int keep_first;    /* sink chunks, always token-level */
    int keep_last;     /* local chunks, always token-level (default 7) */
    int levels;        /* 1 or 2; default 2 (HGA-2) */
    float frac_l1;     /* extra mid-chunk fraction on top of windows, ~0.08 */
    float frac_l2;     /* extra mid-group fraction on top of windows, ~0.04 */
    float theta;       /* RoPE theta (1e6) */
    float mixed_rope_threshold; /* 0.5, matches KvRouter */
    int n_threads;
    int max_seq;
    hga_prec prec;     /* HGA_PREC_I8 by default */
    hga_router router; /* HGA_ROUTER_HIER by default */
    float frac_retr;   /* wave retrieval budget, ~0.018 of clusters */
    float frac_est;    /* wave estimation budget, ~0.232 of clusters */
    int wave_cluster;  /* avg cluster size (16) */
    int wave_seg;      /* tokens per k-means segment (8192) */
    int wave_iters;    /* k-means iterations (3) */
    int wave_update;   /* rebuild after this many new closed tokens (1024) */
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
    double ms_pack;           /* selected KV → packed INT8, included in ms_attn */
    double ms_kernel;         /* score/softmax/value, included in ms_attn */
    /* Large-prefill attribution. The legacy ms_route value includes Q
     * pooling, the hierarchical router, span construction, and Q
     * quantization; these fields split that value without changing it. */
    double ms_q_pool;
    double ms_route_chunk_scan; /* chunk-summary loads + Q dot products */
    double ms_route_chunk_topk;
    double ms_route_group_scan; /* routed group-summary loads + Q dots */
    double ms_route_group_topk;
    double ms_route_other;      /* router setup/bookkeeping */
    double ms_span_keys;        /* span/key-list construction + tile setup */
    double ms_q_quant;
    double ms_attn_kernel;
    double ms_attn_merge;
    double ms_l2_load_fixed;  /* opt-in HGA_PROFILE_L2_LOAD diagnostic */
    double ms_l2_load_routed;
    double ms_l2_load_active;
    int n_fixed_tokens;         /* sink + local closed chunks */
    int n_routed_tokens;        /* selected middle groups/chunks */
    int n_active_tokens;        /* unfinished current chunk */
    int n_route_mid_chunks;     /* chunk summaries considered */
    int n_route_group_candidates;
    uint64_t route_chunk_bytes_unique;
    uint64_t route_chunk_bytes_logical; /* repeats across GQA Q heads */
    uint64_t route_group_bytes_unique;
    uint64_t route_group_bytes_logical;
    /* GPU-prefill route union diagnostics. Requests counts every routed
     * group once per (64-token chunk, query head); union/retained are summed
     * over KV heads. Scores are never compared across those routes. */
    int n_route_group_requests;
    int n_route_group_union;
    int n_route_group_retained;
    int n_route_topk_limit;
    float route_group_overlap;
    /* Raw routed-group fanout before the capacity limit. `head_uses` counts
     * distinct GQA query heads requesting a group anywhere in the physical
     * ubatch; `chunk_uses` counts distinct 64-token query chunks. Requests
     * count every (chunk, head) occurrence and can therefore exceed both. */
    int n_route_group_head_uses;
    int n_route_group_chunk_uses;
    int n_route_group_max_requests;
    int n_route_group_max_heads;
    int n_route_group_max_chunks;
    int n_route_history_selected; /* mandatory + routed rows, summed over KV heads */
    int n_route_history_max;      /* maximum before newest-history filler */
    /* United GPU-prefill CPU staging breakdown. These sum to ms_pack except
     * for small timer/bookkeeping error; ms_route remains separate. */
    double ms_prefill_append;
    double ms_prefill_close;
    double ms_prefill_union;
    double ms_prefill_scale_clear;
    double ms_prefill_kv_copy;
    double ms_prefill_pack_other;
    /* Packed L2 tiles (decode): reuse = key list unchanged, append = prefix+new. */
    int pack_rebuild;
    int pack_append;
    int pack_reuse;
} hga_stats;

/* Product execution has two fixed physical shapes.  `valid_mask` uses bit i
 * for physical lane i.  Production batches must have a contiguous valid
 * prefix: this prevents a later valid token from being assigned a position
 * after an invalid lane. */
typedef enum hga_mode {
    HGA_MODE_PREFILL = 0,
    HGA_MODE_VERIFY  = 1,
} hga_mode;

typedef struct hga_fixed_batch {
    hga_mode mode;
    uint32_t physical_count; /* PREFILL=512, VERIFY=3 */
    uint32_t real_count;     /* 1..physical_count */
    uint64_t valid_mask;     /* low real_count bits must be set */
} hga_fixed_batch;

/* Monotonic, session-owned counters.  They deliberately do not include
 * process-global padding state: target and MTP sessions remain independent. */
typedef struct hga_cache_metrics {
    uint64_t append_calls;
    uint64_t appended_tokens;
    uint64_t truncate_calls;
    uint64_t truncated_tokens;
    uint64_t chunk_closures;
    uint64_t packed_rebuilds;
    uint64_t packed_appends;
    uint64_t packed_reuses;
} hga_cache_metrics;

typedef struct hga_session hga_session;

hga_config hga_config_qwen38_27b(int levels, int max_seq, int n_threads);

hga_session * hga_session_create(const hga_config * cfg, int n_layers);
void          hga_session_free(hga_session * s);
void          hga_session_reset(hga_session * s);

const hga_config * hga_session_config(const hga_session * s);
int                hga_session_n_layers(const hga_session * s);
int                hga_session_n_kv(const hga_session * s, int layer);
void               hga_session_cache_metrics(const hga_session * s,
                                             hga_cache_metrics * out);

/* Returns non-zero only for the immutable product shapes and a contiguous
 * valid prefix.  The core accepts a smaller logical batch only through this
 * descriptor; there is no global real-token count. */
int hga_fixed_batch_validate(const hga_fixed_batch * batch);

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

/* Remove a speculative suffix.  This invalidates derived packed/routing state
 * but retains the allocated KV storage and scratch buffers. */
int hga_truncate(hga_session * s, int layer, int new_length);

/* Attend n_q queries at start_pos. q layout [n_q_heads, n_q, head_dim].
 * out is always f32, same layout. Causal: query i only sees keys <= start_pos+i. */
void hga_attend(hga_session * s, int layer, int start_pos, int n_q,
                const void * q, hga_dtype q_dtype, float * out, hga_stats * stats);

/* Convenience: append + attend in one call (typical decode / prefill block). */
void hga_forward(hga_session * s, int layer, int start_pos, int n_q,
                 const void * q, const void * k_rope, const void * k_raw, const void * v,
                 hga_dtype dtype, float * out, hga_stats * stats);

/* Fixed physical-batch entry point.  Input and output use `physical_count`
 * strides, but only the contiguous valid prefix is appended or attended.
 * Invalid output lanes are zeroed and cannot advance cache state. */
int hga_forward_fixed(hga_session * s, int layer, int start_pos,
                      const hga_fixed_batch * batch,
                      const void * q, const void * k_rope,
                      const void * k_raw, const void * v,
                      hga_dtype dtype, float * out, hga_stats * stats);

/* Output layout for the strided path. */
enum hga_out_layout {
    HGA_OUT_HEAD_MAJOR  = 0, /* [n_q_heads, n_q, head_dim]  (public C API) */
    HGA_OUT_TOKEN_MAJOR = 1, /* [n_q, n_q_heads * head_dim] (ggml attn out) */
};

/* F32 Q/K/V with explicit strides. Element (h, t, d) lives at
 *   ptr[h * head_stride + t * tok_stride + d]
 * with `head_dim` contiguous. Quantizes straight into INT8 KV — no packed
 * F32 workspace. `k_raw` may be NULL (then k_rope is used). */
void hga_forward_strided(hga_session * s, int layer, int start_pos, int n_q,
                         const float * q,      int q_head_stride,  int q_tok_stride,
                         const float * k_rope, int k_head_stride,  int k_tok_stride,
                         const float * k_raw,  int kr_head_stride, int kr_tok_stride,
                         const float * v,      int v_head_stride,  int v_tok_stride,
                         float * out, int out_layout, hga_stats * stats);

/* Hybrid prefill support. Routing and persistent cache ownership remain on the
 * CPU, while llama.cpp consumes a compact INT8 staging image with CUDA
 * attention. Each 64-token query chunk is independently routed, but the
 * exact-token historical KV is united once for the complete physical ubatch.
 * The compact INT8 image layout is:
 *
 *   K    [n_kv_heads, history_capacity, head_dim]
 *   V    [n_kv_heads, history_capacity, head_dim]
 *   K/V scales [n_kv_heads, history_capacity]
 *
 * Only positions older than `start_pos` are copied from the persistent INT8
 * cache. All of them are visible to every query. The current physical ubatch
 * remains in llama.cpp's CUDA K/V tensors and uses one causal triangular mask
 * built on CUDA. The historical staging width is its worst-case group-union
 * capacity and the direct width is the configured physical ubatch; therefore
 * K/V shapes do not depend on prefix length or a short final ubatch. Extra
 * historical and direct columns are masked. Unused INT8 payload is untouched;
 * zero scales neutralize it.
 * Append/routing/summary closure still happens independently every 64 tokens.
 */
/* Worst-case capacity for a reusable n_q-wide graph over this session's full
 * configured context. It is independent of the current layer/cache length. */
int hga_gpu_prefill_capacity(const hga_session * s, int layer, int n_q);
int hga_gpu_prefill_current_capacity(const hga_session * s, int layer,
                                     int start_pos, int n_q);
/* Historical-only portion of the current bound. The current physical ubatch
 * can be consumed directly from its CUDA K/V tensors. */
int hga_gpu_prefill_current_history_capacity(const hga_session * s, int layer,
                                             int start_pos, int n_q);
/* Historical staging width for one routed segment. `total_capacity` is the
 * fixed graph key width; the growing current-ubatch prefix occupies the rest. */
int hga_gpu_prefill_segment_history_capacity(
                         const hga_session * s, int layer, int ubatch_start,
                         int seg_start, int seg_n, int total_capacity);
/* Stable historical width used when all independently routed 64-token chunks
 * of one physical ubatch share a single KV image. It is independent of the
 * current start position so llama.cpp can reuse the graph. */
int hga_gpu_prefill_ubatch_history_capacity(
                         const hga_session * s, int layer, int ubatch_start,
                         int n_q, int total_capacity);

/* Compact INT8 prefill staging image. K and V are copied verbatim from the
 * persistent INT8 cache in [kv_head, key, dim] order. Per-vector scales stay
 * F32 exactly as cached. Offsets are relative to the one physical-ubatch
 * image. Unpopulated K/V bytes are deliberately left
 * untouched; their zero scale makes their GPU-dequantized value finite zero. */
typedef struct hga_gpu_prefill_i8_layout {
    size_t k_offset;
    size_t v_offset;
    size_t k_scale_offset;
    size_t v_scale_offset;
    size_t visibility_offset;        /* deprecated: equals n_bytes */
    size_t direct_visibility_offset; /* deprecated: equals n_bytes */
    size_t n_bytes;
} hga_gpu_prefill_i8_layout;

int hga_gpu_prefill_i8_image_layout(const hga_session * s,
                                    int history_capacity, int graph_n_q,
                                    hga_gpu_prefill_i8_layout * out);
int hga_prepare_gpu_prefill_i8_strided(
                         hga_session * s, int layer, int start_pos, int n_q,
                         const float * q,      int q_head_stride,  int q_tok_stride,
                         const float * k_rope, int k_head_stride,  int k_tok_stride,
                         const float * k_raw,  int kr_head_stride, int kr_tok_stride,
                         const float * v,      int v_head_stride,  int v_tok_stride,
                         void * image, size_t image_bytes, int history_capacity,
                         hga_stats * stats);

/* United-ubatch F16 A/B path: identical route/union selection to the INT8
 * production path, but stores and stages K/V directly as F16. */
int hga_prepare_gpu_prefill_f16_ubatch_strided(
                         hga_session * s, int layer, int start_pos, int n_q,
                         const float * q,      int q_head_stride,  int q_tok_stride,
                         const float * k_rope, int k_head_stride,  int k_tok_stride,
                         const float * k_raw,  int kr_head_stride, int kr_tok_stride,
                         const float * v,      int v_head_stride,  int v_tok_stride,
                         uint16_t * image, size_t image_elems, int history_capacity,
                         hga_stats * stats);

/* Legacy all-F16 staging ABI, retained for HGA_PREC_F16 and unit coverage. */
int hga_prepare_gpu_prefill_f16_strided(
                         hga_session * s, int layer, int start_pos, int n_q,
                         const float * q,      int q_head_stride,  int q_tok_stride,
                         const float * k_rope, int k_head_stride,  int k_tok_stride,
                         const float * k_raw,  int kr_head_stride, int kr_tok_stride,
                         const float * v,      int v_head_stride,  int v_tok_stride,
                         uint16_t * image, size_t image_elems, int total_capacity,
                         hga_stats * stats);

/* Compute top-k chunk count used at a given closed-chunk count (identical for both levels). */
int hga_topk_chunks(const hga_config * cfg, int n_closed);
int hga_topk_groups(const hga_config * cfg, int n_closed, int topk_chunks);
/* Query-aware budgets. A full (keep_last+1)-chunk prefill block absorbs the
 * local window, so only sink chunks are removed from the fraction base. Short
 * query blocks progressively remove the unused part of the local window.
 * Routing has quality floors of three middle chunks and six opened groups,
 * capped by the middle chunks/groups that actually exist. */
int hga_routing_base_chunks(const hga_config * cfg, int n_closed, int n_q);
int hga_topk_chunks_for_query(const hga_config * cfg, int n_closed, int n_q);
int hga_topk_groups_for_query(const hga_config * cfg, int n_closed,
                              int topk_chunks, int n_q);

/* Decode L2 plan: last attended key list (windows + routed + active). */
void hga_last_keys(const hga_session * s, const int ** keys, int * n_keys);
int  hga_last_keys_layer(const hga_session * s);

/* Always-on sink + local + active-chunk keys for `layer` (no Q needed). */
int  hga_window_keys(const hga_session * s, int layer, int q_hi, int * keys, int cap);

/* Build a cache-warming key list for one layer. If all used attention K/V
 * fits in budget_bytes, returns the complete prefix. Otherwise keeps the
 * configured sink chunks and fills the remaining budget with newest keys.
 * Raw/router K is deliberately excluded from the byte estimate and touch. */
int hga_prefetch_keys(const hga_session * s, int layer, size_t budget_bytes,
                      int * keys, int cap);

/* Software-load this thread's HGA K-tile of INT8/F16 KV (decode L2 prefetch).
 * Thread mapping matches the 2D flash kernel (`schedule(static)` over kh,q-tile,k-tile). */
void hga_touch_kv_tile(const hga_session * s, int layer, const int * keys, int n_keys,
                       int tid, int n_threads);

/* Software-load the used chunk/group routing summaries for one layer. */
void hga_touch_summary_tile(const hga_session * s, int layer,
                            int tid, int n_threads);

/* Optional L2 GEMV/prefetch plan attached by llama.cpp glue. */
void  hga_session_set_l2(hga_session * s, void * plan);
void * hga_session_l2(const hga_session * s);

#ifdef __cplusplus
} /* extern "C" */
#endif
