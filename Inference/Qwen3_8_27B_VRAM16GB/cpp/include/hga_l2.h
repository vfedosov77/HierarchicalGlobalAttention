#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Decode-side host K/V GEMV + idle L2 plan (Xeon Gold 6148).
 *
 * Q lives on the GPU. Each physical core keeps a Q4_K/Q6_K *slice* of
 * attn_k or attn_v along hidden_dim (one 256-wide superblock × n_out rows,
 * ~147–210 KiB) in private L2. During the GPU FFN/GDN hole after a dense
 * layer, the same cores preload the *next* dense layer's weight slice and
 * that layer's INT8 KV at the previous layer's HGA key list.
 */

struct hga_session;
typedef struct hga_l2_plan hga_l2_plan;

/* Matches ggml_to_float_t: dequant one superblock (`n` = blck_size, usually 256). */
typedef void (*hga_l2_dequant_fn)(const void * block, float * dst, int64_t n);

hga_l2_plan * hga_l2_plan_create(int n_threads, int n_layers, int n_embd, int n_out);
void          hga_l2_plan_free(hga_l2_plan * p);

/* Pin n_threads workers to physical cores (K team on socket 0, V on socket 1). */
int  hga_l2_plan_start(hga_l2_plan * p);
void hga_l2_plan_stop(hga_l2_plan * p);

/* Repack one layer so each core's superblocks are contiguous (avoids the
 * 2880-byte ggml row stride, which would prefetch neighbouring blocks). */
void hga_l2_bind_weights(hga_l2_plan * p, int layer,
                         const void * wk, size_t wk_row_bytes, size_t wk_blk_bytes,
                         int wk_blk_n, hga_l2_dequant_fn wk_dequant,
                         const void * wv, size_t wv_row_bytes, size_t wv_blk_bytes,
                         int wv_blk_n, hga_l2_dequant_fn wv_dequant);

/* Sharded K/V GEMV: x[n_embd] → k_out[n_out], v_out[n_out]. */
void hga_l2_gemv(hga_l2_plan * p, int layer, const float * x, float * k_out, float * v_out);

/* Non-blocking: fill L2 with layer's weight slices + HGA KV tile (prev keys). */
void hga_l2_kick_prefetch(hga_l2_plan * p, int layer, const hga_session * sess,
                          const int * keys, int n_keys);

/* L3 experiment: use a small, socket-spread subset of the persistent workers
 * to touch attention K/V and routing summaries. Unlike decode L2 prefetch
 * this does not touch per-core projection-weight slices. */
void hga_l3_kick_kv_prefetch(hga_l2_plan * p, int layer,
                             const hga_session * sess,
                             const int * keys, int n_keys, int n_workers);

int    hga_l2_n_threads(const hga_l2_plan * p);
int    hga_l2_n_blocks(const hga_l2_plan * p);
size_t hga_l2_slice_bytes(const hga_l2_plan * p, int layer, int tid);

/* Q4_K helpers for tests (ggml-common.h field order: d, dmin, scales, qs). */
#define HGA_QK_K       256
#define HGA_Q4K_BYTES  144

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
} hga_q4k_block;

#if defined(__cplusplus)
static_assert(sizeof(hga_q4k_block) == HGA_Q4K_BYTES, "hga_q4k_block must be 144 bytes");
#endif

void hga_q4k_dequant_block(const void * block, float * dst, int64_t n);
void hga_q4k_make_uniform(hga_q4k_block * b, float d, uint8_t nibble);

#ifdef __cplusplus
}
#endif
