#pragma once

#include <cstdint>

/* Two-mode CUDA staging for dense QKV + lm_head.
 * Canonical copies stay mmap'd on the host; staging is a pointer swap + H2D copy.
 * Copied into llama.cpp/src by scripts/apply_hga.py.
 */

struct llama_model;
struct llama_context;
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;

enum hga_swap_phase {
    HGA_SWAP_NONE    = 0,
    /* Weight packing is PREFILL vs DECODE. VERIFY reuses DECODE weights
     * and copies the DECODE activation graph (CPU q/k-norm+RoPE → HGA):
     *   Resident (incl. lm_head, dense QKV, FFN/GDN, MTP): CUDA0, never moved.
     *   Exchange layers only (8 stream pairs, step 4, 0↔32 … 28↔60): mmap CPU,
     *     one CUDA slot per pair, H2D on kick. HGA_STREAM_2=1 is 16↔48/32↔63.
     *   PREFILL: extra FFN not swapped; HGA implicit D2H of large activations.
     *            ubatches shorter than n_ubatch are padded so the ggml graph
     *            is always n_ubatch tokens (one CUDA compute size).
     *   DECODE/VERIFY: always VERIFY (even n=1). 2 stream pairs (16↔48 /
     *            24↔56); the other six prefill-exchange pairs pinned CUDA.
     *            Drop the PREFILL ggml graph (n_ubatch=K+1).
     *   VERIFY: spec K+1. ggml_cont of the strided Q+gate view so the
     *            CPU pin is a dense D2H (n=1 decode does not need that). */
    HGA_SWAP_PREFILL = 1,
    HGA_SWAP_DECODE  = 2,
    HGA_SWAP_VERIFY  = 3,
};

static inline bool hga_gpu_pack(int32_t phase) {
    return phase == (int32_t) HGA_SWAP_PREFILL ||
           phase == (int32_t) HGA_SWAP_DECODE ||
           phase == (int32_t) HGA_SWAP_VERIFY;
}

/* DECODE and VERIFY share the decode-resident weight packing. */
static inline bool hga_decode_pack(int32_t phase) {
    return phase == (int32_t) HGA_SWAP_DECODE || phase == (int32_t) HGA_SWAP_VERIFY;
}

struct llama_cparams;

/* lm_head stays on CUDA. Returns true only if HGA_LMHEAD_CPU=1 (escape hatch
 * for logits-on-CPU; the weight is still not unmapped). */
bool hga_lmhead_on_host(const llama_cparams & cparams);

/* Speculative verify is K+1 tokens and must stay DECODE. Only a true prompt
 * (n_tokens > spec_max, default 16) enters PREFILL. Override with HGA_SPEC_MAX. */
#ifndef HGA_SPEC_MAX_DEFAULT
#define HGA_SPEC_MAX_DEFAULT 16u
#endif

struct hga_weight_swap;

hga_weight_swap * hga_weight_swap_init(const llama_model * model, ggml_backend_t gpu);
void              hga_weight_swap_free(hga_weight_swap * sw);

/* Retarget tensors. Returns false on alloc failure (exchange restored to host).
 * stage_output is ignored: lm_head is CUDA-resident and is not unmapped.
 *
 * After init and each packing switch, a placement census walks every weight:
 *   hga-pin: census after PREFILL  GPU=...  CUDA_Host=...  PUSHED=N
 *   hga-pin: PUSHED  blk.N.attn_output.weight  resident  CUDA0 -> CUDA_Host
 * PUSHED = a tensor that should stay on CUDA0 is on CUDA_Host/CPU (silent
 * fallback / eviction). Exchange-slot H2D is classified separately.
 *   HGA_PIN_CHECK=0    disable (default on)
 *   HGA_PIN_ABORT=1    abort() on the first PUSHED expected-GPU tensor
 *   HGA_PIN_VERBOSE=1  list every MOVE, including staged QKV / stream slots
 *   HGA_PREFILL_PIN_ALL=1 force every named Qwen PREFILL graph node onto CUDA
 *   HGA_CUDA_ALLOC_LEDGER=1 keep/dump the last 20 physical CUDA allocations
 *                         with allocator, request, and free VRAM before/after
 *                         (diagnostic; exchange slot contents still swap) */
bool              hga_weight_swap_set_phase(hga_weight_swap * sw, hga_swap_phase phase,
                                            bool stage_output = true);
hga_swap_phase    hga_weight_swap_phase(const hga_weight_swap * sw);

struct hga_split_ffn;
hga_split_ffn * hga_weight_swap_split(hga_weight_swap * sw);
bool            hga_weight_swap_split_layer(hga_weight_swap * sw, int layer_id);
