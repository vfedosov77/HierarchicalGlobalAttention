#pragma once

#include <cstddef>
#include <cstdint>

/* DECODE/VERIFY-only tiled SwiGLU FFN streaming.
 * Owned by hga_weight_swap for the current model/context. Occupancy, events,
 * and deadlines live on that plan — never in process-global state.
 *
 * Copied into llama.cpp/src by scripts/apply_hga.py.
 */

struct ggml_tensor;
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
struct ggml_backend_buffer_type;
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
struct hga_weight_swap;
struct hga_split_ffn;

#ifndef HGA_SPLIT_FFN_MAX_TILES
#define HGA_SPLIT_FFN_MAX_TILES 17
#endif
#ifndef HGA_SPLIT_FFN_MAX_PAIRS
#define HGA_SPLIT_FFN_MAX_PAIRS 2
#endif
#ifndef HGA_SPLIT_FFN_MAX_CORE
#define HGA_SPLIT_FFN_MAX_CORE 32
#endif

/* Opt-in. HGA_SPLIT_FFN=0 (default) keeps the whole-layer VERIFY path. */
bool hga_split_ffn_env_enabled();
int  hga_split_ffn_env_tile_channels();
int  hga_split_ffn_env_safety_mib();
int  hga_split_ffn_env_min_slots();
int  hga_split_ffn_env_copy_streams();

struct hga_split_ffn_layer_src {
    int layer_id = -1;
    ggml_tensor * up   = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    void * host_up   = nullptr;
    void * host_gate = nullptr;
    void * host_down = nullptr;
    ggml_tensor * up_s   = nullptr;
    ggml_tensor * gate_s = nullptr;
    ggml_tensor * down_s = nullptr;
    ggml_tensor * core[HGA_SPLIT_FFN_MAX_CORE] = {};
    void *        core_host[HGA_SPLIT_FFN_MAX_CORE] = {};
    int n_core = 0;
};

struct hga_split_ffn_pair_src {
    char tag[16] = {};
    ggml_backend_buffer_t buf = nullptr;
    size_t cap = 0;
    hga_split_ffn_layer_src a;
    hga_split_ffn_layer_src b;
};

/* Returns nullptr and logs a one-line fallback reason on any failure. */
hga_split_ffn * hga_split_ffn_create(
        ggml_backend_t gpu,
        ggml_backend_buffer_type_t buft,
        const hga_split_ffn_pair_src * pairs,
        int n_pairs,
        bool timing,
        bool async_events);

void hga_split_ffn_free(hga_split_ffn * plan);

bool hga_split_ffn_active(const hga_split_ffn * plan);
bool hga_split_ffn_layer_active(const hga_split_ffn * plan, int layer_id);
int  hga_split_ffn_n_tiles(const hga_split_ffn * plan);
int  hga_split_ffn_n_slots(const hga_split_ffn * plan, int layer_id);

ggml_tensor * hga_split_ffn_slot_up  (hga_split_ffn * plan, int layer_id, int tile);
ggml_tensor * hga_split_ffn_slot_gate(hga_split_ffn * plan, int layer_id, int tile);
ggml_tensor * hga_split_ffn_slot_down(hga_split_ffn * plan, int layer_id, int tile);

/* Inline CUDA-node hook. Returns true for split-FFN tile begin nodes.
 * `norm-*` nodes release the previous layer's final tile and return false so
 * leftover whole-layer pairs still run. */
bool hga_split_ffn_on_cuda_node(hga_split_ffn * plan, ggml_tensor * t);

void hga_split_ffn_begin_ubatch(hga_split_ffn * plan);
void hga_split_ffn_dump_pass(hga_split_ffn * plan);

/* Swap-owned accessors. */
hga_split_ffn * hga_weight_swap_split(hga_weight_swap * sw);
bool            hga_weight_swap_split_layer(hga_weight_swap * sw, int layer_id);

/* Testable helpers (no CUDA). */
int    hga_split_ffn_plan_slots(size_t budget, size_t core, size_t tile,
                                int n_tiles, int min_slots);
int    hga_split_ffn_edf_cmp(int prio_a, int64_t deadline_a, int tile_a,
                             int prio_b, int64_t deadline_b, int tile_b);
bool   hga_split_ffn_can_slice(int64_t blck, int64_t n_embd, int64_t n_ff,
                               int tile_ch);
size_t hga_split_ffn_row_bytes(size_t type_size, int64_t blck, int64_t ne);
void   hga_split_ffn_pack_down(const uint8_t * src, uint8_t * dst,
                               size_t src_row, size_t dst_row, size_t col_off,
                               int64_t n_rows);
int    hga_split_ffn_cyclic_target(int layer, int which /* 0=nearest, 1=next */);
