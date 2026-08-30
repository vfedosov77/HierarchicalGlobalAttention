#include "hga-split-ffn.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
extern "C" bool ggml_cuda_hga_event_record(ggml_backend_t backend, void * event) __attribute__((weak));
extern "C" bool ggml_cuda_hga_event_wait(ggml_backend_t backend, void * event) __attribute__((weak));
#else
extern "C" bool ggml_cuda_hga_event_record(ggml_backend_t backend, void * event);
extern "C" bool ggml_cuda_hga_event_wait(ggml_backend_t backend, void * event);
#endif

using hga_cuda_event_fn = bool (*)(ggml_backend_t, void *);

static hga_cuda_event_fn g_split_event_record = nullptr;
static hga_cuda_event_fn g_split_event_wait   = nullptr;

static void * hga_split_dlsym(const char * name) {
    void * fn = dlsym(RTLD_DEFAULT, name);
    if (fn) {
        return fn;
    }
#ifdef RTLD_NOLOAD
    void * cuda_so = dlopen("libggml-cuda.so", RTLD_NOW | RTLD_NOLOAD);
    if (cuda_so) {
        fn = dlsym(cuda_so, name);
        dlclose(cuda_so);
    }
#endif
    return fn;
}

static bool hga_split_event_bridge_init() {
    static bool initialized = false;
    if (initialized) {
        return g_split_event_record && g_split_event_wait;
    }
    initialized = true;
    g_split_event_record = ggml_cuda_hga_event_record;
    g_split_event_wait   = ggml_cuda_hga_event_wait;
    if (!g_split_event_record) {
        g_split_event_record = reinterpret_cast<hga_cuda_event_fn>(
                hga_split_dlsym("ggml_cuda_hga_event_record"));
    }
    if (!g_split_event_wait) {
        g_split_event_wait = reinterpret_cast<hga_cuda_event_fn>(
                hga_split_dlsym("ggml_cuda_hga_event_wait"));
    }
    return g_split_event_record && g_split_event_wait;
}

enum { HGA_SPLIT_H2D = 1, HGA_SPLIT_STREAM_NONBLOCKING = 1, HGA_SPLIT_EVENT_DISABLE_TIMING = 2 };

static struct {
    bool inited = false;
    bool ok     = false;
    void * lib  = nullptr;
    int (*malloc_host)(void **, size_t) = nullptr;
    int (*free_host)(void *) = nullptr;
    int (*stream_create)(void **, unsigned) = nullptr;
    int (*stream_destroy)(void *) = nullptr;
    int (*stream_sync)(void *) = nullptr;
    int (*stream_wait_event)(void *, void *, unsigned) = nullptr;
    int (*memcpy_async)(void *, const void *, size_t, int, void *) = nullptr;
    int (*event_create)(void **, unsigned) = nullptr;
    int (*event_destroy)(void *) = nullptr;
    int (*event_record)(void *, void *) = nullptr;
    int (*event_query)(void *) = nullptr;
    const char * (*err_str)(int) = nullptr;
} hga_scu;

static bool hga_scu_init() {
    if (hga_scu.inited) {
        return hga_scu.ok;
    }
    hga_scu.inited = true;
    hga_scu.lib = dlopen("libcudart.so.12", RTLD_NOW | RTLD_LOCAL);
    if (!hga_scu.lib) {
        hga_scu.lib = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!hga_scu.lib) {
        return false;
    }
#define HGA_SCU_SYM(field, name) do { \
        *(void **) &hga_scu.field = dlsym(hga_scu.lib, name); \
        if (!hga_scu.field) { return false; } \
    } while (0)
    HGA_SCU_SYM(malloc_host,       "cudaMallocHost");
    HGA_SCU_SYM(free_host,         "cudaFreeHost");
    HGA_SCU_SYM(stream_create,     "cudaStreamCreateWithFlags");
    HGA_SCU_SYM(stream_destroy,    "cudaStreamDestroy");
    HGA_SCU_SYM(stream_sync,       "cudaStreamSynchronize");
    HGA_SCU_SYM(stream_wait_event, "cudaStreamWaitEvent");
    HGA_SCU_SYM(memcpy_async,      "cudaMemcpyAsync");
    HGA_SCU_SYM(event_create,      "cudaEventCreateWithFlags");
    HGA_SCU_SYM(event_destroy,     "cudaEventDestroy");
    HGA_SCU_SYM(event_record,      "cudaEventRecord");
    HGA_SCU_SYM(event_query,       "cudaEventQuery");
    HGA_SCU_SYM(err_str,           "cudaGetErrorString");
#undef HGA_SCU_SYM
    hga_scu.ok = true;
    return true;
}

static void hga_split_log(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("hga-split: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

static int hga_split_env_int(const char * name, int def, int lo, int hi) {
    const char * e = std::getenv(name);
    if (!e || !e[0]) {
        return def;
    }
    char * end = nullptr;
    const long v = std::strtol(e, &end, 10);
    if (end == e || (end && *end) || v < lo || v > hi) {
        hga_split_log("invalid %s=%s (expected %d..%d); using %d", name, e, lo, hi, def);
        return def;
    }
    return (int) v;
}

bool hga_split_ffn_env_enabled() {
    const char * e = std::getenv("HGA_SPLIT_FFN");
    return e && e[0] && e[0] != '0';
}

int hga_split_ffn_env_tile_channels() {
    return hga_split_env_int("HGA_SPLIT_FFN_TILE_CHANNELS", 1024, 32, 17408);
}

int hga_split_ffn_env_safety_mib() {
    return hga_split_env_int("HGA_SPLIT_FFN_SAFETY_MIB", 8, 0, 256);
}

int hga_split_ffn_env_min_slots() {
    return hga_split_env_int("HGA_SPLIT_FFN_MIN_SLOTS", 2, 2, HGA_SPLIT_FFN_MAX_TILES);
}

int hga_split_ffn_env_copy_streams() {
    return hga_split_env_int("HGA_SPLIT_FFN_COPY_STREAMS", 1, 1, 4);
}

int hga_split_ffn_plan_slots(size_t budget, size_t core, size_t tile,
                             int n_tiles, int min_slots) {
    if (tile == 0 || budget <= core) {
        return 0;
    }
    const size_t avail = budget - core;
    const int n = (int) std::min((size_t) n_tiles, avail / tile);
    return n >= min_slots ? n : 0;
}

int hga_split_ffn_edf_cmp(int prio_a, int64_t deadline_a, int tile_a,
                          int prio_b, int64_t deadline_b, int tile_b) {
    if (prio_a != prio_b) {
        return prio_a - prio_b;
    }
    if (deadline_a != deadline_b) {
        return deadline_a < deadline_b ? -1 : 1;
    }
    if (tile_a != tile_b) {
        return tile_a - tile_b;
    }
    return 0;
}

bool hga_split_ffn_can_slice(int64_t blck, int64_t n_embd, int64_t n_ff, int tile_ch) {
    if (blck <= 0 || tile_ch <= 0 || n_embd <= 0 || n_ff <= 0) {
        return false;
    }
    if (n_ff % tile_ch != 0) {
        return false;
    }
    if (n_embd % blck != 0 || n_ff % blck != 0 || tile_ch % blck != 0) {
        return false;
    }
    return true;
}

size_t hga_split_ffn_row_bytes(size_t type_size, int64_t blck, int64_t ne) {
    if (blck <= 0 || ne < 0) {
        return 0;
    }
    return (size_t) ((ne / blck) * (int64_t) type_size);
}

void hga_split_ffn_pack_down(const uint8_t * src, uint8_t * dst,
                             size_t src_row, size_t dst_row, size_t col_off,
                             int64_t n_rows) {
    for (int64_t r = 0; r < n_rows; ++r) {
        std::memcpy(dst + (size_t) r * dst_row, src + (size_t) r * src_row + col_off, dst_row);
    }
}

/* Cyclic VERIFY consumers: 16, 24, 48, 56. */
static const int k_split_targets[4] = {16, 24, 48, 56};

int hga_split_ffn_cyclic_target(int layer, int which) {
    int best[4];
    int n = 0;
    for (int k = 0; k < 4; ++k) {
        const int t = k_split_targets[k];
        const int dist = (t - layer + 64) % 64;
        if (dist == 0) {
            continue; /* currently at this consumer */
        }
        best[n++] = t;
        for (int i = n - 1; i > 0; --i) {
            const int da = (best[i] - layer + 64) % 64;
            const int db = (best[i - 1] - layer + 64) % 64;
            if (da < db) {
                std::swap(best[i], best[i - 1]);
            }
        }
    }
    if (which < 0 || which >= n) {
        return -1;
    }
    return best[which];
}

struct hga_split_slot {
    int occ_layer = -1;
    int occ_tile  = -1;
    int inflight_layer = -1;
    int inflight_tile  = -1;
    void * ev_ready = nullptr;
    void * ev_free  = nullptr;
    size_t off = 0;
};

struct hga_split_layer {
    int layer_id = -1;
    int pair_i   = -1;
    ggml_tensor * up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    void * host_up = nullptr;
    void * host_gate = nullptr;
    void * host_down = nullptr;
    enum ggml_type type_up   = GGML_TYPE_F32;
    enum ggml_type type_gate = GGML_TYPE_F32;
    enum ggml_type type_down = GGML_TYPE_F32;
    int64_t n_embd = 0;
    int64_t n_ff   = 0;
    size_t up_off[HGA_SPLIT_FFN_MAX_TILES] = {};
    size_t gate_off[HGA_SPLIT_FFN_MAX_TILES] = {};
    size_t down_off[HGA_SPLIT_FFN_MAX_TILES] = {};
    size_t packed[HGA_SPLIT_FFN_MAX_TILES] = {};
    void * pin_base = nullptr;
    void * pin[HGA_SPLIT_FFN_MAX_TILES] = {};
    ggml_tensor * t_up[HGA_SPLIT_FFN_MAX_TILES] = {};
    ggml_tensor * t_gate[HGA_SPLIT_FFN_MAX_TILES] = {};
    ggml_tensor * t_down[HGA_SPLIT_FFN_MAX_TILES] = {};
};

struct hga_split_pair {
    char tag[16] = {};
    int layer_a = -1;
    int layer_b = -1;
    ggml_backend_buffer_t buf = nullptr;
    void * copy_stream = nullptr;
    size_t cap = 0;
    size_t core_a = 0;
    size_t core_b = 0;
    size_t tile   = 0;
    int n_slots = 0;
    hga_split_slot slots[HGA_SPLIT_FFN_MAX_TILES];
};

struct hga_split_job {
    bool used = false;
    int prio = 3;
    int64_t deadline = 0;
    int pair = 0;
    int layer = -1;
    int tile = 0;
    int slot = 0;
};

struct hga_split_ffn {
    ggml_backend_t gpu = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    ggml_context * ctx = nullptr;
    bool timing = false;
    bool async_events = false;
    bool active = false;
    int n_pairs = 0;
    int n_tiles = 0;
    int tile_ch = 0;
    hga_split_pair pairs[HGA_SPLIT_FFN_MAX_PAIRS];
    hga_split_layer layers[4];
    int n_layers = 0;
    hga_split_job jobs[64];
    int n_jobs = 0;
    int64_t pass = 0;
    int current_layer = 0;
    bool final_tile_pending[64] = {};
    uint64_t h2d_bytes = 0;
    double copy_ms = 0;
    double wait_ms = 0;
    int deadline_misses = 0;
    int first_miss_layer = -1;
    int miss_layer[64] = {};
    uint64_t wait_us_layer[64] = {};
    size_t align = 256;
};

static double hga_split_mib(size_t n) {
    return n / (1024.0 * 1024.0);
}

static hga_split_layer * hga_split_find_layer(hga_split_ffn * plan, int layer_id) {
    if (!plan) {
        return nullptr;
    }
    for (int i = 0; i < plan->n_layers; ++i) {
        if (plan->layers[i].layer_id == layer_id) {
            return &plan->layers[i];
        }
    }
    return nullptr;
}

static const hga_split_layer * hga_split_find_layer_c(const hga_split_ffn * plan, int layer_id) {
    return hga_split_find_layer(const_cast<hga_split_ffn *>(plan), layer_id);
}

static int64_t hga_split_deadline(int layer, int64_t pass) {
    return pass * 64 + layer;
}

static int64_t hga_split_pass_for(const hga_split_ffn * plan, int from_layer, int target) {
    const int dist = (target - from_layer + 64) % 64;
    /* Crossing from 56 toward 16 is the next pass. */
    if (from_layer > target && dist > 0) {
        return plan->pass + 1;
    }
    if (from_layer >= 57 && target <= 24) {
        return plan->pass + 1;
    }
    return plan->pass;
}

static size_t hga_split_layout_core(hga_split_ffn * plan, ggml_tensor ** ts, int n, size_t start,
                                    std::vector<size_t> & offs) {
    size_t off = GGML_PAD(start, plan->align);
    offs.resize((size_t) n);
    for (int i = 0; i < n; ++i) {
        off = GGML_PAD(off, plan->align);
        offs[(size_t) i] = off;
        off += GGML_PAD(ggml_backend_buft_get_alloc_size(plan->buft, ts[i]), plan->align);
    }
    return off;
}

static bool hga_split_validate_tensor(ggml_tensor * t, int tile_ch, const char * which, int layer) {
    if (!t) {
        hga_split_log("fallback pair=? layer=%d reason=%s tensor missing", layer, which);
        return false;
    }
    if (ggml_n_dims(t) < 2) {
        hga_split_log("fallback pair=? layer=%d reason=%s not rank-2", layer, which);
        return false;
    }
    const int64_t blck = ggml_blck_size(t->type);
    const int64_t n0 = t->ne[0];
    const int64_t n1 = t->ne[1];
    const int64_t n_embd = std::strcmp(which, "down") == 0 ? n1 : n0;
    const int64_t n_ff = std::strcmp(which, "down") == 0 ? n0 : n1;
    const bool ok = blck > 0 && n_embd % blck == 0 && n_ff % blck == 0 &&
            tile_ch % blck == 0 && n_ff % tile_ch == 0;
    if (!ok) {
        hga_split_log("fallback pair=? layer=%d reason=%s tile not block-aligned type=%s blck=%ld ne0=%ld ne1=%ld tile=%d",
                layer, which, ggml_type_name(t->type), (long) blck, (long) n0, (long) n1, tile_ch);
        return false;
    }
    return true;
}

static bool hga_split_pack_layer_tiles(hga_split_ffn * plan, hga_split_layer & L) {
    const int n_tiles = plan->n_tiles;
    const int tile_ch = plan->tile_ch;
    const size_t up_row   = ggml_row_size(L.type_up,   L.n_embd);
    const size_t gate_row = ggml_row_size(L.type_gate, L.n_embd);
    const size_t src_down_row = ggml_row_size(L.type_down, L.n_ff);
    size_t pin_bytes = 0;

    for (int tile = 0; tile < n_tiles; ++tile) {
        const size_t dst_down_row = ggml_row_size(L.type_down, tile_ch);
        L.up_off[tile] = 0;
        size_t off = GGML_PAD(up_row * (size_t) tile_ch, plan->align);
        L.gate_off[tile] = off;
        off += GGML_PAD(gate_row * (size_t) tile_ch, plan->align);
        L.down_off[tile] = off;
        off += GGML_PAD(dst_down_row * (size_t) L.n_embd, plan->align);
        L.packed[tile] = off;
        pin_bytes = GGML_PAD(pin_bytes, plan->align) + off;
    }

    if (hga_scu.malloc_host(&L.pin_base, pin_bytes) != 0) {
        hga_split_log("fallback pair=? layer=%d reason=cudaMallocHost %.2f MiB failed",
                L.layer_id, hga_split_mib(pin_bytes));
        return false;
    }

    size_t pin_off = 0;
    for (int tile = 0; tile < n_tiles; ++tile) {
        const size_t dst_down_row = ggml_row_size(L.type_down, tile_ch);
        pin_off = GGML_PAD(pin_off, plan->align);
        L.pin[tile] = (uint8_t *) L.pin_base + pin_off;
        pin_off += L.packed[tile];
        uint8_t * dst = (uint8_t *) L.pin[tile];
        std::memcpy(dst + L.up_off[tile],
                (uint8_t *) L.host_up + (size_t) tile * (size_t) tile_ch * up_row,
                (size_t) tile_ch * up_row);
        std::memcpy(dst + L.gate_off[tile],
                (uint8_t *) L.host_gate + (size_t) tile * (size_t) tile_ch * gate_row,
                (size_t) tile_ch * gate_row);
        const size_t col_off = ggml_row_size(L.type_down, (int64_t) tile * tile_ch);
        hga_split_ffn_pack_down((const uint8_t *) L.host_down, dst + L.down_off[tile],
                src_down_row, dst_down_row, col_off, L.n_embd);
    }
    return true;
}

static ggml_tensor * hga_split_new_weight(hga_split_ffn * plan, enum ggml_type type,
                                          int64_t ne0, int64_t ne1, ggml_backend_buffer_t buf,
                                          void * data, const char * name) {
    ggml_tensor * t = ggml_new_tensor_2d(plan->ctx, type, ne0, ne1);
    if (!t) {
        return nullptr;
    }
    t->data = data;
    t->buffer = buf;
    t->view_src = nullptr;
    t->view_offs = 0;
    ggml_set_name(t, name);
    ggml_backend_buffer_init_tensor(buf, t);
    return t;
}

static bool hga_split_bind_layer_cores(hga_split_ffn * plan, hga_split_pair & P,
                                       const hga_split_ffn_layer_src & src, size_t start,
                                       size_t * end_out) {
    std::vector<size_t> offs;
    const size_t end = hga_split_layout_core(plan, const_cast<ggml_tensor **>(src.core),
            src.n_core, start, offs);
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(P.buf);
    for (int i = 0; i < src.n_core; ++i) {
        ggml_tensor * t = src.core[i];
        t->buffer = P.buf;
        t->data = base + offs[(size_t) i];
        t->view_src = nullptr;
        t->view_offs = 0;
        ggml_backend_buffer_init_tensor(P.buf, t);
        const size_t n = ggml_nbytes(t);
        const int rc = hga_scu.memcpy_async(t->data, src.core_host[i], n, HGA_SPLIT_H2D,
                P.copy_stream);
        if (rc != 0) {
            hga_split_log("fallback pair=%s layer=%d reason=core H2D failed: %s",
                    P.tag, src.layer_id, hga_scu.err_str ? hga_scu.err_str(rc) : "?");
            return false;
        }
        plan->h2d_bytes += n;
    }
    *end_out = end;
    return true;
}

static void hga_split_job_push(hga_split_ffn * plan, int prio, int64_t deadline,
                               int pair, int layer, int tile, int slot) {
    /* One pending copy per slot: keep the higher-priority job. */
    for (int i = 0; i < 64; ++i) {
        auto & j = plan->jobs[i];
        if (j.used && j.pair == pair && j.slot == slot) {
            if (hga_split_ffn_edf_cmp(prio, deadline, tile, j.prio, j.deadline, j.tile) < 0) {
                j.prio = prio;
                j.deadline = deadline;
                j.layer = layer;
                j.tile = tile;
            }
            return;
        }
    }
    int free_i = -1;
    for (int i = 0; i < 64; ++i) {
        if (!plan->jobs[i].used) {
            free_i = i;
            break;
        }
    }
    if (free_i < 0) {
        return;
    }
    auto & j = plan->jobs[free_i];
    j.used = true;
    j.prio = prio;
    j.deadline = deadline;
    j.pair = pair;
    j.layer = layer;
    j.tile = tile;
    j.slot = slot;
}

static int hga_split_best_job(hga_split_ffn * plan) {
    int best = -1;
    for (int i = 0; i < 64; ++i) {
        auto & j = plan->jobs[i];
        if (!j.used) {
            continue;
        }
        auto & sl = plan->pairs[j.pair].slots[j.slot];
        if (sl.inflight_layer >= 0) {
            continue;
        }
        if (best < 0 ||
                hga_split_ffn_edf_cmp(j.prio, j.deadline, j.tile,
                        plan->jobs[best].prio, plan->jobs[best].deadline,
                        plan->jobs[best].tile) < 0) {
            best = i;
        }
    }
    return best;
}

static bool hga_split_submit_copy(hga_split_ffn * plan, int pair, int layer, int tile, int slot) {
    auto & P = plan->pairs[pair];
    auto & sl = P.slots[slot];
    hga_split_layer * L = hga_split_find_layer(plan, layer);
    if (!L || !L->pin[tile] || !P.buf) {
        return false;
    }
    uint8_t * dst = (uint8_t *) ggml_backend_buffer_get_base(P.buf) + sl.off;
    if (sl.ev_free && hga_scu.stream_wait_event) {
        hga_scu.stream_wait_event(P.copy_stream, sl.ev_free, 0);
    }
    const int64_t t0 = plan->timing ? ggml_time_us() : 0;
    const int rc = hga_scu.memcpy_async(dst, L->pin[tile], L->packed[tile], HGA_SPLIT_H2D,
            P.copy_stream);
    if (rc != 0) {
        hga_split_log("H2D failed pair=%s layer=%d tile=%d: %s",
                P.tag, layer, tile, hga_scu.err_str ? hga_scu.err_str(rc) : "?");
        return false;
    }
    if (sl.ev_ready && hga_scu.event_record) {
        hga_scu.event_record(sl.ev_ready, P.copy_stream);
    }
    sl.inflight_layer = layer;
    sl.inflight_tile  = tile;
    plan->h2d_bytes += L->packed[tile];
    if (plan->timing) {
        plan->copy_ms += (ggml_time_us() - t0) / 1000.0;
    }
    return true;
}

static void hga_split_pump(hga_split_ffn * plan) {
    if (!plan || !plan->active) {
        return;
    }
    for (;;) {
        const int ji = hga_split_best_job(plan);
        if (ji < 0) {
            return;
        }
        auto j = plan->jobs[ji];
        plan->jobs[ji].used = false;
        if (!hga_split_submit_copy(plan, j.pair, j.layer, j.tile, j.slot)) {
            return;
        }
    }
}

static void hga_split_enqueue_prefix(hga_split_ffn * plan, int pair, int layer, int prio) {
    auto & P = plan->pairs[pair];
    hga_split_layer * L = hga_split_find_layer(plan, layer);
    if (!L) {
        return;
    }
    const int64_t dl = hga_split_deadline(layer, hga_split_pass_for(plan, plan->current_layer, layer));
    for (int s = 0; s < P.n_slots; ++s) {
        auto & sl = P.slots[s];
        const bool have = (sl.occ_layer == layer && sl.occ_tile == s) ||
                (sl.inflight_layer == layer && sl.inflight_tile == s);
        if (have) {
            continue;
        }
        hga_split_job_push(plan, prio, dl, pair, layer, s, s);
    }
}

static void hga_split_enqueue_missing(hga_split_ffn * plan, int layer, int tile) {
    hga_split_layer * L = hga_split_find_layer(plan, layer);
    if (!L) {
        return;
    }
    auto & P = plan->pairs[L->pair_i];
    const int slot = tile % P.n_slots;
    auto & sl = P.slots[slot];
    if ((sl.occ_layer == layer && sl.occ_tile == tile) ||
            (sl.inflight_layer == layer && sl.inflight_tile == tile)) {
        return;
    }
    const int64_t dl = hga_split_deadline(layer, plan->pass);
    hga_split_job_push(plan, /*prio*/ 0, dl, L->pair_i, layer, tile, slot);
}

hga_split_ffn * hga_split_ffn_create(
        ggml_backend_t gpu,
        ggml_backend_buffer_type_t buft,
        const hga_split_ffn_pair_src * pairs,
        int n_pairs,
        bool timing,
        bool async_events) {
    auto fallback = [](const char * pair, int layer, const char * reason) -> hga_split_ffn * {
        hga_split_log("fallback pair=%s layer=%d reason=%s",
                pair && pair[0] ? pair : "?", layer, reason);
        return nullptr;
    };

    if (!hga_split_ffn_env_enabled()) {
        return nullptr;
    }
    if (hga_split_ffn_env_copy_streams() != 1) {
        return fallback("?", -1, "HGA_SPLIT_FFN_COPY_STREAMS MVP supports exactly 1");
    }
    if (!gpu || !buft || !pairs || n_pairs <= 0 || n_pairs > HGA_SPLIT_FFN_MAX_PAIRS) {
        return fallback("?", -1, "bad create arguments");
    }
    if (!async_events || !hga_split_event_bridge_init()) {
        return fallback("?", -1, "CUDA event bridge or inline callback installation fails");
    }
    if (!hga_scu_init()) {
        return fallback("?", -1, "libcudart unavailable");
    }

    const int tile_ch = hga_split_ffn_env_tile_channels();
    const int min_slots = hga_split_ffn_env_min_slots();
    const size_t safety = (size_t) hga_split_ffn_env_safety_mib() * 1024 * 1024;

    auto * plan = new hga_split_ffn();
    plan->gpu = gpu;
    plan->buft = buft;
    plan->timing = timing;
    plan->async_events = true;
    plan->tile_ch = tile_ch;
    plan->align = ggml_backend_buft_get_alignment(buft);
    if (plan->align == 0) {
        plan->align = 256;
    }

    const size_t ctx_size = ggml_tensor_overhead() * 512;
    struct ggml_init_params ip = { ctx_size, nullptr, true };
    plan->ctx = ggml_init(ip);
    if (!plan->ctx) {
        hga_split_ffn_free(plan);
        return fallback("?", -1, "ggml context alloc failed");
    }

    plan->n_pairs = n_pairs;
    plan->n_layers = 0;

    for (int pi = 0; pi < n_pairs; ++pi) {
        const auto & in = pairs[pi];
        auto & P = plan->pairs[pi];
        std::snprintf(P.tag, sizeof(P.tag), "%s", in.tag[0] ? in.tag : "?");
        P.buf = in.buf;
        P.cap = in.cap;
        P.layer_a = in.a.layer_id;
        P.layer_b = in.b.layer_id;
        if (hga_scu.stream_create(&P.copy_stream, HGA_SPLIT_STREAM_NONBLOCKING) != 0) {
            hga_split_ffn_free(plan);
            return fallback(P.tag, P.layer_a, "copy stream create failed");
        }
        if (!P.buf || P.cap == 0) {
            hga_split_ffn_free(plan);
            return fallback(P.tag, P.layer_a, "pair buffer missing");
        }
        if (P.cap <= safety) {
            hga_split_ffn_free(plan);
            return fallback(P.tag, P.layer_a, "pair budget smaller than safety margin");
        }

        const hga_split_ffn_layer_src * srcs[2] = {&in.a, &in.b};
        hga_split_layer * Ls[2];
        size_t packed_max = 0;
        size_t core_bytes[2] = {0, 0};

        for (int side = 0; side < 2; ++side) {
            const auto & S = *srcs[side];
            if (S.up_s || S.gate_s || S.down_s) {
                hga_split_ffn_free(plan);
                return fallback(P.tag, S.layer_id, "auxiliary per-weight scale tensors cannot be sliced consistently");
            }
            if (!hga_split_validate_tensor(S.up, tile_ch, "up", S.layer_id) ||
                !hga_split_validate_tensor(S.gate, tile_ch, "gate", S.layer_id) ||
                !hga_split_validate_tensor(S.down, tile_ch, "down", S.layer_id)) {
                hga_split_ffn_free(plan);
                return nullptr;
            }
            if (!S.up->data || !S.gate->data || !S.down->data) {
                hga_split_ffn_free(plan);
                return fallback(P.tag, S.layer_id, "FFN tensor has no host data");
            }
            if (plan->n_layers >= 4) {
                hga_split_ffn_free(plan);
                return fallback(P.tag, S.layer_id, "too many split layers");
            }
            auto & L = plan->layers[plan->n_layers++];
            Ls[side] = &L;
            L.layer_id = S.layer_id;
            L.pair_i = pi;
            L.up = S.up;
            L.gate = S.gate;
            L.down = S.down;
            L.host_up = S.host_up ? S.host_up : S.up->data;
            L.host_gate = S.host_gate ? S.host_gate : S.gate->data;
            L.host_down = S.host_down ? S.host_down : S.down->data;
            L.type_up = S.up->type;
            L.type_gate = S.gate->type;
            L.type_down = S.down->type;
            L.n_embd = S.up->ne[0];
            L.n_ff   = S.up->ne[1];
            if (S.down->ne[0] != L.n_ff || S.down->ne[1] != L.n_embd ||
                    S.gate->ne[0] != L.n_embd || S.gate->ne[1] != L.n_ff) {
                hga_split_ffn_free(plan);
                return fallback(P.tag, S.layer_id, "FFN shapes do not match SwiGLU layout");
            }
            if (plan->n_tiles == 0) {
                plan->n_tiles = (int) (L.n_ff / tile_ch);
                if (plan->n_tiles > HGA_SPLIT_FFN_MAX_TILES) {
                    hga_split_ffn_free(plan);
                    return fallback(P.tag, S.layer_id, "too many tiles");
                }
            } else if (L.n_ff / tile_ch != plan->n_tiles) {
                hga_split_ffn_free(plan);
                return fallback(P.tag, S.layer_id, "tile count mismatch across layers");
            }

            std::vector<size_t> dummy;
            core_bytes[side] = hga_split_layout_core(plan, const_cast<ggml_tensor **>(S.core),
                    S.n_core, plan->align, dummy);
            if (!hga_split_pack_layer_tiles(plan, L)) {
                hga_split_ffn_free(plan);
                return nullptr;
            }
            const size_t layer_packed_max = *std::max_element(
                    L.packed, L.packed + plan->n_tiles);
            packed_max = std::max(packed_max, layer_packed_max);
            hga_split_log("layer=%d types up=%s gate=%s down=%s  n_embd=%ld n_ff=%ld  "
                    "up=%zu gate=%zu down=%zu packed_tile=%.2f MiB core=%.2f MiB",
                    L.layer_id, ggml_type_name(L.type_up), ggml_type_name(L.type_gate),
                    ggml_type_name(L.type_down), (long) L.n_embd, (long) L.n_ff,
                    ggml_nbytes(L.up), ggml_nbytes(L.gate), ggml_nbytes(L.down),
                    hga_split_mib(layer_packed_max), hga_split_mib(core_bytes[side]));
        }

        P.core_a = core_bytes[0];
        P.core_b = core_bytes[1];
        P.tile = packed_max;
        const size_t budget = P.cap - safety;
        const size_t cores = P.core_a + P.core_b;
        P.n_slots = hga_split_ffn_plan_slots(budget, cores, P.tile, plan->n_tiles, min_slots);
        if (P.n_slots < min_slots) {
            char why[192];
            std::snprintf(why, sizeof(why),
                    "both cores plus %d tile slots do not fit budget=%.2f MiB core=%.2f tile=%.2f",
                    min_slots, hga_split_mib(budget), hga_split_mib(cores), hga_split_mib(P.tile));
            hga_split_ffn_free(plan);
            return fallback(P.tag, P.layer_a, why);
        }
        const size_t used = cores + (size_t) P.n_slots * P.tile;
        if (used > P.cap) {
            hga_split_ffn_free(plan);
            return fallback(P.tag, P.layer_a, "split arena exceeds whole-layer pair budget");
        }

        /* Dynamic prefetch bank: use all N slots as reusable A/B tiles. */
        size_t core_end_a = 0, core_end_b = 0;
        if (!hga_split_bind_layer_cores(plan, P, in.a, plan->align, &core_end_a)) {
            hga_split_ffn_free(plan);
            return nullptr;
        }
        if (!hga_split_bind_layer_cores(plan, P, in.b, core_end_a, &core_end_b)) {
            hga_split_ffn_free(plan);
            return nullptr;
        }
        size_t slot_off = GGML_PAD(core_end_b, plan->align);
        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(P.buf);
        const int ev_flags = timing ? 0 : HGA_SPLIT_EVENT_DISABLE_TIMING;
        for (int s = 0; s < P.n_slots; ++s) {
            auto & sl = P.slots[s];
            sl.off = slot_off;
            slot_off = GGML_PAD(slot_off + P.tile, plan->align);
            if (hga_scu.event_create(&sl.ev_ready, ev_flags) != 0 ||
                    hga_scu.event_create(&sl.ev_free, ev_flags) != 0) {
                hga_split_ffn_free(plan);
                return fallback(P.tag, P.layer_a, "tile event create failed");
            }
            /* First copy may proceed immediately. */
            hga_scu.event_record(sl.ev_free, P.copy_stream);
        }
        /* Per-layer views: UD-Q4_K_M may assign different types to A vs B. */
        for (int side = 0; side < 2; ++side) {
            auto * L = Ls[side];
            for (int tile = 0; tile < plan->n_tiles; ++tile) {
                const int s = tile % P.n_slots;
                auto & sl = P.slots[s];
                char nm[64];
                std::snprintf(nm, sizeof(nm), "hga_tile_up_%d_t%d", L->layer_id, tile);
                L->t_up[tile] = hga_split_new_weight(plan, L->type_up, L->n_embd, tile_ch,
                        P.buf, base + sl.off + L->up_off[tile], nm);
                std::snprintf(nm, sizeof(nm), "hga_tile_gate_%d_t%d", L->layer_id, tile);
                L->t_gate[tile] = hga_split_new_weight(plan, L->type_gate, L->n_embd, tile_ch,
                        P.buf, base + sl.off + L->gate_off[tile], nm);
                std::snprintf(nm, sizeof(nm), "hga_tile_down_%d_t%d", L->layer_id, tile);
                L->t_down[tile] = hga_split_new_weight(plan, L->type_down, tile_ch, L->n_embd,
                        P.buf, base + sl.off + L->down_off[tile], nm);
                if (!L->t_up[tile] || !L->t_gate[tile] || !L->t_down[tile]) {
                    hga_split_ffn_free(plan);
                    return fallback(P.tag, L->layer_id, "tile tensor create failed");
                }
            }
        }
        if (slot_off > P.cap) {
            hga_split_ffn_free(plan);
            return fallback(P.tag, P.layer_a, "physical CUDA allocation exceeds old pair slot budget");
        }

        hga_split_log("pair=%s budget=%.2fMiB coreA=%.2f coreB=%.2f tile=%.2f slots=%d/%d",
                P.tag, hga_split_mib(P.cap), hga_split_mib(P.core_a), hga_split_mib(P.core_b),
                hga_split_mib(P.tile), P.n_slots, plan->n_tiles);
    }

    if (hga_scu.stream_sync) {
        for (int pi = 0; pi < plan->n_pairs; ++pi) {
            hga_scu.stream_sync(plan->pairs[pi].copy_stream);
        }
    }

    /* Log PCIe if nvidia-smi is around. */
    {
        FILE * fp = popen("nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader 2>/dev/null", "r");
        if (fp) {
            char line[256] = {};
            if (fgets(line, sizeof(line), fp)) {
                char * nl = std::strchr(line, '\n');
                if (nl) {
                    *nl = 0;
                }
                hga_split_log("gpu %s", line);
            }
            pclose(fp);
        }
    }

    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(gpu);
    if (dev) {
        ggml_backend_dev_memory(dev, &free_b, &total_b);
    }
    hga_split_log("after split-plan creation free=%.0f MiB used=%.0f MiB",
            hga_split_mib(free_b), hga_split_mib(total_b > free_b ? total_b - free_b : 0));

    plan->active = true;
    plan->h2d_bytes = 0; /* cores are one-shot init, not per-pass */
    return plan;
}

void hga_split_ffn_free(hga_split_ffn * plan) {
    if (!plan) {
        return;
    }
    if (hga_scu.stream_sync) {
        for (int pi = 0; pi < plan->n_pairs; ++pi) {
            if (plan->pairs[pi].copy_stream) {
                hga_scu.stream_sync(plan->pairs[pi].copy_stream);
            }
        }
    }
    for (int i = 0; i < plan->n_layers; ++i) {
        auto & L = plan->layers[i];
        if (L.pin_base && hga_scu.free_host) {
            hga_scu.free_host(L.pin_base);
            L.pin_base = nullptr;
        }
    }
    for (int pi = 0; pi < plan->n_pairs; ++pi) {
        auto & P = plan->pairs[pi];
        for (int s = 0; s < P.n_slots; ++s) {
            auto & sl = P.slots[s];
            if (sl.ev_ready && hga_scu.event_destroy) {
                hga_scu.event_destroy(sl.ev_ready);
            }
            if (sl.ev_free && hga_scu.event_destroy) {
                hga_scu.event_destroy(sl.ev_free);
            }
            sl.ev_ready = sl.ev_free = nullptr;
        }
        if (P.copy_stream && hga_scu.stream_destroy) {
            hga_scu.stream_destroy(P.copy_stream);
            P.copy_stream = nullptr;
        }
    }
    if (plan->ctx) {
        ggml_free(plan->ctx);
        plan->ctx = nullptr;
    }
    delete plan;
}

bool hga_split_ffn_active(const hga_split_ffn * plan) {
    return plan && plan->active;
}

bool hga_split_ffn_layer_active(const hga_split_ffn * plan, int layer_id) {
    return hga_split_find_layer_c(plan, layer_id) != nullptr;
}

int hga_split_ffn_n_tiles(const hga_split_ffn * plan) {
    return plan ? plan->n_tiles : 0;
}

int hga_split_ffn_n_slots(const hga_split_ffn * plan, int layer_id) {
    const hga_split_layer * L = hga_split_find_layer_c(plan, layer_id);
    if (!L) {
        return 0;
    }
    return plan->pairs[L->pair_i].n_slots;
}

ggml_tensor * hga_split_ffn_slot_up(hga_split_ffn * plan, int layer_id, int tile) {
    hga_split_layer * L = hga_split_find_layer(plan, layer_id);
    if (!L) {
        return nullptr;
    }
    return tile >= 0 && tile < plan->n_tiles ? L->t_up[tile] : nullptr;
}

ggml_tensor * hga_split_ffn_slot_gate(hga_split_ffn * plan, int layer_id, int tile) {
    hga_split_layer * L = hga_split_find_layer(plan, layer_id);
    if (!L) {
        return nullptr;
    }
    return tile >= 0 && tile < plan->n_tiles ? L->t_gate[tile] : nullptr;
}

ggml_tensor * hga_split_ffn_slot_down(hga_split_ffn * plan, int layer_id, int tile) {
    hga_split_layer * L = hga_split_find_layer(plan, layer_id);
    if (!L) {
        return nullptr;
    }
    return tile >= 0 && tile < plan->n_tiles ? L->t_down[tile] : nullptr;
}

static bool hga_split_parse_tile_name(const char * name, const char * prefix,
                                      int * layer, int * tile) {
    const size_t n = std::strlen(prefix);
    if (!name || std::strncmp(name, prefix, n) != 0) {
        return false;
    }
    int l = 0, t = 0;
    if (std::sscanf(name + n, "%d-%d", &l, &t) != 2) {
        return false;
    }
    *layer = l;
    *tile = t;
    return true;
}

static int hga_split_name_layer(const char * name) {
    if (!name || !name[0]) {
        return -1;
    }
    const char * d = std::strrchr(name, '-');
    if (!d || !d[1]) {
        return -1;
    }
    char * end = nullptr;
    const long v = std::strtol(d + 1, &end, 10);
    if (end == d + 1 || (end && *end)) {
        return -1;
    }
    return (int) v;
}

static void hga_split_on_begin(hga_split_ffn * plan, int layer, int tile) {
    hga_split_layer * L = hga_split_find_layer(plan, layer);
    if (!L) {
        return;
    }
    auto & P = plan->pairs[L->pair_i];
    const int slot = tile % P.n_slots;
    auto & sl = P.slots[slot];
    plan->current_layer = layer;

    /* Previous tile's down/accumulate are already on the compute stream. */
    if (tile > 0) {
        const int prev = tile - 1;
        const int prev_slot = prev % P.n_slots;
        auto & ps = P.slots[prev_slot];
        if (ps.ev_free && g_split_event_record) {
            g_split_event_record(plan->gpu, ps.ev_free);
        }
        const int repl = prev + P.n_slots;
        if (repl < plan->n_tiles) {
            hga_split_enqueue_missing(plan, layer, repl);
        } else {
            const int other = (layer == P.layer_a) ? P.layer_b : P.layer_a;
            const int64_t dl = hga_split_deadline(other,
                    hga_split_pass_for(plan, layer, other));
            hga_split_job_push(plan, 1, dl, L->pair_i, other, prev_slot, prev_slot);
        }
        hga_split_pump(plan);
    }

    if (!((sl.occ_layer == layer && sl.occ_tile == tile) ||
            (sl.inflight_layer == layer && sl.inflight_tile == tile))) {
        hga_split_enqueue_missing(plan, layer, tile);
        hga_split_pump(plan);
    }

    if (sl.ev_ready) {
        const bool pending = hga_scu.event_query && hga_scu.event_query(sl.ev_ready) != 0;
        if (pending) {
            plan->deadline_misses++;
            plan->miss_layer[layer < 64 ? layer : 0]++;
            if (plan->first_miss_layer < 0) {
                plan->first_miss_layer = layer;
                hga_split_log("first deadline miss layer=%d tile=%d slot=%d", layer, tile, slot);
            }
        }
        const int64_t t0 = plan->timing ? ggml_time_us() : 0;
        if (g_split_event_wait) {
            g_split_event_wait(plan->gpu, sl.ev_ready);
        }
        if (plan->timing && pending) {
            const uint64_t us = (uint64_t) (ggml_time_us() - t0);
            plan->wait_ms += us / 1000.0;
            plan->wait_us_layer[layer < 64 ? layer : 0] += us;
        }
    }
    sl.occ_layer = layer;
    sl.occ_tile = tile;
    if (sl.inflight_layer == layer && sl.inflight_tile == tile) {
        sl.inflight_layer = -1;
        sl.inflight_tile = -1;
    }
    if (tile == plan->n_tiles - 1 && layer >= 0 && layer < 64) {
        plan->final_tile_pending[layer] = true;
    }
}

static void hga_split_finish_layer(hga_split_ffn * plan, int layer) {
    hga_split_layer * L = hga_split_find_layer(plan, layer);
    if (!L || layer < 0 || layer >= 64 || !plan->final_tile_pending[layer]) {
        return;
    }
    plan->final_tile_pending[layer] = false;
    auto & P = plan->pairs[L->pair_i];
    const int tile = plan->n_tiles - 1;
    const int slot = tile % P.n_slots;
    auto & sl = P.slots[slot];
    if (sl.ev_free && g_split_event_record) {
        g_split_event_record(plan->gpu, sl.ev_free);
    }

    const int other = (layer == P.layer_a) ? P.layer_b : P.layer_a;
    const int64_t dl = hga_split_deadline(other,
            hga_split_pass_for(plan, layer, other));
    hga_split_job_push(plan, 1, dl, L->pair_i, other, slot, slot);
    hga_split_pump(plan);

    hga_split_log("layer=%d prefetched=%d streamed_at_layer=%d",
            layer, P.n_slots, plan->n_tiles - P.n_slots);
    if (layer == 56) {
        plan->pass++;
        hga_split_ffn_dump_pass(plan);
    }
}

bool hga_split_ffn_on_cuda_node(hga_split_ffn * plan, ggml_tensor * t) {
    if (!plan || !plan->active || !t || !t->name[0]) {
        return false;
    }
    int layer = -1, tile = -1;
    if (hga_split_parse_tile_name(t->name, "hga_split_ffn_begin-", &layer, &tile)) {
        hga_split_on_begin(plan, layer, tile);
        return true;
    }
    if (std::strncmp(t->name, "norm-", 5) == 0 || std::strncmp(t->name, "attn_norm-", 10) == 0) {
        const int il = hga_split_name_layer(t->name);
        if (il >= 0) {
            hga_split_finish_layer(plan, il - 1);
            plan->current_layer = il;
            const int nearest = hga_split_ffn_cyclic_target(il, 0);
            const int follow  = hga_split_ffn_cyclic_target(il, 1);
            if (nearest >= 0) {
                hga_split_layer * N = hga_split_find_layer(plan, nearest);
                if (N) {
                    hga_split_enqueue_prefix(plan, N->pair_i, nearest, 1);
                }
            }
            if (follow >= 0) {
                hga_split_layer * N = hga_split_find_layer(plan, follow);
                if (N) {
                    hga_split_enqueue_prefix(plan, N->pair_i, follow, 2);
                }
            }
            hga_split_pump(plan);
        }
        return false;
    }
    return false;
}

void hga_split_ffn_begin_ubatch(hga_split_ffn * plan) {
    if (!plan || !plan->active) {
        return;
    }
    plan->current_layer = 0;
    for (int pi = 0; pi < plan->n_pairs; ++pi) {
        /* First consumer of the pair this pass. */
        const int target = plan->pairs[pi].layer_a; /* 16 or 24 */
        hga_split_enqueue_prefix(plan, pi, target, 1);
    }
    hga_split_pump(plan);
}

void hga_split_ffn_dump_pass(hga_split_ffn * plan) {
    if (!plan) {
        return;
    }
    const double gib_s = (plan->copy_ms > 0 && plan->h2d_bytes)
            ? (plan->h2d_bytes / (1024.0 * 1024.0 * 1024.0)) / (plan->copy_ms / 1000.0)
            : 0.0;
    hga_split_log("pass h2d=%.1fMiB copy_ms=%.2f wait_ms=%.2f deadline_misses=%d h2d_gib_s=%.2f",
            hga_split_mib(plan->h2d_bytes), plan->copy_ms, plan->wait_ms,
            plan->deadline_misses, gib_s);
    plan->h2d_bytes = 0;
    plan->copy_ms = 0;
    plan->wait_ms = 0;
    plan->deadline_misses = 0;
}
