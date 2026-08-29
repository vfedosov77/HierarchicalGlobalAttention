#include "hga-weight-swap.h"
#include "hga-split-ffn.h"

#include "llama-context.h"
#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static void hga_swap_log(const char * fmt, ...);

/* Exported by the small ggml-cuda patch installed by apply_hga.py.  Keep this
 * weak: CPU-only llama.cpp builds must continue to link without ggml-cuda.
 * A weak reference alone is not reliable with --as-needed: libllama can be
 * linked before libggml-cuda and the dynamic loader can leave it unresolved.
 * Fall back to a runtime lookup of the already-loaded CUDA backend. */
#if defined(__GNUC__) || defined(__clang__)
extern "C" void ggml_cuda_hga_vmm_shrink(void) __attribute__((weak));
extern "C" bool ggml_cuda_hga_event_record(ggml_backend_t backend, void * event) __attribute__((weak));
extern "C" bool ggml_cuda_hga_event_wait(ggml_backend_t backend, void * event) __attribute__((weak));
extern "C" bool ggml_cuda_hga_set_node_callback(
        ggml_backend_t backend, void (* callback)(ggml_tensor *, void *), void * user) __attribute__((weak));
#else
extern "C" void ggml_cuda_hga_vmm_shrink(void);
extern "C" bool ggml_cuda_hga_event_record(ggml_backend_t backend, void * event);
extern "C" bool ggml_cuda_hga_event_wait(ggml_backend_t backend, void * event);
extern "C" bool ggml_cuda_hga_set_node_callback(
        ggml_backend_t backend, void (* callback)(ggml_tensor *, void *), void * user);
#endif

using hga_cuda_event_fn = bool (*)(ggml_backend_t, void *);
using hga_cuda_node_callback = void (*)(ggml_tensor *, void *);
using hga_cuda_set_node_callback_fn = bool (*)(ggml_backend_t, hga_cuda_node_callback, void *);
static hga_cuda_event_fn g_hga_cuda_event_record = nullptr;
static hga_cuda_event_fn g_hga_cuda_event_wait = nullptr;
static hga_cuda_set_node_callback_fn g_hga_cuda_set_node_callback = nullptr;

static void * hga_cuda_backend_symbol(const char * name) {
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

static bool hga_cuda_event_bridge_init() {
    static bool initialized = false;
    if (initialized) {
        return g_hga_cuda_event_record && g_hga_cuda_event_wait && g_hga_cuda_set_node_callback;
    }
    initialized = true;
    g_hga_cuda_event_record = ggml_cuda_hga_event_record;
    g_hga_cuda_event_wait   = ggml_cuda_hga_event_wait;
    g_hga_cuda_set_node_callback = ggml_cuda_hga_set_node_callback;
    if (!g_hga_cuda_event_record) {
        g_hga_cuda_event_record = reinterpret_cast<hga_cuda_event_fn>(
                hga_cuda_backend_symbol("ggml_cuda_hga_event_record"));
    }
    if (!g_hga_cuda_event_wait) {
        g_hga_cuda_event_wait = reinterpret_cast<hga_cuda_event_fn>(
                hga_cuda_backend_symbol("ggml_cuda_hga_event_wait"));
    }
    if (!g_hga_cuda_set_node_callback) {
        g_hga_cuda_set_node_callback = reinterpret_cast<hga_cuda_set_node_callback_fn>(
                hga_cuda_backend_symbol("ggml_cuda_hga_set_node_callback"));
    }
    return g_hga_cuda_event_record && g_hga_cuda_event_wait && g_hga_cuda_set_node_callback;
}

static void hga_trim_prefill_vmm() {
    using hga_vmm_shrink_fn = void (*)(void);
    hga_vmm_shrink_fn shrink = ggml_cuda_hga_vmm_shrink;
#if defined(__GNUC__) || defined(__clang__)
    if (!shrink) {
        shrink = reinterpret_cast<hga_vmm_shrink_fn>(
            dlsym(RTLD_DEFAULT, "ggml_cuda_hga_vmm_shrink"));
    }
    if (!shrink) {
#ifdef RTLD_NOLOAD
        void *cuda_so = dlopen("libggml-cuda.so", RTLD_NOW | RTLD_NOLOAD);
        if (cuda_so) {
            shrink = reinterpret_cast<hga_vmm_shrink_fn>(
                dlsym(cuda_so, "ggml_cuda_hga_vmm_shrink"));
            dlclose(cuda_so);
        }
#endif
    }
#endif
    if (!shrink) {
        hga_swap_log("VMM shrink hook unavailable; refusing leftover VERIFY pin");
        return;
    }
    hga_swap_log("calling explicit CUDA VMM shrink before leftover VERIFY pin");
    shrink();
}

struct hga_slot {
    ggml_tensor * t            = nullptr;
    ggml_backend_buffer_t host_buffer = nullptr;
    void * host_data           = nullptr;
    void * host_extra          = nullptr;
    bool on_cuda               = false;
};

struct hga_weight_swap {
    const llama_model * model = nullptr;
    ggml_backend_t gpu        = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    hga_swap_phase phase      = HGA_SWAP_NONE;

    std::vector<hga_slot> q;
    std::vector<hga_slot> kv;
    std::vector<hga_slot> output;

    ggml_backend_buffer_t buf_q   = nullptr;
    ggml_backend_buffer_t buf_kv  = nullptr;
    ggml_backend_buffer_t buf_out = nullptr;

    bool verified = false;

    /* Default prefill: 8 slots, step-4, offset 32
     * (0↔32, 4↔36, …, 28↔60). HGA_VERIFY_STREAMS chooses 2 or 3 of
     * those slots after prefill; the other pairs become CUDA-resident.
     * HGA_STREAM_2=1 restores the old two-pair topology for every phase and is
     * too large for the current prefill graph. HGA_STREAM_UNIFORM=1 is
     * 10↔42, 21↔53, 32↔63. Spec keeps MTP blk.64 CUDA-resident. */
    struct stream_pair {
        std::vector<hga_slot> a, b, extra_a, extra_b;
        std::vector<size_t>   off_a, off_b, off_ea, off_eb;
        size_t bytes_a = 0, bytes_b = 0, bytes_a_pf = 0, bytes_b_pf = 0;
        void * pin_a = nullptr, * pin_b = nullptr;
        void * pin_a_pf = nullptr, * pin_b_pf = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        size_t cap = 0;
        int occ = 0, copy = 0;
        uint8_t * copy_dst = nullptr;       /* next paced destination byte */
        const uint8_t * copy_src = nullptr; /* next paced source byte */
        size_t copy_left = 0;               /* paced bytes not submitted yet */
        size_t copy_step = 0;               /* bytes submitted at one HGA boundary */
        int layer_a = -1, layer_b = -1;
        int wait_a = -1, wait_b = -1;
        int wait_a_pf = -1, wait_b_pf = -1;
        char tag[12] = {};
        int n_kick_b = 0, n_stall_b = 0;
        void * copy_stream = nullptr; /* independent of ggml compute and of the other pair */
        void * ev_done = nullptr;     /* recorded after memcpy; consumer joins here */
        void * ev_compute = nullptr;  /* copy stream waits until the consumer releases the slot */
        bool verify_pair = true;      /* generate/VERIFY streams this pair; else pin CUDA */
        bool both_resident = false;   /* leftover VERIFY pair: A and B both on CUDA */
        bool prefill_resident = false;/* omit this exchange stream during PREFILL */
        bool split_ffn = false;       /* DECODE/VERIFY tiled FFN; cores stay resident */
    } pair[8];
    int n_pairs = 8;
    std::vector<hga_slot> ffn_res; /* FFN 31+63, decode-resident */
    std::vector<size_t>   ffn_res_off;
    ggml_backend_buffer_t buf_ffn_res = nullptr;
    bool leftover_pinned = false; /* leftover VERIFY pairs attempted after drop-prefill */
    bool stream_ok = false;
    bool async_events = false;    /* CUDA-event bridge available for inline dependencies */
    bool stream_timing = false;   /* diagnostic: report CPU wait remaining at slot consumers */
    size_t stream_chunk = 0;      /* verify H2D chunk size; 0 submits one large DMA */
    bool stream_paced = false;    /* submit one weight segment after each completed HGA */
    int stream_paced_pairs = 0;   /* supported two/three-pair periodic plan */
    bool prefill_stream_async = false; /* inline CUDA events; no scheduler host sync */
    bool prefill_stream_paced = false; /* delay one full pair image to post-HGA window */
    int stream_pace_last_layer = -1; /* suppress duplicate norm-N CUDA nodes */
    int  layer_last = -1;
    int  layer_mtp  = -1;
    ggml_backend_sched_eval_callback prev_eval = nullptr;
    void * prev_eval_ud = nullptr;
    hga_split_ffn * split = nullptr; /* DECODE/VERIFY tiled FFN plan; null = whole-layer */

    /* Placement census: resident CUDA0 weights vs host-mmap / CUDA_Host. */
    enum {
        PIN_RESIDENT = 0, /* must stay CUDA0; CUDA_Host is the silent fallback */
        PIN_STAGED   = 1, /* dense Q/K/V we H2D in PREFILL/DECODE */
        PIN_LMHEAD   = 2,
        PIN_EXCHANGE = 3, /* stream pair A/B; CUDA only while bound to the slot */
        PIN_XFFN     = 4, /* FFN 31/63: stream in prefill, optional decode-resident */
        PIN_HOST     = 5, /* embed, q/k-norm, -ot CPU leftovers */
        PIN_NCLS     = 6,
    };
    struct pin_rec {
        ggml_tensor * t = nullptr;
        uint8_t cls = PIN_RESIDENT;
        int8_t last_kind = -1; /* -1 unknown, 0 GPU, 1 CUDA_Host, 2 CPU, 3 null */
        char last_buf[24] = {};
    };
    std::vector<pin_rec> pins;
};

static double hga_mib(size_t n) {
    return n / (1024.0 * 1024.0);
}

static bool hga_name_is_gdn_qkv(const char * name) {
    return name && std::strstr(name, "attn_qkv") != nullptr;
}

static size_t hga_nbytes_sum(const std::vector<hga_slot> & slots) {
    size_t n = 0;
    for (const auto & s : slots) {
        n += ggml_nbytes(s.t);
    }
    return n;
}

static size_t hga_staging_bytes(ggml_backend_buffer_type_t buft, const std::vector<hga_slot> & slots) {
    const size_t align = ggml_backend_buft_get_alignment(buft);
    size_t n = align;
    for (const auto & s : slots) {
        n += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, s.t), align);
    }
    return n;
}

static void hga_swap_log(const char * fmt, ...) {
    /* llama.cpp's default verbosity hides most LLAMA_LOG_INFO. Always print. */
    va_list ap;
    va_start(ap, fmt);
    fputs("hga-swap: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

/* Ring of recent CUDA weight allocations, keyed by module, so an OOM dump
 * can name who took the last slices of VRAM. */
static constexpr int HGA_ALLOC_RING = 32;
static constexpr int HGA_ALLOC_DUMP_N = 20;
static constexpr int HGA_ALLOC_LIVE_MAX = 64;

struct hga_alloc_ev {
    int seq = 0;
    char module[32] = {};
    char who[48] = {};
    size_t bytes = 0;
    double free_before = 0;
    double free_after = 0;
    int ok = 0;
};

struct hga_alloc_live {
    ggml_backend_buffer_t buf = nullptr;
    char module[32] = {};
    char who[48] = {};
    size_t bytes = 0;
};

static struct {
    int seq = 0;
    int n = 0;
    int head = 0;
    hga_alloc_ev ring[HGA_ALLOC_RING];
    hga_alloc_live live[HGA_ALLOC_LIVE_MAX];
    int n_live = 0;
    ggml_backend_t gpu = nullptr;
} hga_amem;

static double hga_gpu_free_mib(ggml_backend_t gpu) {
    if (!gpu) {
        return 0;
    }
    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(gpu);
    if (dev) {
        ggml_backend_dev_memory(dev, &free_b, &total_b);
    }
    return hga_mib(free_b);
}

static void hga_alloc_live_del(ggml_backend_buffer_t buf) {
    if (!buf) {
        return;
    }
    for (int i = 0; i < hga_amem.n_live; ++i) {
        if (hga_amem.live[i].buf != buf) {
            continue;
        }
        hga_amem.live[i] = hga_amem.live[hga_amem.n_live - 1];
        hga_amem.n_live--;
        return;
    }
}

static void hga_alloc_live_add(ggml_backend_buffer_t buf, const char * module,
                               const char * who, size_t bytes) {
    if (!buf) {
        return;
    }
    hga_alloc_live_del(buf);
    if (hga_amem.n_live >= HGA_ALLOC_LIVE_MAX) {
        return;
    }
    hga_alloc_live & L = hga_amem.live[hga_amem.n_live++];
    L.buf = buf;
    L.bytes = bytes;
    std::snprintf(L.module, sizeof(L.module), "%s", module ? module : "?");
    std::snprintf(L.who, sizeof(L.who), "%s", who ? who : "?");
}

static void hga_alloc_push(const char * module, const char * who, size_t bytes,
                           double free_before, double free_after, int ok) {
    hga_alloc_ev & e = hga_amem.ring[hga_amem.head];
    e.seq = ++hga_amem.seq;
    e.bytes = bytes;
    e.free_before = free_before;
    e.free_after = free_after;
    e.ok = ok;
    std::snprintf(e.module, sizeof(e.module), "%s", module ? module : "?");
    std::snprintf(e.who, sizeof(e.who), "%s", who ? who : "?");
    hga_amem.head = (hga_amem.head + 1) % HGA_ALLOC_RING;
    if (hga_amem.n < HGA_ALLOC_RING) {
        hga_amem.n++;
    }
    hga_swap_log("hga-alloc: #%d  module=%s  who=%s  mib=%.2f  free_before=%.0f  "
            "free_after=%.0f  ok=%d",
            e.seq, e.module, e.who, hga_mib(bytes), free_before, free_after, ok);
}

static void hga_alloc_dump_oom(const char * why) {
    fprintf(stderr, "hga-alloc-oom: BEGIN why=%s\n", why ? why : "?");
    fprintf(stderr, "hga-alloc-oom: last%d\n", HGA_ALLOC_DUMP_N);
    const int n = hga_amem.n;
    const int dump = n < HGA_ALLOC_DUMP_N ? n : HGA_ALLOC_DUMP_N;
    for (int i = 0; i < dump; ++i) {
        const int idx = (hga_amem.head - dump + i + HGA_ALLOC_RING) % HGA_ALLOC_RING;
        const hga_alloc_ev & e = hga_amem.ring[idx];
        fprintf(stderr,
                "hga-alloc-oom: %2d  module=%s  who=%s  mib=%.2f  "
                "free_before=%.0f  free_after=%.0f  ok=%d\n",
                e.seq, e.module, e.who, hga_mib(e.bytes),
                e.free_before, e.free_after, e.ok);
    }
    fprintf(stderr, "hga-alloc-oom: live by module\n");
    struct {
        char module[32];
        int n;
        double mib;
    } sum[16];
    int ns = 0;
    for (int i = 0; i < hga_amem.n_live; ++i) {
        const hga_alloc_live & L = hga_amem.live[i];
        int k = 0;
        for (; k < ns; ++k) {
            if (std::strcmp(sum[k].module, L.module) == 0) {
                break;
            }
        }
        if (k == ns) {
            if (ns >= 16) {
                continue;
            }
            std::snprintf(sum[ns].module, sizeof(sum[ns].module), "%s", L.module);
            sum[ns].n = 0;
            sum[ns].mib = 0;
            k = ns++;
        }
        sum[k].n++;
        sum[k].mib += hga_mib(L.bytes);
    }
    for (int k = 0; k < ns; ++k) {
        fprintf(stderr, "hga-alloc-oom:  module=%s  n=%d  mib=%.1f\n",
                sum[k].module, sum[k].n, sum[k].mib);
    }
    fprintf(stderr, "hga-alloc-oom: END\n");
    std::fflush(stderr);
}

static ggml_backend_buffer_t hga_alloc_cuda(hga_weight_swap * sw, const char * module,
                                            const char * who, size_t bytes) {
    if (!sw || !sw->buft) {
        return nullptr;
    }
    hga_amem.gpu = sw->gpu;
    const double fb = hga_gpu_free_mib(sw->gpu);
    char alloc_scope[24];
    std::snprintf(alloc_scope, sizeof(alloc_scope), "%s:%s",
            module && module[0] ? module : "hga",
            who && who[0] ? who : "?");
    ggml_backend_hga_alloc_scope_set(alloc_scope);
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(sw->buft, bytes);
    ggml_backend_hga_alloc_scope_set(nullptr);
    const double fa = hga_gpu_free_mib(sw->gpu);
    const int ok = buf ? 1 : 0;
    hga_alloc_push(module, who, bytes, fb, fa, ok);
    if (ok) {
        hga_alloc_live_add(buf, module, who, bytes);
    } else {
        hga_alloc_dump_oom(who);
    }
    return buf;
}

static void hga_log_tensor(const char * tag, const ggml_tensor * t) {
    const char * bname = (t && t->buffer) ? ggml_backend_buffer_name(t->buffer) : "null";
    const int host = (t && t->buffer) ? (int) ggml_backend_buffer_is_host(t->buffer) : -1;
    hga_swap_log("  %-8s %-32s  buf=%-12s host=%d  %.2f MiB",
            tag,
            t && t->name[0] ? t->name : "?",
            bname, host,
            t ? hga_mib(ggml_nbytes(t)) : 0.0);
}

static bool hga_slot_from_tensor(ggml_tensor * t, hga_slot & out, const char * expect) {
    if (!t) {
        return false;
    }
    if (hga_name_is_gdn_qkv(t->name)) {
        hga_swap_log("refusing GDN tensor %s (regex too broad?)", t->name);
        return false;
    }
    if (!t->data) {
        return false; /* fit-probe context: weights not loaded yet */
    }
    if (!t->buffer || !ggml_backend_buffer_is_host(t->buffer)) {
        const char * bname = t->buffer ? ggml_backend_buffer_name(t->buffer) : "null";
        const int host = t->buffer ? (int) ggml_backend_buffer_is_host(t->buffer) : -1;
        hga_swap_log("%s is not host mmap (need -ot CPU); skip %s  buf=%s host=%d",
                t->name[0] ? t->name : "?", expect, bname, host);
        return false;
    }
    out.t           = t;
    out.host_buffer = t->buffer;
    out.host_data   = t->data;
    out.host_extra  = t->extra;
    out.on_cuda     = false;
    return true;
}

static bool hga_verify_slot(const hga_slot & s) {
    const size_t n = ggml_nbytes(s.t);
    std::vector<uint8_t> tmp(n);
    ggml_backend_tensor_get(s.t, tmp.data(), 0, n);
    return std::memcmp(tmp.data(), s.host_data, n) == 0;
}

static void hga_cuda_to_host(std::vector<hga_slot> & slots, ggml_backend_buffer_t * buf) {
    for (auto & s : slots) {
        if (!s.t) {
            continue;
        }
        s.t->buffer = s.host_buffer;
        s.t->data   = s.host_data;
        s.t->extra  = s.host_extra;
        s.on_cuda   = false;
    }
    if (buf && *buf) {
        hga_alloc_live_del(*buf);
        ggml_backend_buffer_free(*buf);
        *buf = nullptr;
    }
}

static bool hga_host_to_cuda(
        hga_weight_swap * sw,
        std::vector<hga_slot> & slots,
        ggml_backend_buffer_t * buf,
        const char * tag) {
    if (slots.empty()) {
        return true;
    }
    if (*buf) {
        return true;
    }

    const size_t bytes = hga_staging_bytes(sw->buft, slots);
    ggml_backend_buffer_t new_buf = hga_alloc_cuda(sw, tag, tag, bytes);
    if (!new_buf) {
        hga_swap_log("CUDA alloc %.2f MiB for %s failed", hga_mib(bytes), tag);
        return false;
    }
    ggml_backend_buffer_set_usage(new_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (auto & s : slots) {
        s.t->buffer = nullptr;
        s.t->data   = nullptr;
        s.t->extra  = nullptr;
    }

    ggml_tallocr tallocr = ggml_tallocr_new(new_buf);
    for (auto & s : slots) {
        if (ggml_tallocr_alloc(&tallocr, s.t) != GGML_STATUS_SUCCESS) {
            hga_swap_log("tallocr_alloc failed for %s (%s)", s.t->name, tag);
            for (auto & r : slots) {
                r.t->buffer = r.host_buffer;
                r.t->data   = r.host_data;
                r.t->extra  = r.host_extra;
                r.on_cuda   = false;
            }
            hga_alloc_live_del(new_buf);
            ggml_backend_buffer_free(new_buf);
            return false;
        }
    }

    for (auto & s : slots) {
        ggml_backend_tensor_set(s.t, s.host_data, 0, ggml_nbytes(s.t));
        s.on_cuda = true;
    }

    *buf = new_buf;
    return true;
}

static void hga_log_cuda_mem(ggml_backend_t gpu, const char * when) {
    ggml_backend_dev_t dev = ggml_backend_get_device(gpu);
    if (!dev) {
        return;
    }
    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_memory(dev, &free_b, &total_b);
    hga_swap_log("CUDA memory %s: free %.0f / total %.0f MiB  (used %.0f)",
            when, hga_mib(free_b), hga_mib(total_b),
            hga_mib(total_b > free_b ? total_b - free_b : 0));
}

static void hga_vram_step(hga_weight_swap * sw, const char * when,
                          uint32_t n_tokens = 0, uint32_t n_ubatch = 0, uint32_t n_rs = 0) {
    if (!sw || !sw->gpu) {
        return;
    }
    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(sw->gpu);
    if (dev) {
        ggml_backend_dev_memory(dev, &free_b, &total_b);
    }
    size_t slot = 0;
    for (int i = 0; i < sw->n_pairs; ++i) {
        slot += sw->pair[i].cap;
    }
    hga_swap_log("VRAM %-32s  free=%6.0f used=%6.0f  slots=%.1f MiB  n_tok=%u n_ubatch=%u n_rs=%u phase=%d",
            when,
            hga_mib(free_b),
            hga_mib(total_b > free_b ? total_b - free_b : 0),
            hga_mib(slot),
            n_tokens, n_ubatch, n_rs, (int) sw->phase);
}

/* libcudart via dlopen so this .cpp stays a regular CXX file (no nvcc). */
enum { HGA_CU_H2D = 1, HGA_CU_STREAM_NONBLOCKING = 1, HGA_CU_EVENT_DISABLE_TIMING = 2 };

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
    int (*event_sync)(void *) = nullptr;
    int (*event_query)(void *) = nullptr;
    const char * (*err_str)(int) = nullptr;
} hga_cu;

static bool hga_cu_init() {
    if (hga_cu.inited) {
        return hga_cu.ok;
    }
    hga_cu.inited = true;
    hga_cu.lib = dlopen("libcudart.so.12", RTLD_NOW | RTLD_LOCAL);
    if (!hga_cu.lib) {
        hga_cu.lib = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!hga_cu.lib) {
        hga_swap_log("stream: libcudart not found");
        return false;
    }
#define HGA_CU_SYM(field, name) do { \
        *(void **) &hga_cu.field = dlsym(hga_cu.lib, name); \
        if (!hga_cu.field) { \
            hga_swap_log("stream: missing %s", name); \
            return false; \
        } \
    } while (0)
    HGA_CU_SYM(malloc_host,    "cudaMallocHost");
    HGA_CU_SYM(free_host,      "cudaFreeHost");
    HGA_CU_SYM(stream_create,  "cudaStreamCreateWithFlags");
    HGA_CU_SYM(stream_destroy, "cudaStreamDestroy");
    HGA_CU_SYM(stream_sync,    "cudaStreamSynchronize");
    HGA_CU_SYM(stream_wait_event, "cudaStreamWaitEvent");
    HGA_CU_SYM(memcpy_async,   "cudaMemcpyAsync");
    HGA_CU_SYM(event_create,   "cudaEventCreateWithFlags");
    HGA_CU_SYM(event_destroy,  "cudaEventDestroy");
    HGA_CU_SYM(event_record,   "cudaEventRecord");
    HGA_CU_SYM(event_sync,     "cudaEventSynchronize");
    HGA_CU_SYM(event_query,    "cudaEventQuery");
    HGA_CU_SYM(err_str,        "cudaGetErrorString");
#undef HGA_CU_SYM
    hga_cu.ok = true;
    return true;
}

static const char * hga_cu_err(int rc) {
    return (hga_cu.err_str && rc) ? hga_cu.err_str(rc) : "ok";
}

static bool hga_bind_host(ggml_tensor * t, hga_slot & out) {
    if (!t || !t->data || !t->buffer || !ggml_backend_buffer_is_host(t->buffer)) {
        return false;
    }
    out.t           = t;
    out.host_buffer = t->buffer;
    out.host_data   = t->data;
    out.host_extra  = t->extra;
    out.on_cuda     = false;
    return true;
}

static void hga_collect_one(ggml_tensor * t, std::vector<hga_slot> & out,
                            std::unordered_set<const void *> & seen) {
    if (!t || !t->data) {
        return;
    }
    if (t->view_src) {
        return;
    }
    const char * nm = t->name;
    if (nm && (std::strstr(nm, "embed_tokens") || std::strstr(nm, "shared_head_head"))) {
        hga_swap_log("stream skip %s  %.1f MiB (shared embed/head)",
                nm, hga_mib(ggml_nbytes(t)));
        return;
    }
    if (!seen.insert(t->data).second) {
        return;
    }
    hga_slot s;
    if (!hga_bind_host(t, s)) {
        hga_swap_log("stream skip %s (not host mmap)", nm && nm[0] ? nm : "?");
        return;
    }
    out.push_back(s);
}

static void hga_collect_layer(const llama_layer & layer, std::vector<hga_slot> & out,
                              std::unordered_set<const void *> & seen,
                              bool skip_dense_qkv) {
    ggml_tensor * ts[] = {
        layer.attn_norm, layer.attn_post_norm,
        skip_dense_qkv ? nullptr : layer.attn_q_norm,
        skip_dense_qkv ? nullptr : layer.attn_k_norm,
        skip_dense_qkv ? nullptr : layer.wq,
        skip_dense_qkv ? nullptr : layer.wk,
        skip_dense_qkv ? nullptr : layer.wv,
        layer.wo,
        layer.wqkv, layer.wqkv_gate, layer.wg,
        layer.ssm_conv1d, layer.ssm_dt, layer.ssm_a, layer.ssm_out, layer.ssm_norm,
        layer.ssm_alpha, layer.ssm_beta, layer.ssm_beta_alpha,
        layer.ffn_gate, layer.ffn_down, layer.ffn_up,
        layer.nextn.eh_proj, layer.nextn.enorm, layer.nextn.hnorm,
        layer.nextn.embed_tokens, layer.nextn.shared_head_head, layer.nextn.shared_head_norm,
    };
    for (ggml_tensor * t : ts) {
        hga_collect_one(t, out, seen);
    }
}

static size_t hga_stream_layout(hga_weight_swap * sw, const std::vector<hga_slot> & slots,
                                std::vector<size_t> & offs) {
    const size_t align = ggml_backend_buft_get_alignment(sw->buft);
    size_t off = align;
    offs.resize(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        off = GGML_PAD(off, align);
        offs[i] = off;
        off += GGML_PAD(ggml_backend_buft_get_alloc_size(sw->buft, slots[i].t), align);
    }
    return off;
}

static void hga_stream_unbind_group(std::vector<hga_slot> & slots) {
    for (auto & s : slots) {
        if (!s.t) {
            continue;
        }
        s.t->buffer = s.host_buffer;
        s.t->data   = s.host_data;
        s.t->extra  = s.host_extra;
        s.on_cuda   = false;
    }
}

static void hga_stream_bind_group(ggml_backend_buffer_t buf, std::vector<hga_slot> & slots,
                                  const std::vector<size_t> & offs) {
    if (!buf) {
        return;
    }
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buf);
    for (size_t i = 0; i < slots.size(); ++i) {
        auto & s = slots[i];
        s.t->buffer   = buf;
        s.t->data     = base + offs[i];
        s.t->view_src = nullptr;
        s.t->view_offs = 0;
        ggml_backend_buffer_init_tensor(buf, s.t);
        s.on_cuda = true;
    }
}

static void hga_stream_pack_pin(void * pin, const std::vector<hga_slot> & slots,
                                const std::vector<size_t> & offs, size_t bytes) {
    std::memset(pin, 0, bytes);
    for (size_t i = 0; i < slots.size(); ++i) {
        std::memcpy((uint8_t *) pin + offs[i], slots[i].host_data, ggml_nbytes(slots[i].t));
    }
}

static void hga_pair_clear_buf(hga_weight_swap::stream_pair & p) {
    hga_stream_unbind_group(p.a);
    hga_stream_unbind_group(p.b);
    hga_stream_unbind_group(p.extra_a);
    hga_stream_unbind_group(p.extra_b);
    if (p.buf) {
        hga_alloc_live_del(p.buf);
        ggml_backend_buffer_free(p.buf);
        p.buf = nullptr;
    }
    p.cap = 0;
    p.occ = 0;
    p.copy = 0;
    p.copy_dst = nullptr;
    p.copy_src = nullptr;
    p.copy_left = 0;
    p.copy_step = 0;
    p.both_resident = false;
}

static void hga_stream_free_buffers(hga_weight_swap * sw) {
    for (int i = 0; i < sw->n_pairs; ++i) {
        auto & p = sw->pair[i];
        if (p.copy_stream && hga_cu.stream_sync) {
            hga_cu.stream_sync(p.copy_stream);
        }
        hga_pair_clear_buf(p);
        auto free_pin = [&](void *& ptr) {
            if (ptr && hga_cu.free_host) {
                hga_cu.free_host(ptr);
                ptr = nullptr;
            }
        };
        free_pin(p.pin_a);
        free_pin(p.pin_b);
        free_pin(p.pin_a_pf);
        free_pin(p.pin_b_pf);
        if (p.ev_done && hga_cu.event_destroy) {
            hga_cu.event_destroy(p.ev_done);
            p.ev_done = nullptr;
        }
        if (p.ev_compute && hga_cu.event_destroy) {
            hga_cu.event_destroy(p.ev_compute);
            p.ev_compute = nullptr;
        }
        if (p.copy_stream && hga_cu.stream_destroy) {
            hga_cu.stream_destroy(p.copy_stream);
            p.copy_stream = nullptr;
        }
    }
    hga_stream_unbind_group(sw->ffn_res);
    if (sw->buf_ffn_res) {
        hga_alloc_live_del(sw->buf_ffn_res);
        ggml_backend_buffer_free(sw->buf_ffn_res);
        sw->buf_ffn_res = nullptr;
    }
    sw->leftover_pinned = false;
}

static bool hga_pair_ensure(hga_weight_swap * sw, hga_weight_swap::stream_pair & p,
                            size_t need, bool shrink) {
    if (p.buf && p.cap == need) {
        return true;
    }
    if (p.buf && p.cap >= need && !shrink) {
        return true;
    }
    if (p.copy_stream && hga_cu.stream_sync) {
        hga_cu.stream_sync(p.copy_stream);
        p.occ = p.copy ? p.copy : p.occ;
        p.copy = 0;
    }
    hga_pair_clear_buf(p);
    hga_log_cuda_mem(sw->gpu, "before stream slot alloc");
    p.buf = hga_alloc_cuda(sw, "stream-slot", p.tag, need);
    if (!p.buf) {
        hga_swap_log("stream %s CUDA alloc %.2f MiB failed", p.tag, hga_mib(need));
        hga_log_cuda_mem(sw->gpu, "after stream slot alloc FAIL");
        return false;
    }
    hga_swap_log("stream %s CUDA alloc %.2f MiB ok", p.tag, hga_mib(need));
    hga_log_cuda_mem(sw->gpu, "after stream slot alloc");
    ggml_backend_buffer_set_usage(p.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    p.cap = need;
    return true;
}

/* Finish submitting a paced copy before a consumer/phase transition waits on
 * ev_done.  The normal paced schedule reaches zero before this is needed; a
 * non-zero remainder is a deadline miss and is submitted as one final DMA so
 * correctness never depends on callback timing. */
static int hga_pair_submit_rest(hga_weight_swap * sw, hga_weight_swap::stream_pair & p) {
    if (!p.copy || !p.copy_left) {
        return 0;
    }
    const size_t n = p.copy_left;
    const int rc = hga_cu.memcpy_async(
            p.copy_dst, p.copy_src, n, HGA_CU_H2D, p.copy_stream);
    if (rc != 0) {
        hga_swap_log("stream %s paced remainder H2D failed: %s", p.tag, hga_cu_err(rc));
        return rc;
    }
    p.copy_dst += n;
    p.copy_src += n;
    p.copy_left = 0;
    if (p.ev_done && hga_cu.event_record) {
        hga_cu.event_record(p.ev_done, p.copy_stream);
    }
    if (sw && sw->stream_timing) {
        hga_swap_log("stream-timing %s paced deadline flush-%c %.1f MiB",
                p.tag, p.copy == 2 ? 'B' : 'A', hga_mib(n));
    }
    return 0;
}

/* Join the CPU only if this pair's copy is not done. Does not touch the
 * other pair or the ggml compute stream. */
static void hga_pair_join(hga_weight_swap * sw, hga_weight_swap::stream_pair & p) {
    if (!p.copy || !p.ev_done || !hga_cu.event_sync) {
        if (p.copy) {
            p.occ = p.copy;
            p.copy = 0;
        }
        return;
    }
    if (hga_pair_submit_rest(sw, p) != 0) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    const int rc = hga_cu.event_sync(p.ev_done);
    const double ms = (ggml_time_us() - t0) / 1000.0;
    if (rc != 0) {
        hga_swap_log("stream %s event sync failed: %s", p.tag, hga_cu_err(rc));
    }
    if (sw && sw->stream_timing) {
        hga_swap_log("stream-timing %s wait-%c %.3f ms",
                p.tag, p.copy == 2 ? 'B' : 'A', ms);
    }
    if (p.copy == 2) {
        if (ms >= 1.0) {
            p.n_stall_b++;
        }
        if (ms >= 0.5) {
            hga_swap_log("stream %s join B  %.2f ms  (stalls=%d  kicks=%d)",
                    p.tag, ms, p.n_stall_b, p.n_kick_b);
        }
    }
    p.occ = p.copy;
    p.copy = 0;
}

static void hga_stream_wait_all(hga_weight_swap * sw) {
    for (int i = 0; i < sw->n_pairs; ++i) {
        hga_pair_join(sw, sw->pair[i]);
    }
}

/* Order a consumer behind the H2D without blocking its host thread.  Once
 * cudaStreamWaitEvent has been queued, the slot is logically resident for
 * subsequent callback decisions even if the DMA is still in flight. */
static void hga_pair_wait_async(hga_weight_swap * sw, hga_weight_swap::stream_pair & p) {
    if (!p.copy) {
        return;
    }
    if (hga_pair_submit_rest(sw, p) != 0) {
        hga_swap_log("stream %s cannot finish paced copy before consumer", p.tag);
        return;
    }
    if (p.ev_done && g_hga_cuda_event_wait &&
            g_hga_cuda_event_wait(sw->gpu, p.ev_done)) {
        p.occ = p.copy;
        p.copy = 0;
        return;
    }
    hga_swap_log("stream %s compute wait-event failed; CPU-joining copy", p.tag);
    hga_pair_join(sw, p);
}

static void hga_pair_kick(hga_weight_swap * sw, int pi, int dest,
                          bool after_compute = false) {
    if (!sw->stream_ok || pi < 0 || pi >= sw->n_pairs) {
        return;
    }
    auto & p = sw->pair[pi];
    if (p.split_ffn) {
        return;
    }
    if (!p.buf || !p.copy_stream) {
        return;
    }
    if (p.occ == dest && p.copy == 0) {
        return;
    }
    if (p.copy == dest) {
        return;
    }
    /* Same buffer: a B copy already queued on this pair's stream will complete
     * before the next memcpy on that stream. Do not CPU-wait the other pair. */
    if (p.copy && p.copy != dest) {
        hga_pair_join(sw, p);
    }
    const bool pf = (sw->phase == HGA_SWAP_PREFILL);
    const void * src = nullptr;
    size_t n = 0;
    if (dest == 1) {
        src = (pf && p.pin_a_pf) ? p.pin_a_pf : p.pin_a;
        n   = (pf && p.pin_a_pf) ? p.bytes_a_pf : p.bytes_a;
    } else {
        src = (pf && p.pin_b_pf) ? p.pin_b_pf : p.pin_b;
        n   = (pf && p.pin_b_pf) ? p.bytes_b_pf : p.bytes_b;
    }
    if (!src || n == 0) {
        return;
    }
    if (after_compute) {
        const bool recorded = p.ev_compute && g_hga_cuda_event_record &&
                g_hga_cuda_event_record(sw->gpu, p.ev_compute);
        const int wait_rc = recorded && hga_cu.stream_wait_event ?
                hga_cu.stream_wait_event(p.copy_stream, p.ev_compute, 0) : -1;
        if (!recorded || wait_rc != 0) {
            hga_swap_log("stream %s release event failed (record=%d wait=%s); synchronizing compute",
                    p.tag, (int) recorded, wait_rc < 0 ? "unavailable" : hga_cu_err(wait_rc));
            ggml_backend_synchronize(sw->gpu);
        }
    }
    void * dst = ggml_backend_buffer_get_base(p.buf);
    const bool pf_paced = sw->prefill_stream_paced &&
            sw->phase == HGA_SWAP_PREFILL;
    const bool verify_paced = sw->stream_paced &&
            hga_decode_pack((int32_t) sw->phase);
    const bool paced = after_compute && (pf_paced || verify_paced);
    if (paced) {
        p.copy = dest;
        p.copy_dst = (uint8_t *) dst;
        p.copy_src = (const uint8_t *) src;
        p.copy_left = n;
        /* Two pairs use four segments per transfer. The three-pair layout has
         * three ~211 MiB and three ~246 MiB images. Two small images use two
         * segments; the remaining four images use three: 2*2 + 4*3 = all
         * sixteen HGA-to-HGA windows. */
        int parts = pf_paced ? 1 : 4;
        if (!pf_paced && sw->stream_paced_pairs == 3) {
            const size_t small_limit = (size_t) 224 * 1024 * 1024;
            parts = n <= small_limit ? ((pi == 4 && dest == 2) ? 3 : 2) : 3;
        }
        p.copy_step = (n + (size_t) parts - 1) / (size_t) parts;
        if (sw->stream_timing) {
            hga_swap_log("stream-timing %s arm-%c %.1f MiB in %d paced segments",
                    p.tag, dest == 2 ? 'B' : 'A', hga_mib(n), parts);
        }
        if (dest == 2) {
            p.n_kick_b++;
        }
        return;
    }
    const int64_t memcpy_t0 = sw->stream_timing && hga_decode_pack((int32_t) sw->phase) ?
            ggml_time_us() : 0;
    const size_t chunk = hga_decode_pack((int32_t) sw->phase) ? sw->stream_chunk : 0;
    uint8_t * dst_cur = (uint8_t *) dst;
    const uint8_t * src_cur = (const uint8_t *) src;
    size_t left = n;
    int rc = 0;
    do {
        const size_t step = chunk && chunk < left ? chunk : left;
        rc = hga_cu.memcpy_async(dst_cur, src_cur, step, HGA_CU_H2D, p.copy_stream);
        if (rc != 0) {
            break;
        }
        dst_cur += step;
        src_cur += step;
        left -= step;
    } while (left);
    if (memcpy_t0) {
        hga_swap_log("stream-timing %s enqueue-%c %.3f ms  %.1f MiB",
                p.tag, dest == 2 ? 'B' : 'A',
                (ggml_time_us() - memcpy_t0) / 1000.0, hga_mib(n));
    }
    if (rc != 0) {
        hga_swap_log("stream %s H2D %c failed: %s", p.tag, dest == 1 ? 'A' : 'B', hga_cu_err(rc));
        return;
    }
    if (p.ev_done && hga_cu.event_record) {
        hga_cu.event_record(p.ev_done, p.copy_stream);
    }
    p.copy = dest;
    if (dest == 2) {
        p.n_kick_b++;
        if (p.n_kick_b <= 3) {
            hga_swap_log("stream %s H2D B  %.1f MiB  after blk.%d (until blk.%d)  async",
                    p.tag, hga_mib(n), p.layer_a, pf ? p.wait_b_pf : p.wait_b);
        }
    } else if (p.n_kick_b <= 3) {
        hga_swap_log("stream %s H2D A  %.1f MiB  async", p.tag, hga_mib(n));
    }
}

static bool hga_pair_submit_paced(hga_weight_swap * sw,
                                  hga_weight_swap::stream_pair & p, int il) {
    if (!p.copy || !p.copy_left || !p.copy_step) {
        return false;
    }
    const size_t n = std::min(p.copy_left, p.copy_step);
    const int rc = hga_cu.memcpy_async(
            p.copy_dst, p.copy_src, n, HGA_CU_H2D, p.copy_stream);
    if (rc != 0) {
        hga_swap_log("stream %s paced H2D %c failed: %s",
                p.tag, p.copy == 2 ? 'B' : 'A', hga_cu_err(rc));
        return false;
    }
    p.copy_dst += n;
    p.copy_src += n;
    p.copy_left -= n;
    if (!p.copy_left && p.ev_done && hga_cu.event_record) {
        hga_cu.event_record(p.ev_done, p.copy_stream);
    }
    if (sw->stream_timing) {
        hga_swap_log("stream-timing %s pace-%c after HGA before blk.%d  %.1f MiB  left=%.1f MiB",
                p.tag, p.copy == 2 ? 'B' : 'A', il,
                hga_mib(n), hga_mib(p.copy_left));
    }
    return true;
}

/* norm-(4*n) is the first CUDA node after full-attention layer 4*n-1.  The
 * scheduler has already completed the small CPU->GPU HGA activation copy at
 * this point. Submit at most one weight segment, choosing the transfer whose
 * consumer is closest in cyclic layer order. */
static void hga_stream_pace_after_hga(hga_weight_swap * sw, int il) {
    const bool pf = sw->phase == HGA_SWAP_PREFILL;
    const bool active = pf ? sw->prefill_stream_paced : sw->stream_paced;
    if (!active || il < 0 || il > sw->layer_last || (il & 3) != 0) {
        return;
    }
    if (sw->stream_pace_last_layer == il) {
        return;
    }
    sw->stream_pace_last_layer = il;
    const int n_layer = sw->layer_last + 1;
    int best = -1;
    int best_distance = n_layer + 1;
    for (int pi = 0; pi < sw->n_pairs; ++pi) {
        auto & p = sw->pair[pi];
        if ((pf && p.prefill_resident) || (!pf && !p.verify_pair) || p.split_ffn ||
                !p.copy || !p.copy_left) {
            continue;
        }
        const int consumer = p.copy == 1 ? (pf ? p.wait_a_pf : p.wait_a)
                                         : (pf ? p.wait_b_pf : p.wait_b);
        const int distance = (consumer - il + n_layer) % n_layer;
        if (distance < best_distance) {
            best = pi;
            best_distance = distance;
        }
    }
    if (best >= 0) {
        hga_pair_submit_paced(sw, sw->pair[best], il);
    }
}

static int hga_name_layer_id(const char * name) {
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

static bool hga_name_prefix(const char * name, const char * key) {
    const size_t n = std::strlen(key);
    return name && std::strncmp(name, key, n) == 0 && (name[n] == 0 || name[n] == '-');
}

/* Prefill Q/K/V D2H and packed-KV H2D, timed separately from HGA compute.
 * Only these named copies are synchronized so FFN/flash-attn stay overlapped. */
struct hga_xfer_prof {
    double d2h_ms = 0.0;
    double h2d_ms = 0.0;
    uint64_t d2h_bytes = 0;
    uint64_t h2d_bytes = 0;
    int d2h_n = 0;
    int h2d_n = 0;
    double win_d2h_ms = 0.0;
    double win_h2d_ms = 0.0;
    uint64_t win_d2h_bytes = 0;
    uint64_t win_h2d_bytes = 0;
    int win_h2d_n = 0;
    std::chrono::steady_clock::time_point t0{};
    bool timing = false;
    bool h2d = false;
    bool dumped = false;
};

static hga_xfer_prof g_hga_xfer;

static double hga_xfer_gib_s(uint64_t bytes, double ms) {
    if (ms <= 0.0 || bytes == 0) {
        return 0.0;
    }
    return (bytes / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
}

static void hga_xfer_dump_total() {
    if (g_hga_xfer.dumped || (g_hga_xfer.d2h_n == 0 && g_hga_xfer.h2d_n == 0)) {
        return;
    }
    g_hga_xfer.dumped = true;
    std::fprintf(stderr,
                 "hga-prof prefill xfer TOTAL d2h_qkv=%.1f ms / %.1f MiB (%.2f GiB/s)  "
                 "h2d_kv=%.1f ms / %.1f MiB (%.2f GiB/s)  n_d2h=%d n_h2d=%d\n",
                 g_hga_xfer.d2h_ms, g_hga_xfer.d2h_bytes / (1024.0 * 1024.0),
                 hga_xfer_gib_s(g_hga_xfer.d2h_bytes, g_hga_xfer.d2h_ms),
                 g_hga_xfer.h2d_ms, g_hga_xfer.h2d_bytes / (1024.0 * 1024.0),
                 hga_xfer_gib_s(g_hga_xfer.h2d_bytes, g_hga_xfer.h2d_ms),
                 g_hga_xfer.d2h_n, g_hga_xfer.h2d_n);
}

static const char * hga_xfer_kind(const ggml_tensor * t) {
    if (!t || !t->name[0]) {
        return nullptr;
    }
    const bool copy = t->op == GGML_OP_CPY || t->op == GGML_OP_DUP ||
            t->op == GGML_OP_CONT;
    if (!copy) {
        return nullptr;
    }
    const bool src_host = t->src[0] && t->src[0]->buffer &&
            ggml_backend_buffer_is_host(t->src[0]->buffer);
    const bool dst_host = t->buffer && ggml_backend_buffer_is_host(t->buffer);
    if (std::strstr(t->name, "hga_gpu_stage_h2d") ||
            std::strstr(t->name, "hga_gpu_stage_united") ||
            (t->src[0] && t->src[0]->name[0] &&
             std::strstr(t->src[0]->name, "hga_gpu_stage_cpu")) ||
            (src_host && !dst_host && ggml_nbytes(t) >= 256 * 1024)) {
        return "h2d";
    }
    if (std::strstr(t->name, "hga_prefill_Q_d2h") ||
            std::strstr(t->name, "hga_prefill_K_d2h") ||
            std::strstr(t->name, "hga_prefill_V_d2h") ||
            std::strstr(t->name, "hga_prefill_Kraw_d2h") ||
            (t->src[0] && t->src[0]->name[0] &&
             std::strstr(t->src[0]->name, "hga_prefill_") &&
             std::strstr(t->src[0]->name, "_d2h"))) {
        return "d2h";
    }
    return nullptr;
}

static void hga_xfer_begin(const ggml_tensor * t, ggml_backend_t gpu) {
    const char * kind = hga_xfer_kind(t);
    if (!kind || g_hga_xfer.timing) {
        return;
    }
    /* Drain in-flight GPU work so the timer is the copy, not the wait for
     * the previous layer's FFN/flash-attn. */
    if (gpu) {
        ggml_backend_synchronize(gpu);
    }
    g_hga_xfer.timing = true;
    g_hga_xfer.h2d = std::strcmp(kind, "h2d") == 0;
    g_hga_xfer.t0 = std::chrono::steady_clock::now();
}

static void hga_xfer_end(const ggml_tensor * t, ggml_backend_t gpu) {
    if (!g_hga_xfer.timing) {
        return;
    }
    g_hga_xfer.timing = false;
    if (gpu) {
        ggml_backend_synchronize(gpu);
    }
    const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - g_hga_xfer.t0).count();
    const uint64_t bytes = ggml_nbytes(t);
    static bool atexit_set = false;
    if (!atexit_set) {
        atexit_set = true;
        std::atexit(hga_xfer_dump_total);
    }
    if (g_hga_xfer.h2d) {
        g_hga_xfer.h2d_ms += ms;
        g_hga_xfer.h2d_bytes += bytes;
        g_hga_xfer.h2d_n += 1;
        g_hga_xfer.win_h2d_ms += ms;
        g_hga_xfer.win_h2d_bytes += bytes;
        g_hga_xfer.win_h2d_n += 1;
        if (g_hga_xfer.win_h2d_n == 16) {
            std::fprintf(stderr,
                         "hga-prof prefill xfer 16 layers d2h_qkv=%.1f ms / %.1f MiB (%.2f GiB/s)  "
                         "h2d_kv=%.1f ms / %.1f MiB (%.2f GiB/s)\n",
                         g_hga_xfer.win_d2h_ms,
                         g_hga_xfer.win_d2h_bytes / (1024.0 * 1024.0),
                         hga_xfer_gib_s(g_hga_xfer.win_d2h_bytes, g_hga_xfer.win_d2h_ms),
                         g_hga_xfer.win_h2d_ms,
                         g_hga_xfer.win_h2d_bytes / (1024.0 * 1024.0),
                         hga_xfer_gib_s(g_hga_xfer.win_h2d_bytes, g_hga_xfer.win_h2d_ms));
            g_hga_xfer.win_d2h_ms = g_hga_xfer.win_h2d_ms = 0.0;
            g_hga_xfer.win_d2h_bytes = g_hga_xfer.win_h2d_bytes = 0;
            g_hga_xfer.win_h2d_n = 0;
        }
    } else {
        g_hga_xfer.d2h_ms += ms;
        g_hga_xfer.d2h_bytes += bytes;
        g_hga_xfer.d2h_n += 1;
        g_hga_xfer.win_d2h_ms += ms;
        g_hga_xfer.win_d2h_bytes += bytes;
    }
}

static bool hga_stream_eval_cb(ggml_tensor * t, bool ask, void * user) {
    auto * sw = (hga_weight_swap *) user;
    const char * nm = (t && t->name[0]) ? t->name : "";
    const int il = hga_name_layer_id(nm);
    const bool live = sw->stream_ok &&
            (hga_decode_pack((int32_t) sw->phase) || sw->phase == HGA_SWAP_PREFILL);
    const bool pf = sw->phase == HGA_SWAP_PREFILL;
    const bool async = sw->async_events &&
            ((pf && sw->prefill_stream_async) ||
             (!pf && hga_decode_pack((int32_t) sw->phase)));
    const bool lout = hga_name_prefix(nm, "l_out") || hga_name_prefix(nm, "post_ffn");
    const bool anorm = hga_name_prefix(nm, "attn_norm");

    struct hit {
        int pi;
        int kind; /* 1 wait_a, 2 kick_b, 3 wait_b, 4 kick_a */
    };
    hit hits[32];
    int nh = 0;
    if (live) {
        for (int pi = 0; pi < sw->n_pairs; ++pi) {
            auto & p = sw->pair[pi];
            if ((pf && p.prefill_resident) || (!pf && !p.verify_pair) || p.split_ffn) {
                continue;
            }
            const int wa = pf ? p.wait_a_pf : p.wait_a;
            const int wb = pf ? p.wait_b_pf : p.wait_b;
            if ((lout && il == wa - 1) ||
                    (anorm && il == wa && p.occ != 1 && p.copy != 1)) {
                hits[nh++] = {pi, 1};
            }
            /* Some layer outputs (notably l_out-0 in Qwen3.8 VERIFY) are
             * aliases and never become evaluated nodes. attn_norm-(A+1)
             * is the first guaranteed node after layer A, and still leaves
             * B-A-1 whole layers in which to hide the copy. */
            /* copy is authoritative while a transfer is in flight. In
             * particular, occ can still say B while A is overwriting the
             * slot at the start of the next ubatch. */
            const bool effectively_b = p.copy == 2 || (p.copy == 0 && p.occ == 2);
            if (!effectively_b &&
                    ((lout && il == p.layer_a) ||
                     (anorm && il == p.layer_a + 1))) {
                hits[nh++] = {pi, 2};
            }
            if ((lout && il == wb - 1) || (anorm && il == wb && p.occ != 2)) {
                hits[nh++] = {pi, 3};
            }
            if (p.copy != 1 && p.occ != 1 && lout && il == p.layer_b) {
                hits[nh++] = {pi, 4};
            }
        }
    }

    if (ask) {
        /* Do not launch copies here. ask=true runs on every node; kicking A
         * whenever occ!=A would overwrite B while later layers still need it. */
        const bool xfer = hga_xfer_kind(t) != nullptr;
        if (xfer) {
            hga_xfer_begin(t, sw->gpu);
        }
        /* Returning true isolates D2H/H2D nodes so the timer is not fused
         * with FFN/flash-attn/HGA compute in the same scheduler split. */
        const bool ours = nh > 0 || xfer;
        const bool theirs = sw->prev_eval && sw->prev_eval(t, true, sw->prev_eval_ud);
        return ours || theirs;
    }

    for (int i = 0; i < nh; ++i) {
        const int pi = hits[i].pi;
        auto & p = sw->pair[pi];
        switch (hits[i].kind) {
        case 1:
            if (async) {
                hga_pair_wait_async(sw, p);
            } else {
                hga_pair_join(sw, p);
            }
            break;
        case 2:
            /* In async mode record layer A's completion on the ggml compute
             * stream, then make only this pair's copy stream wait for it. */
            hga_pair_kick(sw, pi, 2, async);
            break;
        case 3:
            /* If an A copy is still recorded, finish it before deciding
             * whether B is resident.  Looking only at occ here mistakes the
             * stale pre-copy occupant for the current contents. */
            if (p.copy && p.copy != 2) {
                if (async) {
                    hga_pair_wait_async(sw, p);
                } else {
                    hga_pair_join(sw, p);
                }
            }
            if (p.occ != 2 && p.copy != 2) {
                hga_swap_log("stream %s late H2D B (node %s)", p.tag, nm);
                hga_pair_kick(sw, pi, 2, async);
            }
            if (async) {
                hga_pair_wait_async(sw, p);
            } else {
                hga_pair_join(sw, p);
            }
            break;
        case 4:
            /* Queue A for the next token after layer B releases the slot. */
            hga_pair_kick(sw, pi, 1, async);
            break;
        }
    }
    hga_xfer_end(t, sw->gpu);
    return sw->prev_eval ? sw->prev_eval(t, false, sw->prev_eval_ud) : true;
}

/* Fast VERIFY path.  ggml-cuda calls this immediately before enqueueing each
 * CUDA node, so norm-N is a stable layer boundary: N waits for its slot,
 * while N+1 proves that all work from layer N is already on the compute
 * stream.  This preserves a single backend graph submission; the generic
 * scheduler callback remains the synchronous fallback. */
static void hga_stream_cuda_node_cb(ggml_tensor * t, void * user) {
    auto * sw = (hga_weight_swap *) user;
    if (!sw || !sw->stream_ok || !sw->async_events || !t) {
        return;
    }
    if (sw->split && hga_split_ffn_on_cuda_node(sw->split, t)) {
        return;
    }
    const bool pf = sw->phase == HGA_SWAP_PREFILL;
    if ((!pf && !hga_decode_pack((int32_t) sw->phase)) ||
            (pf && !sw->prefill_stream_async)) {
        return;
    }
    if (!hga_name_prefix(t->name, "norm")) {
        return;
    }
    const int il = hga_name_layer_id(t->name);
    if (il < 0) {
        return;
    }
    for (int pi = 0; pi < sw->n_pairs; ++pi) {
        auto & p = sw->pair[pi];
        if ((pf && p.prefill_resident) || (!pf && !p.verify_pair) || p.split_ffn) {
            continue;
        }
        const int wait_a = pf ? p.wait_a_pf : p.wait_a;
        const int wait_b = pf ? p.wait_b_pf : p.wait_b;
        if (il == wait_a) {
            hga_pair_wait_async(sw, p);
        }
        if (il == p.layer_a + 1) {
            const bool effectively_b = p.copy == 2 || (p.copy == 0 && p.occ == 2);
            if (!effectively_b) {
                hga_pair_kick(sw, pi, 2, true);
            }
        }
        if (il == wait_b) {
            if (p.copy && p.copy != 2) {
                hga_pair_wait_async(sw, p);
            }
            if (p.occ != 2 && p.copy != 2) {
                hga_swap_log("stream %s inline late H2D B (node %s)", p.tag, t->name);
                hga_pair_kick(sw, pi, 2, true);
            }
            hga_pair_wait_async(sw, p);
        }
        if (il == p.layer_b + 1 && p.copy != 1 && p.occ != 1) {
            hga_pair_kick(sw, pi, 1, true);
        }
    }
    hga_stream_pace_after_hga(sw, il);
}

static bool hga_cu_pin(void ** dst, size_t n, const char * tag) {
    const int rc = hga_cu.malloc_host(dst, n);
    if (rc != 0) {
        hga_swap_log("stream cudaMallocHost %s %.1f MiB failed: %s", tag, hga_mib(n), hga_cu_err(rc));
        *dst = nullptr;
        return false;
    }
    return true;
}

static size_t hga_layout_from(hga_weight_swap * sw, const std::vector<hga_slot> & slots,
                              std::vector<size_t> & offs, size_t start) {
    const size_t align = ggml_backend_buft_get_alignment(sw->buft);
    size_t off = GGML_PAD(start, align);
    offs.resize(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        off = GGML_PAD(off, align);
        offs[i] = off;
        off += GGML_PAD(ggml_backend_buft_get_alloc_size(sw->buft, slots[i].t), align);
    }
    return off;
}

static bool hga_stream_setup(hga_weight_swap * sw) {
    if (std::getenv("HGA_STREAM") && std::getenv("HGA_STREAM")[0] == '0') {
        hga_swap_log("HGA_STREAM=0 — weight streaming disabled");
        return false;
    }
    if (!hga_cu_init()) {
        return false;
    }
    const char * async_e = std::getenv("HGA_STREAM_ASYNC");
    const bool async_requested = !async_e || !async_e[0] || async_e[0] != '0';
    sw->async_events = async_requested && hga_cuda_event_bridge_init();
    const char * timing_e = std::getenv("HGA_STREAM_TIMING");
    sw->stream_timing = timing_e && timing_e[0] && timing_e[0] != '0';
    const char * paced_e = std::getenv("HGA_STREAM_PACED");
    const bool paced_requested = paced_e && paced_e[0] && paced_e[0] != '0';
    sw->stream_paced = paced_requested && sw->async_events;
    const char * pf_paced_e = std::getenv("HGA_PREFILL_STREAM_PACED");
    const bool pf_paced_requested =
            pf_paced_e && pf_paced_e[0] && pf_paced_e[0] != '0';
    const char * pf_async_e = std::getenv("HGA_PREFILL_STREAM_ASYNC");
    const bool pf_async_requested = pf_paced_requested ||
            (pf_async_e && pf_async_e[0] && pf_async_e[0] != '0');
    sw->prefill_stream_async = pf_async_requested && sw->async_events;
    sw->prefill_stream_paced = pf_paced_requested && sw->prefill_stream_async;
    if (const char * e = std::getenv("HGA_STREAM_CHUNK_MIB")) {
        char * end = nullptr;
        const long value = std::strtol(e, &end, 10);
        if (end != e && *end == '\0' && value >= 0 && value <= 256) {
            sw->stream_chunk = (size_t) value * 1024 * 1024;
        } else {
            hga_swap_log("invalid HGA_STREAM_CHUNK_MIB=%s (expected 0..256); using 0", e);
        }
    }
    if (async_requested && !sw->async_events) {
        hga_swap_log("stream: CUDA event bridge unavailable; using synchronous callbacks");
    } else {
        hga_swap_log("stream: verify callback mode=%s (HGA_STREAM_ASYNC=0 restores sync)",
                sw->async_events ? "CUDA-event" : "synchronous");
    }
    if (paced_requested && !sw->stream_paced) {
        hga_swap_log("stream: paced VERIFY H2D requires CUDA-event callbacks; disabled");
    }
    if (pf_async_requested && !sw->prefill_stream_async) {
        hga_swap_log("stream: async PREFILL H2D requires CUDA-event callbacks; disabled");
    }
    if (sw->stream_chunk) {
        hga_swap_log("stream: verify H2D submission=%.1f MiB chunks",
                hga_mib(sw->stream_chunk));
    } else {
        hga_swap_log("stream: verify H2D submission=single DMA");
    }
    const int n_layer = (int) sw->model->hparams.n_layer();
    const int n_layers_sz = (int) sw->model->layers.size();
    sw->layer_last = n_layer - 1;
    sw->layer_mtp  = n_layer;
    auto layer_ok = [&](int il) { return il >= 0 && il < n_layers_sz; };

    const char * uniform_e = std::getenv("HGA_STREAM_UNIFORM");
    if (!uniform_e) {
        uniform_e = std::getenv("HGA_STREAM_24_56");
    }
    const bool uniform = uniform_e && uniform_e[0] && uniform_e[0] != '0';
    const char * two_e = std::getenv("HGA_STREAM_2");
    const bool two = two_e && two_e[0] && two_e[0] != '0';
    const char * spec_e = std::getenv("HGA_SPEC");
    const bool spec_mtp = spec_e && spec_e[0] && spec_e[0] != '0';

    auto fill_group = [&](int pi, int a_first, int a_last, int b_first, int b_last) {
        auto & p = sw->pair[pi];
        p.layer_a = a_last;
        p.layer_b = b_last;
        std::snprintf(p.tag, sizeof(p.tag), "%d:%d-%d:%d",
                a_first, a_last, b_first, b_last);
        {
            std::unordered_set<const void *> seen;
            for (int il = a_first; il <= a_last; ++il) {
                if (layer_ok(il)) {
                    hga_collect_layer(sw->model->layers[il], p.a, seen, false);
                }
            }
        }
        {
            std::unordered_set<const void *> seen;
            for (int il = b_first; il <= b_last; ++il) {
                if (layer_ok(il)) {
                    hga_collect_layer(sw->model->layers[il], p.b, seen, false);
                }
            }
        }
        p.wait_a = p.wait_a_pf = a_first;
        p.wait_b = p.wait_b_pf = b_first;
    };
    auto fill_pair = [&](int pi, int la, int lb) {
        fill_group(pi, la, la, lb, lb);
        std::snprintf(sw->pair[pi].tag, sizeof(sw->pair[pi].tag), "%d-%d", la, lb);
    };

    int block_n = 0;
    if (const char * e = std::getenv("HGA_STREAM_BLOCK")) {
        char * end = nullptr;
        const long value = std::strtol(e, &end, 10);
        if (end != e && *end == '\0' && (value == 0 || value == 2 || value == 3)) {
            block_n = (int) value;
        } else {
            hga_swap_log("invalid HGA_STREAM_BLOCK=%s (expected 0, 2, or 3); using 0", e);
        }
    }

    if (block_n) {
        /* Coarse exchange experiment: one slot contains adjacent A/B blocks,
         * so VERIFY issues one upload and one restore for block_n layer pairs.
         * The remaining groups preserve a total of 16 host-mapped layers, the
         * same model-memory budget as the default eight one-layer pairs. */
        const int n_left = 8 - block_n;
        sw->n_pairs = 1 + n_left;
        fill_group(0, 1, block_n, 32, 31 + block_n);
        sw->pair[0].verify_pair = true;
        for (int j = 0; j < n_left; ++j) {
            const int i = j + 1;
            fill_pair(j + 1, i * 4, i * 4 + 32);
            sw->pair[j + 1].verify_pair = false;
        }
        hga_swap_log("stream: packed VERIFY block 1..%d↔32..%d in one slot; "
                "PREFILL groups=%d, leftover groups=%d",
                block_n, 31 + block_n, sw->n_pairs, n_left);
    } else if (uniform) {
        sw->n_pairs = 3;
        fill_pair(0, 10, 42);
        fill_pair(1, 21, 53);
        int lb = spec_mtp ? sw->layer_last : sw->layer_mtp;
        if (!spec_mtp && !layer_ok(sw->layer_mtp)) {
            lb = sw->layer_last;
        }
        fill_pair(2, 32, lb);
    } else if (two) {
        sw->n_pairs = 2;
        fill_pair(0, 16, 48);
        int lb = spec_mtp ? sw->layer_last : sw->layer_mtp;
        if (!layer_ok(lb) || (spec_mtp && lb == sw->layer_mtp)) {
            lb = sw->layer_last;
        }
        fill_pair(1, 32, lb);
    } else {
        int verify_streams = 2;
        if (const char * e = std::getenv("HGA_VERIFY_STREAMS")) {
            char * end = nullptr;
            const long value = std::strtol(e, &end, 10);
            if (end != e && *end == '\0' && (value == 2 || value == 3)) {
                verify_streams = (int) value;
            } else {
                hga_swap_log("invalid HGA_VERIFY_STREAMS=%s (expected 2 or 3); using 2", e);
            }
        }
        /* 8 pairs, step 4, offset 32: 0↔32 … 28↔60. MTP stays CUDA. */
        sw->n_pairs = 8;
        for (int i = 0; i < 8; ++i) {
            fill_pair(i, i * 4, i * 4 + 32);
            /* Keep two complete layer-pair streams, but use the larger
             * 24↔56 pair in place of 28↔60. This leaves about 35 MiB more
             * VERIFY headroom without a partial-layer third stream. */
            sw->pair[i].verify_pair = (i == 4 || i == 6 || (verify_streams == 3 && i == 3));
        }
        hga_swap_log("stream: 8 PREFILL pairs step-4  0↔32 4↔36 … 28↔60; "
                "VERIFY streams=%d (%s)", verify_streams,
                verify_streams == 2 ? "16-48 / 24-56" : "12-44 / 16-48 / 24-56");
    }

    int n_verify_pairs = 0;
    for (int pi = 0; pi < sw->n_pairs; ++pi) {
        n_verify_pairs += sw->pair[pi].verify_pair ? 1 : 0;
    }
    const bool paced_common = sw->n_pairs == 8 &&
            sw->pair[4].verify_pair && sw->pair[4].layer_a == 16 &&
            sw->pair[4].layer_b == 48 && sw->pair[6].verify_pair &&
            sw->pair[6].layer_a == 24 && sw->pair[6].layer_b == 56;
    const bool paced_layout = paced_common &&
            (n_verify_pairs == 2 ||
             (n_verify_pairs == 3 && sw->pair[3].verify_pair &&
              sw->pair[3].layer_a == 12 && sw->pair[3].layer_b == 44));
    if (sw->stream_paced && sw->stream_chunk) {
        hga_swap_log("stream: paced VERIFY H2D disabled by explicit chunk-size diagnostic");
        sw->stream_paced = false;
    } else if (sw->stream_paced && !paced_layout) {
        hga_swap_log("stream: paced VERIFY H2D requires the default two/three-pair layout; disabled");
        sw->stream_paced = false;
    } else if (sw->stream_paced) {
        sw->stream_paced_pairs = n_verify_pairs;
        hga_swap_log("stream: paced VERIFY H2D=%d pairs, one segment per post-HGA window",
                n_verify_pairs);
    }
    bool paced_prefill_layout = sw->n_pairs == 8;
    for (int pi = 0; paced_prefill_layout && pi < sw->n_pairs; ++pi) {
        paced_prefill_layout = sw->pair[pi].layer_a == pi * 4 &&
                sw->pair[pi].layer_b == pi * 4 + 32;
    }
    if (sw->prefill_stream_paced && !paced_prefill_layout) {
        hga_swap_log("stream: paced PREFILL H2D requires eight step-4 pairs; disabled");
        sw->prefill_stream_paced = false;
    } else if (sw->prefill_stream_paced) {
        hga_swap_log("stream: paced PREFILL H2D=8 pairs, one full image per post-HGA window");
    } else if (sw->prefill_stream_async) {
        hga_swap_log("stream: async PREFILL H2D=inline CUDA events, immediate whole image");
    }

    bool missing = false;
    for (int pi = 0; pi < sw->n_pairs; ++pi) {
        if (sw->pair[pi].a.empty() || sw->pair[pi].b.empty()) {
            missing = true;
            break;
        }
    }
    if (missing) {
        hga_swap_log("stream: missing pair tensors n_pairs=%d — need -ot CPU on exchange layers",
                sw->n_pairs);
        for (int pi = 0; pi < sw->n_pairs; ++pi) {
            hga_swap_log("  pair %s  A=%zu B=%zu", sw->pair[pi].tag,
                    sw->pair[pi].a.size(), sw->pair[pi].b.size());
        }
        return false;
    }

    for (int pi = 0; pi < sw->n_pairs; ++pi) {
        auto & p = sw->pair[pi];
        p.bytes_a = hga_stream_layout(sw, p.a, p.off_a);
        p.bytes_b = hga_stream_layout(sw, p.b, p.off_b);
        p.bytes_a_pf = hga_layout_from(sw, p.extra_a, p.off_ea, p.bytes_a);
        p.bytes_b_pf = hga_layout_from(sw, p.extra_b, p.off_eb, p.bytes_b);
        if (p.extra_a.empty()) {
            p.bytes_a_pf = p.bytes_a;
        }
        if (p.extra_b.empty()) {
            p.bytes_b_pf = p.bytes_b;
        }
    }

    int prefill_streams = sw->n_pairs;
    if (const char * e = std::getenv("HGA_PREFILL_STREAMS")) {
        char * end = nullptr;
        const long value = std::strtol(e, &end, 10);
        if (end != e && *end == '\0' && value >= 1 && value <= sw->n_pairs) {
            prefill_streams = (int) value;
        } else {
            hga_swap_log("invalid HGA_PREFILL_STREAMS=%s; using %d", e, sw->n_pairs);
        }
    }
    int make_resident = sw->n_pairs - prefill_streams;
    std::vector<int> resident_candidates;
    for (int pi = 0; pi < sw->n_pairs; ++pi) {
        if (!sw->pair[pi].verify_pair) {
            resident_candidates.push_back(pi);
        }
    }
    std::sort(resident_candidates.begin(), resident_candidates.end(), [&](int a, int b) {
        const auto & pa = sw->pair[a];
        const auto & pb = sw->pair[b];
        return pa.bytes_a_pf + pa.bytes_b_pf > pb.bytes_a_pf + pb.bytes_b_pf;
    });
    make_resident = std::min(make_resident, (int) resident_candidates.size());
    for (int i = 0; i < make_resident; ++i) {
        sw->pair[resident_candidates[i]].prefill_resident = true;
    }
    hga_swap_log("stream: PREFILL exchange=%d resident-pairs=%d", sw->n_pairs - make_resident, make_resident);

    for (int pi = 0; pi < sw->n_pairs; ++pi) {
        auto & p = sw->pair[pi];
        if (p.prefill_resident) {
            hga_swap_log("stream %s PREFILL resident  A+B=%.1f MiB (removes %.1f MiB/cycle H2D)",
                    p.tag, hga_mib(p.bytes_a_pf + p.bytes_b_pf),
                    hga_mib(p.bytes_a_pf + p.bytes_b_pf));
            continue;
        }
        char ta[16], tb[16];
        std::snprintf(ta, sizeof(ta), "%s-A", p.tag);
        std::snprintf(tb, sizeof(tb), "%s-B", p.tag);
        if (!hga_cu_pin(&p.pin_a, p.bytes_a, ta) || !hga_cu_pin(&p.pin_b, p.bytes_b, tb)) {
            return false;
        }
        hga_stream_pack_pin(p.pin_a, p.a, p.off_a, p.bytes_a);
        hga_stream_pack_pin(p.pin_b, p.b, p.off_b, p.bytes_b);
        if (p.bytes_a_pf > p.bytes_a) {
            char t[16];
            std::snprintf(t, sizeof(t), "%s-Apf", p.tag);
            if (!hga_cu_pin(&p.pin_a_pf, p.bytes_a_pf, t)) {
                return false;
            }
            std::memset(p.pin_a_pf, 0, p.bytes_a_pf);
            std::memcpy(p.pin_a_pf, p.pin_a, p.bytes_a);
            for (size_t i = 0; i < p.extra_a.size(); ++i) {
                std::memcpy((uint8_t *) p.pin_a_pf + p.off_ea[i],
                        p.extra_a[i].host_data, ggml_nbytes(p.extra_a[i].t));
            }
        }
        if (p.bytes_b_pf > p.bytes_b) {
            char t[16];
            std::snprintf(t, sizeof(t), "%s-Bpf", p.tag);
            if (!hga_cu_pin(&p.pin_b_pf, p.bytes_b_pf, t)) {
                return false;
            }
            std::memset(p.pin_b_pf, 0, p.bytes_b_pf);
            std::memcpy(p.pin_b_pf, p.pin_b, p.bytes_b);
            for (size_t i = 0; i < p.extra_b.size(); ++i) {
                std::memcpy((uint8_t *) p.pin_b_pf + p.off_eb[i],
                        p.extra_b[i].host_data, ggml_nbytes(p.extra_b[i].t));
            }
        }
        if (hga_cu.stream_create(&p.copy_stream, HGA_CU_STREAM_NONBLOCKING) != 0 ||
            hga_cu.event_create(&p.ev_done, HGA_CU_EVENT_DISABLE_TIMING) != 0 ||
            (sw->async_events &&
             hga_cu.event_create(&p.ev_compute, HGA_CU_EVENT_DISABLE_TIMING) != 0)) {
            hga_swap_log("stream %s copy stream/event create failed", p.tag);
            return false;
        }
        hga_swap_log("stream %s  A(blk.%d..%d)=%.1f MiB  B(blk.%d..%d)=%.1f MiB  prefill A/B=%.1f/%.1f  independent copy stream",
                p.tag, p.wait_a, p.layer_a, hga_mib(p.bytes_a),
                p.wait_b, p.layer_b, hga_mib(p.bytes_b),
                hga_mib(p.bytes_a_pf), hga_mib(p.bytes_b_pf));
    }
    if (!sw->ffn_res.empty()) {
        hga_stream_layout(sw, sw->ffn_res, sw->ffn_res_off);
    }
    sw->stream_ok = true;
    return true;
}


static void hga_pair_bind_phase(hga_weight_swap::stream_pair & p, bool prefill) {
    hga_stream_bind_group(p.buf, p.a, p.off_a);
    hga_stream_bind_group(p.buf, p.b, p.off_b);
    if (prefill) {
        hga_stream_bind_group(p.buf, p.extra_a, p.off_ea);
        hga_stream_bind_group(p.buf, p.extra_b, p.off_eb);
    } else {
        hga_stream_unbind_group(p.extra_a);
        hga_stream_unbind_group(p.extra_b);
    }
}

static double hga_cuda_free_mib(hga_weight_swap * sw);

static bool hga_pair_pin_prefill_resident(hga_weight_swap * sw, hga_weight_swap::stream_pair & p) {
    std::vector<hga_slot> all = p.a;
    all.insert(all.end(), p.b.begin(), p.b.end());
    all.insert(all.end(), p.extra_a.begin(), p.extra_a.end());
    all.insert(all.end(), p.extra_b.begin(), p.extra_b.end());
    std::vector<size_t> offsets;
    const size_t need = hga_stream_layout(sw, all, offsets);
    hga_pair_clear_buf(p);
    p.buf = hga_alloc_cuda(sw, "prefill-resident", p.tag, need);
    if (!p.buf) {
        hga_swap_log("PREFILL resident pair %s %.2f MiB allocation failed", p.tag, hga_mib(need));
        return false;
    }
    ggml_backend_buffer_set_usage(p.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    hga_stream_bind_group(p.buf, all, offsets);
    for (auto & s : all) {
        ggml_backend_tensor_set(s.t, s.host_data, 0, ggml_nbytes(s.t));
    }
    p.cap = need;
    p.both_resident = true;
    hga_swap_log("PREFILL resident pair %s CUDA %.2f MiB ok  free_after=%.0f MiB",
            p.tag, hga_mib(need), hga_cuda_free_mib(sw));
    return true;
}

static void hga_weight_swap_split_reset(hga_weight_swap * sw);
static void hga_weight_swap_try_split(hga_weight_swap * sw);

static void hga_stream_enter_prefill(hga_weight_swap * sw) {
    if (!sw->stream_ok) {
        return;
    }
    hga_weight_swap_split_reset(sw);
    hga_stream_wait_all(sw);
    hga_stream_unbind_group(sw->ffn_res);
    if (sw->buf_ffn_res) {
        hga_alloc_live_del(sw->buf_ffn_res);
        ggml_backend_buffer_free(sw->buf_ffn_res);
        sw->buf_ffn_res = nullptr;
    }
    sw->leftover_pinned = false;
    for (int pi = 0; pi < sw->n_pairs; ++pi) {
        auto & p = sw->pair[pi];
        if (p.prefill_resident) {
            if (!p.both_resident && !hga_pair_pin_prefill_resident(sw, p)) {
                sw->stream_ok = false;
                return;
            }
            continue;
        }
        // A persistent server can return from VERIFY to PREFILL many times.
        // VERIFY pins both sides of non-stream pairs; retaining those buffers
        // while re-creating eight PREFILL slots exhausts the V100 before the
        // new prompt graph can reserve its working buffer.  The host-mmap
        // source remains authoritative, so release only the VERIFY-resident
        // copies before binding the normal streaming layout again.
        if (p.both_resident) {
            hga_swap_log("PREFILL release leftover VERIFY pair %s", p.tag);
            hga_pair_clear_buf(p);
        }
        const size_t need = std::max(p.bytes_a_pf, p.bytes_b_pf);
        if (!hga_pair_ensure(sw, p, need, false)) {
            sw->stream_ok = false;
            return;
        }
        hga_pair_bind_phase(p, true);
        hga_pair_kick(sw, pi, 1);
    }
    hga_stream_wait_all(sw);
    {
        char when[96];
        int n_exchange = 0;
        for (int pi = 0; pi < sw->n_pairs; ++pi) {
            n_exchange += sw->pair[pi].prefill_resident ? 0 : 1;
        }
        std::snprintf(when, sizeof(when), "after stream PREFILL (%d exchange pairs)", n_exchange);
        hga_log_cuda_mem(sw->gpu, when);
    }
}

static double hga_cuda_free_mib(hga_weight_swap * sw) {
    return sw ? hga_gpu_free_mib(sw->gpu) : 0;
}

/* VERIFY: pin leftover (non-stream) pairs one-by-one, A and B both resident.
 * Called AFTER dropping the prefill ggml graph so the 512-token compute buffer
 * is gone. Logs PIN-STEP n/N and the first cudaMalloc failure. */
static void hga_stream_pin_leftover_verify(hga_weight_swap * sw) {
    if (!sw || !sw->stream_ok || sw->leftover_pinned) {
        return;
    }
    sw->leftover_pinned = true;
    const char * skip = std::getenv("HGA_NO_RESIDENT_FFN");
    if (skip && skip[0] && skip[0] != '0') {
        hga_swap_log("PIN-STEP skip leftover pairs (HGA_NO_RESIDENT_FFN=1)");
        return;
    }
    int n_left = 0;
    for (int pi = 0; pi < sw->n_pairs; ++pi) {
        if (!sw->pair[pi].verify_pair) {
            n_left++;
        }
    }
    if (n_left == 0) {
        return;
    }
    hga_swap_log("PIN-STEP start  leftover=%d  stream=%d  free=%.0f MiB",
            n_left, sw->n_pairs - n_left, hga_cuda_free_mib(sw));
    int step = 0;
    int n_ok = 0;
    int n_fallback = 0;
    int oom_step = 0;
    char oom_tag[12] = {};
    for (int pi = 0; pi < sw->n_pairs; ++pi) {
        auto & p = sw->pair[pi];
        if (p.verify_pair) {
            continue;
        }
        step++;
        if (p.both_resident) {
            n_ok++;
            hga_swap_log("PIN-STEP %d/%d reuse PREFILL-resident pair %s  cap=%.2f MiB",
                    step, n_left, p.tag, hga_mib(p.cap));
            continue;
        }
        std::vector<hga_slot> both = p.a;
        both.insert(both.end(), p.b.begin(), p.b.end());
        if (both.empty()) {
            hga_swap_log("PIN-STEP %d/%d leftover %s  empty, skip", step, n_left, p.tag);
            continue;
        }
        std::vector<size_t> off;
        const size_t need = hga_stream_layout(sw, both, off);
        const double free0 = hga_cuda_free_mib(sw);
        hga_swap_log("PIN-STEP %d/%d leftover pair %s  need=%.2f MiB  free=%.0f MiB",
                step, n_left, p.tag, hga_mib(need), free0);
        hga_pair_clear_buf(p);
        p.buf = hga_alloc_cuda(sw, "leftover-pin", p.tag, need);
        if (!p.buf) {
            hga_swap_log("OOM at PIN-STEP %d/%d leftover pair %s  need=%.2f MiB  free=%.0f MiB  "
                    "(cudaMalloc failed — trying a stream-slot fallback)",
                    step, n_left, p.tag, hga_mib(need), free0);
            hga_log_cuda_mem(sw->gpu, "after leftover pin FAIL");
            const size_t stream_need = std::max(p.bytes_a, p.bytes_b);
            if (hga_pair_ensure(sw, p, stream_need, true)) {
                p.verify_pair = true;
                hga_pair_bind_phase(p, false);
                hga_pair_kick(sw, pi, 1);
                n_fallback++;
                hga_swap_log("PIN-STEP %d/%d pair %s fallback: VERIFY stream slot %.2f MiB",
                        step, n_left, p.tag, hga_mib(stream_need));
            } else {
                if (oom_step == 0) {
                    oom_step = step;
                    std::snprintf(oom_tag, sizeof(oom_tag), "%s", p.tag);
                }
            }
            continue;
        }
        ggml_backend_buffer_set_usage(p.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        hga_stream_bind_group(p.buf, both, off);
        for (auto & s : both) {
            ggml_backend_tensor_set(s.t, s.host_data, 0, ggml_nbytes(s.t));
        }
        p.cap = need;
        p.occ = 0;
        p.copy = 0;
        p.both_resident = true;
        n_ok++;
        hga_swap_log("PIN-STEP %d/%d leftover pair %s  CUDA alloc %.2f MiB ok  free_after=%.0f MiB",
                step, n_left, p.tag, hga_mib(need), hga_cuda_free_mib(sw));
    }
    if (oom_step > 0) {
        hga_swap_log("PIN-STEP summary  pinned %d/%d leftover pairs; stream fallback=%d; "
                "unrecovered OOM at step %d pair %s",
                n_ok, n_left, n_fallback, oom_step, oom_tag);
    } else {
        hga_swap_log("PIN-STEP summary  pinned %d/%d leftover pairs; stream fallback=%d; no unrecovered OOM",
                n_ok, n_left, n_fallback);
    }
    hga_log_cuda_mem(sw->gpu, "after leftover VERIFY pins");
    if (n_ok + n_fallback < n_left) {
        hga_swap_log("FATAL leftover VERIFY pin incomplete %d/%d — "
                "decode would keep host weights on non-stream pairs",
                n_ok + n_fallback, n_left);
        hga_alloc_dump_oom("leftover-pin-incomplete");
        const char * abort_e = std::getenv("HGA_PIN_ABORT");
        if (abort_e && abort_e[0] && abort_e[0] != '0') {
            abort();
        }
    }
}

static bool hga_tensor_is_tiled_ffn(ggml_tensor * t, const llama_layer & layer) {
    return t && (t == layer.ffn_up || t == layer.ffn_gate || t == layer.ffn_down);
}

static void hga_split_fill_src(hga_weight_swap * sw, int pi, bool side_b,
                               hga_split_ffn_layer_src & out) {
    auto & p = sw->pair[pi];
    const int il = side_b ? p.layer_b : p.layer_a;
    const auto & slots = side_b ? p.b : p.a;
    const llama_layer & layer = sw->model->layers[il];
    out = {};
    out.layer_id = il;
    out.up = layer.ffn_up;
    out.gate = layer.ffn_gate;
    out.down = layer.ffn_down;
    out.up_s = layer.ffn_up_s;
    out.gate_s = layer.ffn_gate_s;
    out.down_s = layer.ffn_down_s;
    for (const auto & s : slots) {
        if (!s.t) {
            continue;
        }
        if (s.t == layer.ffn_up) {
            out.host_up = s.host_data;
        } else if (s.t == layer.ffn_gate) {
            out.host_gate = s.host_data;
        } else if (s.t == layer.ffn_down) {
            out.host_down = s.host_data;
        }
        if (hga_tensor_is_tiled_ffn(s.t, layer)) {
            continue;
        }
        if (out.n_core >= HGA_SPLIT_FFN_MAX_CORE) {
            continue;
        }
        out.core[out.n_core] = s.t;
        out.core_host[out.n_core] = s.host_data;
        out.n_core++;
    }
}

static void hga_weight_swap_split_reset(hga_weight_swap * sw) {
    if (!sw) {
        return;
    }
    if (sw->split) {
        hga_split_ffn_free(sw->split);
        sw->split = nullptr;
    }
    for (int pi = 0; pi < sw->n_pairs; ++pi) {
        sw->pair[pi].split_ffn = false;
    }
}

/* After leftover VERIFY pin: try to replace whole-layer VERIFY images with
 * resident cores + a dynamic FFN tile bank. Failure keeps the current schedule. */
static void hga_weight_swap_try_split(hga_weight_swap * sw) {
    if (!sw || !sw->stream_ok || sw->split) {
        return;
    }
    if (!hga_split_ffn_env_enabled()) {
        return;
    }
    if (!sw->async_events) {
        hga_swap_log("hga-split: fallback pair=? layer=-1 reason=CUDA event bridge unavailable");
        return;
    }
    hga_split_ffn_pair_src src[HGA_SPLIT_FFN_MAX_PAIRS];
    int n = 0;
    int idx[HGA_SPLIT_FFN_MAX_PAIRS];
    for (int pi = 0; pi < sw->n_pairs && n < HGA_SPLIT_FFN_MAX_PAIRS; ++pi) {
        auto & p = sw->pair[pi];
        if (!p.verify_pair || p.both_resident || !p.buf) {
            continue;
        }
        if (p.copy_stream && hga_cu.stream_sync) {
            hga_cu.stream_sync(p.copy_stream);
            p.occ = p.copy ? p.copy : p.occ;
            p.copy = 0;
        }
        /* Unbind the whole-layer image so FFN packing reads mmap bytes and
         * cores can be re-laid into the same pair buffer. */
        hga_stream_unbind_group(p.a);
        hga_stream_unbind_group(p.b);
        auto & s = src[n];
        std::snprintf(s.tag, sizeof(s.tag), "%s", p.tag);
        s.buf = p.buf;
        s.cap = p.cap;
        hga_split_fill_src(sw, pi, false, s.a);
        hga_split_fill_src(sw, pi, true,  s.b);
        idx[n] = pi;
        n++;
    }
    if (n == 0) {
        hga_swap_log("hga-split: fallback pair=? layer=-1 reason=no VERIFY stream pairs");
        return;
    }
    sw->split = hga_split_ffn_create(sw->gpu, sw->buft, src, n,
            sw->stream_timing, sw->async_events);
    if (!sw->split) {
        /* Restore whole-layer occupancy so VERIFY can proceed. */
        for (int i = 0; i < n; ++i) {
            auto & p = sw->pair[idx[i]];
            hga_pair_bind_phase(p, false);
            hga_pair_kick(sw, idx[i], 1);
        }
        return;
    }
    for (int i = 0; i < n; ++i) {
        sw->pair[idx[i]].split_ffn = true;
        sw->pair[idx[i]].occ = 0;
        sw->pair[idx[i]].copy = 0;
    }
}

static void hga_stream_enter_decode(hga_weight_swap * sw) {
    if (!sw->stream_ok) {
        return;
    }
    hga_stream_wait_all(sw);
    /* Prefill uses all 8 slots. VERIFY keeps 2 or 3 streaming pairs. The other
     * pairs are pinned A+B CUDA-resident after the prefill graph is dropped
     * (hga_stream_pin_leftover_verify). Pinning them here OOMs: the 512-token
     * compute buffer is still alive. */
    sw->ffn_res.clear();
    sw->ffn_res_off.clear();
    if (sw->buf_ffn_res) {
        hga_alloc_live_del(sw->buf_ffn_res);
        ggml_backend_buffer_free(sw->buf_ffn_res);
        sw->buf_ffn_res = nullptr;
    }
    sw->leftover_pinned = false;
    int n_verify = 0;
    int n_left = 0;
    for (int pi = 0; pi < sw->n_pairs; ++pi) {
        auto & p = sw->pair[pi];
        if (!p.verify_pair) {
            n_left++;
            if (p.both_resident) {
                hga_swap_log("PIN-STEP keep PREFILL-resident pair %s for VERIFY  cap=%.1f MiB",
                        p.tag, hga_mib(p.cap));
                continue;
            }
            hga_swap_log("PIN-STEP free stream slot %s  cap=%.1f MiB  (leftover, pin later)",
                    p.tag, hga_mib(p.cap));
            hga_pair_clear_buf(p);
            continue;
        }
        n_verify++;
        const size_t need = std::max(p.bytes_a, p.bytes_b);
        if (!hga_pair_ensure(sw, p, need, true)) {
            sw->stream_ok = false;
            return;
        }
        hga_pair_bind_phase(p, false);
        hga_pair_kick(sw, pi, 1);
    }
    hga_swap_log("VERIFY stream %d/%d pairs; %d non-stream pairs resident/deferred  free=%.0f MiB",
            n_verify, sw->n_pairs, n_left, hga_cuda_free_mib(sw));
    hga_log_cuda_mem(sw->gpu, "after stream DECODE (leftover not pinned yet)");
}

static void hga_stream_begin_ubatch(hga_weight_swap * sw) {
    if (!sw->stream_ok) {
        return;
    }
    if (sw->phase != HGA_SWAP_DECODE && sw->phase != HGA_SWAP_PREFILL &&
            sw->phase != HGA_SWAP_VERIFY) {
        return;
    }
    for (int pi = 0; pi < sw->n_pairs; ++pi) {
        auto & p = sw->pair[pi];
        if (sw->phase == HGA_SWAP_PREFILL && p.prefill_resident) {
            continue;
        }
        if (hga_decode_pack((int32_t) sw->phase) && !p.verify_pair) {
            continue;
        }
        if (p.split_ffn) {
            continue;
        }
        if (p.occ != 1 && p.copy != 1) {
            hga_pair_kick(sw, pi, 1);
        }
    }
    if (hga_decode_pack((int32_t) sw->phase) && sw->split) {
        hga_split_ffn_begin_ubatch(sw->split);
    }
}

static void hga_stream_bind_eval(llama_cparams & cparams, hga_weight_swap * sw,
                                 ggml_backend_sched_t sched) {
    if (!sw || !sw->stream_ok) {
        return;
    }
    if (cparams.cb_eval != hga_stream_eval_cb) {
        sw->prev_eval    = cparams.cb_eval;
        sw->prev_eval_ud = cparams.cb_eval_user_data;
        cparams.cb_eval = hga_stream_eval_cb;
        cparams.cb_eval_user_data = sw;
    }
    if (sched) {
        ggml_backend_sched_set_eval_callback(sched, hga_stream_eval_cb, sw);
        const bool wanted = sw->async_events &&
                (hga_decode_pack((int32_t) sw->phase) ||
                 (sw->phase == HGA_SWAP_PREFILL && sw->prefill_stream_async)) &&
                !std::getenv("HGA_PROF_OPS");
        const bool callback_ok = g_hga_cuda_set_node_callback &&
                g_hga_cuda_set_node_callback(sw->gpu,
                        wanted ? hga_stream_cuda_node_cb : nullptr,
                        wanted ? sw : nullptr);
        const bool async = wanted && callback_ok;
        if (wanted && !callback_ok) {
            hga_swap_log("stream: failed to install inline CUDA callback; using scheduler fallback");
        }
        ggml_backend_sched_set_eval_callback_async(sched, async);
    }
}

static void hga_pin_census(hga_weight_swap * sw, const char * when);

hga_weight_swap * hga_weight_swap_init(const llama_model * model, ggml_backend_t gpu) {
    if (!model || !gpu) {
        return nullptr;
    }

    auto * sw = new hga_weight_swap();
    sw->model = model;
    sw->gpu   = gpu;
    sw->buft  = ggml_backend_get_default_buffer_type(gpu);
    const char * pin_all = std::getenv("HGA_PREFILL_PIN_ALL");
    if (pin_all && pin_all[0] && pin_all[0] != '0') {
        hga_swap_log("PREFILL PIN PROBE enabled: every named Qwen node forced to CUDA; "
                "exchange slots still replace A/B in place");
    }
    if (!sw->buft) {
        hga_swap_log("GPU has no default buffer type");
        delete sw;
        return nullptr;
    }

    const int n_layer = (int) model->hparams.n_layer();
    sw->layer_last = n_layer - 1;
    sw->layer_mtp  = n_layer;
    /* Non-exchange dense QKV stay CUDA-resident from load. Only stream pairs
     * are host mmap. Do not stage Q/K/V — that was the extra migrate. */
    if (model->output) {
        hga_slot so;
        if (hga_slot_from_tensor(model->output, so, "output.weight")) {
            sw->output.push_back(so);
        } else {
            hga_swap_log("output.weight already device-resident");
        }
    }

    if (!hga_stream_setup(sw)) {
        hga_swap_log("stream disabled — exchange H2D off; resident weights stay where they loaded");
    }

    /* Pin lm_head once. Never cuda_to_host it during PREFILL/DECODE/VERIFY. */
    if (!sw->output.empty()) {
        if (!hga_host_to_cuda(sw, sw->output, &sw->buf_out, "output.weight")) {
            hga_swap_log("could not pin output.weight on CUDA — census will PUSHED");
        } else {
            hga_swap_log("lm_head pinned CUDA  %.1f MiB (resident until context free)",
                    hga_mib(hga_nbytes_sum(sw->output)));
            hga_log_tensor("init-out", sw->output.front().t);
        }
    }

    hga_swap_log("staging lists Q=%zu K/V=%zu out=%zu (0 QKV = load-pinned CUDA)  buft=%s",
            sw->q.size(), sw->kv.size(), sw->output.size(),
            ggml_backend_buft_name(sw->buft));
    hga_pin_census(sw, "at init");
    return sw;
}

void hga_weight_swap_free(hga_weight_swap * sw) {
    if (!sw) {
        return;
    }
    if (g_hga_cuda_set_node_callback && sw->gpu) {
        g_hga_cuda_set_node_callback(sw->gpu, nullptr, nullptr);
    }
    hga_weight_swap_split_reset(sw);
    hga_weight_swap_set_phase(sw, HGA_SWAP_NONE);
    hga_stream_free_buffers(sw);
    delete sw;
}

hga_swap_phase hga_weight_swap_phase(const hga_weight_swap * sw) {
    return sw ? sw->phase : HGA_SWAP_NONE;
}

hga_split_ffn * hga_weight_swap_split(hga_weight_swap * sw) {
    return sw ? sw->split : nullptr;
}

bool hga_weight_swap_split_layer(hga_weight_swap * sw, int layer_id) {
    return sw && hga_split_ffn_layer_active(sw->split, layer_id);
}

bool hga_lmhead_on_host(const llama_cparams & cparams) {
    (void) cparams;
    const char * e = std::getenv("HGA_LMHEAD_CPU");
    return e && e[0] && e[0] != '0';
}

bool hga_weight_swap_set_phase(hga_weight_swap * sw, hga_swap_phase phase, bool stage_output) {
    if (!sw) {
        return false;
    }
    if (sw->phase == phase) {
        return true;
    }
    /* VERIFY reuses DECODE-resident weights and the DECODE graph. Do not restage. */
    if (hga_decode_pack((int32_t) sw->phase) && hga_decode_pack((int32_t) phase)) {
        sw->phase = phase;
        hga_swap_log("%s     weights unchanged (decode packing)",
                phase == HGA_SWAP_VERIFY ? "VERIFY" : "DECODE");
        static bool census_once = false;
        if (!census_once) {
            census_once = true;
            hga_pin_census(sw, "VERIFY (decode packing, weights unchanged)");
        }
        return true;
    }

    const int64_t t0 = ggml_time_us();
    ggml_backend_synchronize(sw->gpu);

    if (phase == HGA_SWAP_NONE) {
        hga_weight_swap_split_reset(sw);
        hga_stream_wait_all(sw);
        for (int pi = 0; pi < sw->n_pairs; ++pi) {
            hga_pair_clear_buf(sw->pair[pi]);
        }
        hga_stream_unbind_group(sw->ffn_res);
        if (sw->buf_ffn_res) {
            hga_alloc_live_del(sw->buf_ffn_res);
            ggml_backend_buffer_free(sw->buf_ffn_res);
            sw->buf_ffn_res = nullptr;
        }
        hga_cuda_to_host(sw->kv,  &sw->buf_kv);
        hga_cuda_to_host(sw->output, &sw->buf_out);
        hga_cuda_to_host(sw->q,   &sw->buf_q);
        sw->phase = HGA_SWAP_NONE;
        hga_swap_log("NONE     all staged tensors → host mmap  %.1f ms",
                (ggml_time_us() - t0) / 1000.0);
        hga_pin_census(sw, "after NONE");
        return true;
    }

    if (phase == HGA_SWAP_PREFILL) {
        /* lm_head stays CUDA. Stream only the exchange-layer pairs. */
        if (!hga_host_to_cuda(sw, sw->q, &sw->buf_q, "Q") ||
            !hga_host_to_cuda(sw, sw->kv, &sw->buf_kv, "K/V")) {
            hga_cuda_to_host(sw->kv, &sw->buf_kv);
            hga_cuda_to_host(sw->q,  &sw->buf_q);
            sw->phase = HGA_SWAP_NONE;
            hga_pin_census(sw, "PREFILL H2D failed");
            return false;
        }

        sw->phase = HGA_SWAP_PREFILL;
        hga_stream_enter_prefill(sw);
        hga_swap_log("PREFILL  exchange %d pairs  lm_head CUDA  in %.1f ms",
                sw->n_pairs, (ggml_time_us() - t0) / 1000.0);
        if (!sw->output.empty()) {
            hga_log_tensor("out", sw->output.front().t);
        }
        hga_log_cuda_mem(sw->gpu, "after PREFILL");
        hga_pin_census(sw, "after PREFILL");
        return true;
    }

    /* DECODE/VERIFY: resident weights already CUDA. Only exchange occupancy. */
    if (!hga_host_to_cuda(sw, sw->q, &sw->buf_q, "Q") ||
        !hga_host_to_cuda(sw, sw->kv, &sw->buf_kv, "K/V")) {
        sw->phase = HGA_SWAP_NONE;
        hga_pin_census(sw, "DECODE H2D failed");
        return false;
    }

    sw->phase = (phase == HGA_SWAP_VERIFY) ? HGA_SWAP_VERIFY : HGA_SWAP_DECODE;
    hga_stream_enter_decode(sw);

    if (!sw->output.empty() && !sw->buf_out) {
        if (!hga_host_to_cuda(sw, sw->output, &sw->buf_out, "output.weight")) {
            hga_swap_log("DECODE could not pin output.weight — census will PUSHED");
        }
    }
    (void) stage_output;

    hga_swap_log("%s   exchange slots + resident QKV/lm_head CUDA  in %.1f ms",
            sw->phase == HGA_SWAP_VERIFY ? "VERIFY" : "DECODE",
            (ggml_time_us() - t0) / 1000.0);
    if (!sw->output.empty()) {
        hga_log_tensor("out", sw->output.front().t);
        if (sw->output.front().t->buffer &&
                ggml_backend_buffer_is_host(sw->output.front().t->buffer)) {
            hga_swap_log("output.weight still on host — should be CUDA-resident");
        }
    }
    hga_log_cuda_mem(sw->gpu, "after DECODE");
    hga_pin_census(sw, "after DECODE");
    return true;
}

/* Exclusive per-op graph profiler (HGA_PROF_OPS=1). Synchronizes so GPU
 * kernel time is not just the launch. It is normally used for decode/verify;
 * HGA_PROF_PREFILL=1 additionally hooks and dumps every prefill graph.
 * Synchronization inflates wall time, so use a separate unprofiled run for
 * real throughput. */
struct hga_ops_acc {
    double ms = 0;
    int n = 0;
    size_t bytes = 0;
};

static const char * hga_buf_kind(const ggml_tensor * t) {
    if (!t || !t->buffer) return "none";
    return ggml_backend_buffer_is_host(t->buffer) ? "CPU" : "GPU";
}

static std::string hga_weight_class(const char * name) {
    if (!name || !name[0]) return "other";
    if (std::strstr(name, "attn_qkv")) return "GDN_qkv";
    if (std::strstr(name, "attn_q.weight") || std::strstr(name, "attn_q ")) return "Q";
    if (std::strstr(name, "attn_k.weight") || std::strstr(name, "attn_k ")) return "K";
    if (std::strstr(name, "attn_v.weight") || std::strstr(name, "attn_v ")) return "V";
    if (std::strstr(name, "attn_output") || std::strstr(name, "attn_out") ||
        std::strstr(name, "attn_gate")) return "o_proj/gate";
    if (std::strstr(name, "ffn_")) return "FFN";
    if (std::strstr(name, "output.weight") || std::strcmp(name, "output") == 0) return "lm_head";
    if (std::strstr(name, "ssm_") || std::strstr(name, "linear_attn")) return "GDN";
    if (std::strstr(name, "token_embd")) return "embed";
    return name;
}

static std::string hga_ops_bucket(const ggml_tensor * t) {
    const char * op = ggml_op_name(t->op);
    const char * dst = hga_buf_kind(t);
    if (t->op == GGML_OP_CPY || t->op == GGML_OP_DUP || t->op == GGML_OP_CONT) {
        const char * src = hga_buf_kind(t->src[0]);
        return std::string("copy ") + src + "->" + dst;
    }
    if (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) {
        const char * w = (t->src[0] && t->src[0]->name[0]) ? t->src[0]->name : "?";
        return std::string(dst) + " MUL_MAT " + hga_weight_class(w);
    }
    if (t->name[0] && (std::strstr(t->name, "hga") || std::strstr(t->name, "HGA"))) {
        return std::string(dst) + " " + t->name;
    }
    return std::string(dst) + " " + op;
}

/* Coarse modules for decode-vs-verify compare.
 * Copies are classified by CPY first — tensors named hga_Q_cpu are D2H, not HGA math. */
static std::string hga_ops_module(const ggml_tensor * t) {
    const char * nm = t->name[0] ? t->name : "";
    const char * op = ggml_op_name(t->op);
    if (std::strstr(nm, "hga_gpu_stage_h2d_united") &&
        (t->op == GGML_OP_CPY || t->op == GGML_OP_DUP ||
         t->op == GGML_OP_CONT)) {
        return "HGA_prefill_KV_H2D";
    }
    if (std::strstr(nm, "hga_gpu_stage_h2d_united")) {
        return "HGA_prefill_KV_views";
    }
    if (std::strstr(nm, "hga_gpu_stage_cpu_united")) {
        return "HGA_prefill_CPU_stage";
    }
    if (std::strstr(nm, "hga_prefill_") || std::strstr(nm, "hga_mask_")) {
        return nm;
    }
    if (t->op == GGML_OP_CPY || t->op == GGML_OP_DUP || t->op == GGML_OP_CONT) {
        const bool sh = t->src[0] && t->src[0]->buffer && ggml_backend_buffer_is_host(t->src[0]->buffer);
        const bool dh = t->buffer && ggml_backend_buffer_is_host(t->buffer);
        const bool hga_act = std::strstr(nm, "hga_Q") || std::strstr(nm, "hga_K") ||
                             std::strstr(nm, "hga_V") || std::strstr(nm, "hga_Kraw");
        const bool hga_out = std::strstr(nm, "hga_attn");
        if (hga_act && !sh && dh) return "D2H_QKV";
        if (hga_out && sh && !dh) return "H2D_attn";
        if (!sh && dh) return "D2H_other";
        if (sh && !dh) return "H2D_other";
        if (sh && dh) return "copy_CPU";
        return "copy_GPU";
    }
    if (t->op == GGML_OP_CUSTOM || (std::strstr(nm, "hga_attn") && !std::strstr(nm, "gpu"))) {
        return "HGA_kernel";
    }
    if (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) {
        return hga_weight_class(t->src[0] && t->src[0]->name[0] ? t->src[0]->name : "");
    }
    if (std::strstr(op, "ROPE")) return "RoPE";
    if (std::strstr(op, "RMS") || std::strstr(nm, "norm")) {
        if (std::strstr(nm, "Qcur") || std::strstr(nm, "Kcur") || std::strstr(nm, "q_norm") ||
            std::strstr(nm, "k_norm") || std::strstr(nm, "normed")) {
            return "q/k-norm";
        }
        return "RMSNorm";
    }
    if (t->op == GGML_OP_MUL && (std::strstr(nm, "normed") || std::strstr(nm, "Qcur") ||
                                 std::strstr(nm, "Kcur"))) {
        return "q/k-norm";
    }
    if (std::strstr(op, "DELTA") || std::strstr(op, "GDN") || std::strstr(nm, "ssm_") ||
        std::strstr(nm, "linear_attn") || std::strstr(nm, "gated_delta")) {
        return "GDN";
    }
    if (std::strstr(nm, "ffn_")) return "FFN";
    return op;
}

static int hga_logits_nq(const ggml_tensor * t) {
    if (!t) return 1;
    /* result_output is [n_vocab, n_q] with n_vocab ~ 2e5. */
    if (t->ne[0] > 1000 && t->ne[1] > 0 && t->ne[1] < 64) return (int) t->ne[1];
    if (t->ne[1] > 1000 && t->ne[0] > 0 && t->ne[0] < 64) return (int) t->ne[0];
    if (t->ne[2] > 1 && t->ne[2] < 64) return (int) t->ne[2];
    return 1;
}

struct hga_ops_ud {
    ggml_backend_t cpu = nullptr;
    ggml_backend_t gpu = nullptr;
};

static void hga_ops_sync(hga_ops_ud * ud) {
    if (ud && ud->cpu) ggml_backend_synchronize(ud->cpu);
    if (ud && ud->gpu) ggml_backend_synchronize(ud->gpu);
}

static void hga_ops_dump(const char * tag, int n_tok,
                         const std::unordered_map<std::string, hga_ops_acc> & m, bool mean) {
    std::vector<std::pair<double, std::string>> rows;
    rows.reserve(m.size());
    double sum = 0;
    const double div = mean ? std::max(1, n_tok) : 1.0;
    for (auto & kv : m) {
        const double ms = kv.second.ms / div;
        rows.push_back({ms, kv.first});
        sum += ms;
    }
    std::sort(rows.begin(), rows.end(), [](auto & x, auto & y) { return x.first > y.first; });
    fprintf(stderr, "hga-ops %s  n_graph=%d  sum=%.2f ms  (~%.1f graphs/s exclusive)\n",
            tag, n_tok, sum, 1000.0 / std::max(0.01, sum));
    /* Prefill is where several individually-small HGA materialization and
       masking nodes matter.  Show enough rows to keep them out of the tail. */
    const size_t nshow = mean ? 20 : (strstr(tag, "PREFILL") ? 48 : 18);
    for (size_t i = 0; i < rows.size() && i < nshow; ++i) {
        const auto & acc = m.at(rows[i].second);
        if (mean) {
            fprintf(stderr, "  %7.2f ms/graph  n=%-5.1f  %s\n",
                    rows[i].first, acc.n / div, rows[i].second.c_str());
        } else {
            fprintf(stderr, "  %7.2f ms  n=%-4d  %8.1f KiB  %s\n",
                    acc.ms, acc.n, acc.bytes / 1024.0, rows[i].second.c_str());
        }
    }
}

static void hga_ops_compare(const std::unordered_map<std::string, hga_ops_acc> & dec,
                            int n_dec,
                            const std::unordered_map<std::string, hga_ops_acc> & ver,
                            int n_ver) {
    if (n_dec < 1 || n_ver < 1) {
        return;
    }
    std::unordered_set<std::string> keys;
    for (auto & kv : dec) keys.insert(kv.first);
    for (auto & kv : ver) keys.insert(kv.first);
    struct row { double d; double v; double delta; std::string k; };
    std::vector<row> rows;
    double sd = 0, sv = 0;
    for (const auto & k : keys) {
        const double d = (dec.count(k) ? dec.at(k).ms : 0) / n_dec;
        const double v = (ver.count(k) ? ver.at(k).ms : 0) / n_ver;
        rows.push_back({d, v, v - d, k});
        sd += d;
        sv += v;
    }
    std::sort(rows.begin(), rows.end(), [](const row & a, const row & b) {
        return std::fabs(a.delta) > std::fabs(b.delta);
    });
    fprintf(stderr,
            "hga-ops COMPARE  mean ms/graph  decode n=%d  verify n=%d  "
            "(exclusive GPU-sync; inflated vs wall — use deltas)\n",
            n_dec, n_ver);
    fprintf(stderr, "  %-18s %10s %10s %10s\n", "module", "decode", "verify", "delta");
    for (const auto & r : rows) {
        if (r.d < 0.3 && r.v < 0.3) {
            continue;
        }
        fprintf(stderr, "  %-18s %10.2f %10.2f %10.2f%s\n",
                r.k.c_str(), r.d, r.v, r.delta,
                std::fabs(r.delta) > 10.0 ? "  <==" : "");
    }
    fprintf(stderr, "  %-18s %10.2f %10.2f %10.2f\n", "SUM", sd, sv, sv - sd);
}

static bool hga_ops_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * ud = (hga_ops_ud *) user_data;
    using clock = std::chrono::steady_clock;
    static clock::time_point t0;
    static std::unordered_map<std::string, hga_ops_acc> tok;
    static std::unordered_map<std::string, hga_ops_acc> dec;
    static std::unordered_map<std::string, hga_ops_acc> ver;
    static int n_dec = 0, n_ver = 0, n_skip = 0, n_prefill = 0;
    static int graph_nq = 1;

    if (ask) {
        hga_ops_sync(ud);
        t0 = clock::now();
        return true;
    }

    hga_ops_sync(ud);
    const double ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    const std::string key = hga_ops_module(t);
    tok[key].ms += ms;
    tok[key].n += 1;
    tok[key].bytes += ggml_nbytes(t);

    /* Logits layout is unreliable; HGA Q is always [dh, n_head, n_q]. */
    if (t->name[0] && std::strstr(t->name, "hga_prefill_flash_attn") &&
        t->ne[2] > 0) {
        /* United PREFILL attention is [dh, n_heads, n_q, 1].  The old
           hga_attn name disappeared when the per-head graphs were united. */
        graph_nq = (int) t->ne[2];
    } else if (t->name[0] && std::strstr(t->name, "hga_Q_cpu") && t->ne[2] > 0) {
        graph_nq = (int) t->ne[2];
    } else if (t->name[0] && std::strstr(t->name, "hga_attn") && t->ne[1] > 0) {
        graph_nq = (int) t->ne[1];
    }

    const bool tok_end =
        key == "lm_head" ||
        (t->name[0] && std::strstr(t->name, "result_output"));
    if (!tok_end) {
        return true;
    }

    const int n_q = graph_nq > 1 ? graph_nq : hga_logits_nq(t);
    graph_nq = 1;
    if (n_q >= 8) {
        const char * pf_e = std::getenv("HGA_PROF_PREFILL");
        const bool profile_prefill = pf_e && pf_e[0] && pf_e[0] != '0';
        if (profile_prefill) {
            ++n_prefill;
            char tag[80];
            std::snprintf(tag, sizeof(tag), "PREFILL n_q=%d #%d", n_q, n_prefill);
            hga_ops_dump(tag, 1, tok, false);
        } else {
            ++n_skip;
        }
        tok.clear();
        return true;
    }

    const bool is_ver = n_q > 1;
    auto & acc = is_ver ? ver : dec;
    int & ng = is_ver ? n_ver : n_dec;
    ng++;
    const char * all_e = std::getenv("HGA_PROF_OPS_ALL");
    const bool dump_all = all_e && all_e[0] && all_e[0] != '0';
    if (dump_all || ng == 1 || (is_ver && n_ver <= 2) || (!is_ver && n_dec <= 2)) {
        char tag[80];
        std::snprintf(tag, sizeof(tag), "%s n_q=%d #%d",
                is_ver ? "VERIFY" : "DECODE", n_q, ng);
        hga_ops_dump(tag, 1, tok, false);
    }
    for (auto & kv : tok) {
        acc[kv.first].ms += kv.second.ms;
        acc[kv.first].n += kv.second.n;
        acc[kv.first].bytes += kv.second.bytes;
    }
    tok.clear();

    fprintf(stderr, "hga-ops graph n_q=%d  %s #%d  (skip_prefill=%d)\n",
            n_q, is_ver ? "VERIFY" : "DECODE", ng, n_skip);
    if (n_dec >= 2 && n_ver >= 2 && (n_ver == 2 || n_ver == 4 || n_ver == 8)) {
        hga_ops_compare(dec, n_dec, ver, n_ver);
    }
    return true;
}

static hga_ops_ud g_hga_ops_ud;

/* Placement census. Default on (HGA_PIN_CHECK=0 to disable). Logs every
 * resident CUDA0 weight that lands on CUDA_Host / CPU — the o_proj-class
 * silent fallback that becomes a per-eval PCIe re-upload. HGA_PIN_ABORT=1
 * abort()s on PUSHED residents. HGA_PIN_VERBOSE=1 lists every MOVE. */
static bool hga_pin_enabled() {
    const char * e = std::getenv("HGA_PIN_CHECK");
    return !(e && e[0] == '0');
}

static bool hga_pin_verbose() {
    const char * e = std::getenv("HGA_PIN_VERBOSE");
    return e && e[0] && e[0] != '0';
}

static bool hga_pin_abort_on() {
    const char * e = std::getenv("HGA_PIN_ABORT");
    return e && e[0] && e[0] != '0';
}

static void hga_pin_log(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("hga-pin: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

enum { HGA_PLACE_GPU = 0, HGA_PLACE_CUDA_HOST = 1, HGA_PLACE_CPU = 2, HGA_PLACE_NULL = 3 };

static const char * hga_place_str(int kind) {
    switch (kind) {
        case HGA_PLACE_GPU:       return "CUDA0";
        case HGA_PLACE_CUDA_HOST: return "CUDA_Host";
        case HGA_PLACE_CPU:       return "CPU";
        default:                  return "null";
    }
}

static const char * hga_pin_cls_str(uint8_t cls) {
    switch (cls) {
        case hga_weight_swap::PIN_RESIDENT: return "resident";
        case hga_weight_swap::PIN_STAGED:   return "staged";
        case hga_weight_swap::PIN_LMHEAD:   return "lm_head";
        case hga_weight_swap::PIN_EXCHANGE: return "exchange";
        case hga_weight_swap::PIN_XFFN:     return "xffn";
        case hga_weight_swap::PIN_HOST:     return "host-ok";
        default:                            return "?";
    }
}

static int hga_place_kind(const ggml_tensor * t, const char ** name_out) {
    if (!t || !t->buffer) {
        if (name_out) {
            *name_out = "null";
        }
        return HGA_PLACE_NULL;
    }
    const char * n = ggml_backend_buffer_name(t->buffer);
    if (!n || !n[0]) {
        n = "?";
    }
    if (name_out) {
        *name_out = n;
    }
    if (!ggml_backend_buffer_is_host(t->buffer)) {
        return HGA_PLACE_GPU;
    }
    if (std::strstr(n, "CUDA_Host") || std::strstr(n, "CUDA_Pinned") ||
        (std::strstr(n, "CUDA") && std::strstr(n, "Host"))) {
        return HGA_PLACE_CUDA_HOST;
    }
    return HGA_PLACE_CPU;
}

static int hga_tensor_blk_id(const char * name) {
    /* Weight names are blk.N.*; graph nodes use a trailing -N (hga_name_layer_id). */
    if (!name || !name[0]) {
        return -1;
    }
    const char * p = std::strstr(name, "blk.");
    if (!p) {
        return -1;
    }
    char * end = nullptr;
    const long v = std::strtol(p + 4, &end, 10);
    if (end == p + 4) {
        return -1;
    }
    return (int) v;
}

static bool hga_name_host_ok(const char * name) {
    return name && (std::strstr(name, "token_embd") ||
                    std::strstr(name, ".attn_q_norm.weight") ||
                    std::strstr(name, ".attn_k_norm.weight"));
}

static bool hga_is_ot_cpu_layer(const hga_weight_swap * sw, int il, const char * name) {
    (void) name;
    if (il < 0) {
        return false;
    }
    for (int pi = 0; pi < sw->n_pairs; ++pi) {
        if (il == sw->pair[pi].layer_a || il == sw->pair[pi].layer_b) {
            return true;
        }
    }
    return false;
}

static bool hga_slot_on_cuda(const std::vector<hga_slot> & slots, const ggml_tensor * t) {
    for (const auto & s : slots) {
        if (s.t == t) {
            return s.on_cuda;
        }
    }
    return false;
}

static bool hga_pin_expect_gpu(const hga_weight_swap * sw, const hga_weight_swap::pin_rec & r) {
    switch (r.cls) {
        case hga_weight_swap::PIN_RESIDENT:
            return true;
        case hga_weight_swap::PIN_STAGED:
            return sw->phase != HGA_SWAP_NONE;
        case hga_weight_swap::PIN_LMHEAD:
            return sw->phase != HGA_SWAP_NONE;
        case hga_weight_swap::PIN_EXCHANGE:
            for (int pi = 0; pi < sw->n_pairs; ++pi) {
                if (hga_slot_on_cuda(sw->pair[pi].a, r.t) ||
                    hga_slot_on_cuda(sw->pair[pi].b, r.t)) {
                    return true;
                }
            }
            return false;
        case hga_weight_swap::PIN_XFFN:
            if (sw->buf_ffn_res && hga_slot_on_cuda(sw->ffn_res, r.t)) {
                return true;
            }
            for (int pi = 0; pi < sw->n_pairs; ++pi) {
                if (hga_slot_on_cuda(sw->pair[pi].extra_a, r.t) ||
                    hga_slot_on_cuda(sw->pair[pi].extra_b, r.t)) {
                    return true;
                }
            }
            return false;
        default:
            return false;
    }
}

static void hga_visit_layer_tensors(const llama_layer & layer,
                                    void (*fn)(ggml_tensor *, void *), void * ud) {
    ggml_tensor * ts[] = {
        layer.attn_norm, layer.attn_post_norm,
        layer.attn_q_norm, layer.attn_k_norm,
        layer.wq, layer.wk, layer.wv, layer.wo,
        layer.wqkv, layer.wqkv_gate, layer.wg,
        layer.ssm_conv1d, layer.ssm_dt, layer.ssm_a, layer.ssm_out, layer.ssm_norm,
        layer.ssm_alpha, layer.ssm_beta, layer.ssm_beta_alpha,
        layer.ffn_gate, layer.ffn_down, layer.ffn_up,
        layer.nextn.eh_proj, layer.nextn.enorm, layer.nextn.hnorm,
        layer.nextn.embed_tokens, layer.nextn.shared_head_head, layer.nextn.shared_head_norm,
    };
    for (ggml_tensor * t : ts) {
        if (t) {
            fn(t, ud);
        }
    }
}

static void hga_pin_add_one(ggml_tensor * t, void * ud) {
    auto * sw = (hga_weight_swap *) ud;
    if (!t || t->view_src || !t->data) {
        return;
    }
    for (const auto & p : sw->pins) {
        if (p.t == t) {
            return;
        }
    }
    hga_weight_swap::pin_rec r;
    r.t = t;
    r.cls = hga_weight_swap::PIN_RESIDENT;
    sw->pins.push_back(r);
}

static void hga_pin_mark(std::unordered_map<ggml_tensor *, uint8_t> & cls,
                         const std::vector<hga_slot> & slots, uint8_t c) {
    for (const auto & s : slots) {
        if (s.t) {
            cls[s.t] = c;
        }
    }
}

static void hga_pin_build(hga_weight_swap * sw) {
    if (!sw || !sw->model || !sw->pins.empty()) {
        return;
    }
    std::unordered_map<ggml_tensor *, uint8_t> cls;
    hga_pin_mark(cls, sw->q,      hga_weight_swap::PIN_STAGED);
    hga_pin_mark(cls, sw->kv,     hga_weight_swap::PIN_STAGED);
    hga_pin_mark(cls, sw->output, hga_weight_swap::PIN_LMHEAD);
    for (int pi = 0; pi < sw->n_pairs; ++pi) {
        hga_pin_mark(cls, sw->pair[pi].a,       hga_weight_swap::PIN_EXCHANGE);
        hga_pin_mark(cls, sw->pair[pi].b,       hga_weight_swap::PIN_EXCHANGE);
        hga_pin_mark(cls, sw->pair[pi].extra_a, hga_weight_swap::PIN_XFFN);
        hga_pin_mark(cls, sw->pair[pi].extra_b, hga_weight_swap::PIN_XFFN);
    }
    hga_pin_mark(cls, sw->ffn_res, hga_weight_swap::PIN_XFFN);

    hga_pin_add_one(sw->model->tok_embd, sw);
    hga_pin_add_one(sw->model->output, sw);
    hga_pin_add_one(sw->model->output_norm, sw);
    for (const auto & layer : sw->model->layers) {
        hga_visit_layer_tensors(layer, hga_pin_add_one, sw);
    }

    for (auto & r : sw->pins) {
        auto it = cls.find(r.t);
        if (it != cls.end()) {
            r.cls = it->second;
            continue;
        }
        const char * nm = r.t->name;
        const int il = hga_tensor_blk_id(nm);
        if (hga_name_host_ok(nm) || hga_is_ot_cpu_layer(sw, il, nm)) {
            r.cls = hga_weight_swap::PIN_HOST;
        } else {
            r.cls = hga_weight_swap::PIN_RESIDENT;
        }
    }
    int n_cls[hga_weight_swap::PIN_NCLS] = {};
    for (const auto & r : sw->pins) {
        n_cls[r.cls]++;
    }
    hga_pin_log("tracking %zu tensors  resident=%d staged=%d lm_head=%d exchange=%d xffn=%d host-ok=%d",
            sw->pins.size(),
            n_cls[hga_weight_swap::PIN_RESIDENT],
            n_cls[hga_weight_swap::PIN_STAGED],
            n_cls[hga_weight_swap::PIN_LMHEAD],
            n_cls[hga_weight_swap::PIN_EXCHANGE],
            n_cls[hga_weight_swap::PIN_XFFN],
            n_cls[hga_weight_swap::PIN_HOST]);
}

static const char * hga_pin_phase_name(hga_swap_phase p) {
    switch (p) {
        case HGA_SWAP_PREFILL: return "PREFILL";
        case HGA_SWAP_DECODE:  return "DECODE";
        case HGA_SWAP_VERIFY:  return "VERIFY";
        default:               return "NONE";
    }
}

static void hga_pin_census(hga_weight_swap * sw, const char * when) {
    if (!sw || !hga_pin_enabled()) {
        return;
    }
    if (sw->pins.empty()) {
        hga_pin_build(sw);
    }

    const bool verbose = hga_pin_verbose();
    int n_kind[4] = {};
    double mib_kind[4] = {};
    int n_cls[hga_weight_swap::PIN_NCLS] = {};
    int n_cls_gpu[hga_weight_swap::PIN_NCLS] = {};
    int n_move = 0, n_move_res = 0, n_pushed = 0;
    int n_move_cls[hga_weight_swap::PIN_NCLS] = {};

    struct row {
        double mib;
        const char * name;
        const char * cls;
        const char * from;
        const char * to;
        uint8_t pcls;
    };
    std::vector<row> pushed;
    std::vector<row> moves;

    for (auto & r : sw->pins) {
        const char * bname = "null";
        const int kind = hga_place_kind(r.t, &bname);
        const double mib = hga_mib(ggml_nbytes(r.t));
        n_kind[kind]++;
        mib_kind[kind] += mib;
        n_cls[r.cls]++;
        if (kind == HGA_PLACE_GPU) {
            n_cls_gpu[r.cls]++;
        }

        const bool expect = hga_pin_expect_gpu(sw, r);
        const bool is_gpu = kind == HGA_PLACE_GPU;
        const char * nm = r.t->name[0] ? r.t->name : "?";

        if (r.last_kind >= 0 && r.last_kind != kind) {
            n_move++;
            n_move_cls[r.cls]++;
            if (r.cls == hga_weight_swap::PIN_RESIDENT ||
                r.cls == hga_weight_swap::PIN_HOST) {
                n_move_res++;
                moves.push_back({mib, nm, hga_pin_cls_str(r.cls),
                        r.last_buf[0] ? r.last_buf : hga_place_str(r.last_kind),
                        bname, r.cls});
            } else if (verbose) {
                moves.push_back({mib, nm, hga_pin_cls_str(r.cls),
                        r.last_buf[0] ? r.last_buf : hga_place_str(r.last_kind),
                        bname, r.cls});
            }
        }

        if (expect && !is_gpu) {
            n_pushed++;
            pushed.push_back({mib, nm, hga_pin_cls_str(r.cls),
                    r.last_kind >= 0
                        ? (r.last_buf[0] ? r.last_buf : hga_place_str(r.last_kind))
                        : "(never CUDA0)",
                    bname, r.cls});
        }

        r.last_kind = (int8_t) kind;
        std::snprintf(r.last_buf, sizeof(r.last_buf), "%s", bname);
    }

    std::sort(pushed.begin(), pushed.end(),
            [](const row & a, const row & b) { return a.mib > b.mib; });
    std::sort(moves.begin(), moves.end(),
            [](const row & a, const row & b) { return a.mib > b.mib; });

    hga_pin_log("census %s  phase=%s  tensors=%zu  "
            "GPU=%d/%.1fMiB  CUDA_Host=%d/%.1fMiB  CPU=%d/%.1fMiB  "
            "PUSHED=%d  MOVE=%d (resident %d)",
            when, hga_pin_phase_name(sw->phase), sw->pins.size(),
            n_kind[HGA_PLACE_GPU], mib_kind[HGA_PLACE_GPU],
            n_kind[HGA_PLACE_CUDA_HOST], mib_kind[HGA_PLACE_CUDA_HOST],
            n_kind[HGA_PLACE_CPU], mib_kind[HGA_PLACE_CPU],
            n_pushed, n_move, n_move_res);

    for (uint8_t c = 0; c < hga_weight_swap::PIN_NCLS; ++c) {
        if (n_cls[c] == 0) {
            continue;
        }
        hga_pin_log("  %-9s %3d GPU / %3d host  (n=%d)",
                hga_pin_cls_str(c), n_cls_gpu[c], n_cls[c] - n_cls_gpu[c], n_cls[c]);
    }

    ggml_tensor * lm = !sw->output.empty() ? sw->output.front().t :
            (sw->model ? sw->model->output : nullptr);
    if (lm) {
        const char * bname = "null";
        const int kind = hga_place_kind(lm, &bname);
        hga_pin_log("  lm_head  buf=%s  %s  %.1f MiB  staged=%d",
                bname, hga_place_str(kind),
                hga_mib(ggml_nbytes(lm)),
                sw->buf_out ? 1 : 0);
    }

    for (int pi = 0; pi < sw->n_pairs; ++pi) {
        const auto & p = sw->pair[pi];
        if (!p.tag[0]) {
            continue;
        }
        hga_pin_log("  slot %-7s occ=%c copy=%c  cap=%.1f MiB  buf=%s",
                p.tag,
                p.occ == 1 ? 'A' : (p.occ == 2 ? 'B' : '-'),
                p.copy == 1 ? 'A' : (p.copy == 2 ? 'B' : '-'),
                hga_mib(p.cap),
                p.buf ? ggml_backend_buffer_name(p.buf) : "unbound");
    }

    const size_t nshow = 24;
    for (size_t i = 0; i < pushed.size() && i < nshow; ++i) {
        const auto & r = pushed[i];
        hga_pin_log("PUSHED  %-40s  class=%-9s  %s -> %s  %.2f MiB",
                r.name, r.cls, r.from, r.to, r.mib);
    }
    if (pushed.size() > nshow) {
        hga_pin_log("PUSHED  ... %zu more", pushed.size() - nshow);
    }

    for (size_t i = 0; i < moves.size() && i < nshow; ++i) {
        const auto & r = moves[i];
        hga_pin_log("MOVE    %-40s  class=%-9s  %s -> %s  %.2f MiB",
                r.name, r.cls, r.from, r.to, r.mib);
    }
    if (n_move && !verbose) {
        hga_pin_log("MOVE    staged=%d lm_head=%d exchange=%d xffn=%d  "
                "(HGA_PIN_VERBOSE=1 lists them)",
                n_move_cls[hga_weight_swap::PIN_STAGED],
                n_move_cls[hga_weight_swap::PIN_LMHEAD],
                n_move_cls[hga_weight_swap::PIN_EXCHANGE],
                n_move_cls[hga_weight_swap::PIN_XFFN]);
    }

    if (n_pushed == 0) {
        hga_pin_log("ok  no expected-GPU tensor on CUDA_Host/CPU");
    } else if (std::strcmp(when, "at init") == 0) {
        hga_pin_log("PUSHED at load — these never sat on CUDA0 "
                "(silent CUDA_Host fallback / -ot miss)");
    }

    if (n_pushed > 0 && hga_pin_abort_on()) {
        hga_pin_log("ABORT HGA_PIN_ABORT=1  %d PUSHED tensors", n_pushed);
        std::fflush(stderr);
        std::abort();
    }
    std::fflush(stderr);
}

static void hga_ops_hook_cparams(llama_cparams & cparams, ggml_backend_sched_t sched,
                                 ggml_backend_t cpu, ggml_backend_t gpu) {
    if (!std::getenv("HGA_PROF_OPS")) {
        return;
    }
    g_hga_ops_ud.cpu = cpu;
    g_hga_ops_ud.gpu = gpu;
    cparams.cb_eval = hga_ops_eval_cb;
    cparams.cb_eval_user_data = &g_hga_ops_ud;
    if (sched) {
        ggml_backend_sched_set_eval_callback(sched, hga_ops_eval_cb, &g_hga_ops_ud);
    }
    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "hga-ops: hooked cparams.cb_eval (exclusive per-op; inflates tok/s)\n");
    }
}

void llama_context::hga_vram_log(const char * tag, uint32_t n_tokens) {
    static int step = 0;
    size_t free_b = 0, total_b = 0, compute_gpu = 0, compute_cpu = 0;
    for (auto & b : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(b.get());
        if (!dev) {
            continue;
        }
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            ggml_backend_dev_memory(dev, &free_b, &total_b);
            if (sched) {
                compute_gpu = ggml_backend_sched_get_buffer_size(sched.get(), b.get());
            }
        } else if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && sched) {
            compute_cpu = ggml_backend_sched_get_buffer_size(sched.get(), b.get());
        }
    }
    const double out_mib = buf_output
            ? hga_mib(ggml_backend_buffer_get_size(buf_output.get())) : 0.0;
    hga_swap_log("STEP %02d %-32s  free=%6.0f used=%6.0f  compute=%.1f cpu=%.1f out=%.1f  "
            "n_tok=%u n_ubatch=%u n_omax=%u n_rs=%u phase=%d ctx=%d",
            ++step, tag ? tag : "?",
            hga_mib(free_b),
            hga_mib(total_b > free_b ? total_b - free_b : 0),
            hga_mib(compute_gpu), hga_mib(compute_cpu), out_mib,
            n_tokens, cparams.n_ubatch, cparams.n_outputs_max, cparams.n_rs_seq,
            cparams.hga_phase, (int) cparams.ctx_type);
    const char * t = tag ? tag : "";
    if (std::strstr(t, "FAIL") || std::strstr(t, "fail") || std::strstr(t, "OOM")) {
        hga_alloc_dump_oom(t);
    }
    if (std::strstr(t, "reserve") || std::strstr(t, "drop") || std::strstr(t, "ctor") ||
            std::strstr(t, "leftover") || std::strstr(t, "swap init")) {
        const char * keep = std::getenv("HGA_VMM_KEEP");
        hga_swap_log("MEM  %-32s  cuda_free=%.0f used=%.0f  compute_gpu=%.1f compute_cpu=%.1f  "
                "buf_output=%.1f  n_ubatch=%u n_ubatch_orig=%u n_omax=%u n_ctx=%u  VMM_KEEP=%s",
                t,
                hga_mib(free_b),
                hga_mib(total_b > free_b ? total_b - free_b : 0),
                hga_mib(compute_gpu), hga_mib(compute_cpu), out_mib,
                cparams.n_ubatch, cparams.hga_n_ubatch_orig, cparams.n_outputs_max,
                cparams.n_ctx,
                (keep && keep[0] && keep[0] != '0') ? "1" : "0");
    }
    std::fflush(stderr);
}

/* The speculative runner owns one trunk and one MTP context.  This pointer
 * is used only to release the trunk's idle PREFILL compute arena before MTP
 * prompt catch-up.  Model buffers, including output.weight/lm_head, have a
 * separate lifetime and remain resident on CUDA. */
static llama_context * g_hga_target_context = nullptr;
static llama_context * g_hga_mtp_context = nullptr;

void llama_context::hga_swap_init() {
    if (cparams.hga_swap) {
        return;
    }
    if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
        hga_swap_log("init skipped (MTP draft context)");
        return;
    }
    if (!cparams.hga_enabled) {
        hga_swap_log("init skipped (hga_enabled=0)");
        return;
    }
    const char * env = std::getenv("HGA_SWAP");
    if (env && env[0] == '0') {
        hga_swap_log("HGA_SWAP=0 — two-mode QKV packing disabled");
        return;
    }

    ggml_backend_t gpu = nullptr;
    for (auto & b : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(b.get());
        const char * bname = ggml_backend_name(b.get());
        const int dtype = dev ? (int) ggml_backend_dev_type(dev) : -1;
        hga_swap_log("backend %s type=%d", bname ? bname : "?", dtype);
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = b.get();
            break;
        }
    }
    if (!gpu) {
        hga_swap_log("no GPU backend — two-mode packing disabled");
        return;
    }

    cparams.hga_n_ubatch_orig = cparams.n_ubatch;
    cparams.hga_swap = hga_weight_swap_init(&model, gpu);
    if (cparams.hga_swap) {
        g_hga_target_context = this;
        hga_swap_log("packing ready (resident CUDA incl. lm_head/QKV; exchange stream only; decode contiguous HGA D2H) n_rs_seq=%u n_ubatch=%u",
                cparams.n_rs_seq, cparams.n_ubatch);
        hga_vram_step((hga_weight_swap *) cparams.hga_swap, "at swap init",
                0, cparams.n_ubatch, cparams.n_rs_seq);
    } else {
        hga_swap_log("init failed — staying on host mmap QKV");
    }
}

static uint32_t hga_env_u32(const char * name, uint32_t fallback) {
    const char * e = std::getenv(name);
    if (!e || !e[0]) {
        return fallback;
    }
    const int v = std::atoi(e);
    return v > 0 ? (uint32_t) v : fallback;
}

/* Batches larger than this are prompt prefill. Speculative verify is K+1
 * (K = n_rs_seq / HGA_SPEC, typically 2–4). llama_decode() calls
 * hga_swap_ensure with n_tokens_all, not leftover ubatch sizes. */
static uint32_t hga_spec_max_tokens(const llama_cparams & cparams) {
    uint32_t spec_max = hga_env_u32("HGA_SPEC_MAX", HGA_SPEC_MAX_DEFAULT);
    if (spec_max < 2) {
        spec_max = HGA_SPEC_MAX_DEFAULT;
    }
    const uint32_t verify = cparams.n_rs_seq > 0 ? cparams.n_rs_seq + 1 : 0;
    if (verify > spec_max) {
        spec_max = verify;
    }
    const uint32_t spec_k = hga_env_u32("HGA_SPEC", 0);
    if (spec_k + 1 > spec_max) {
        spec_max = spec_k + 1;
    }
    return spec_max;
}

/* Recurrent rollback keeps the trailing n_keep_tail = n_rs_seq+1 tokens in
 * ONE ubatch. split_equal requires n_ubatch >= n_keep_tail (the tail fits in
 * the next ubatch; `>` was an off-by-one). Generate graph is K+1 tokens. */
static uint32_t hga_decode_ubatch(const llama_cparams & cparams) {
    uint32_t k = cparams.n_rs_seq;
    const uint32_t spec_k = hga_env_u32("HGA_SPEC", 0);
    if (spec_k > k) {
        k = spec_k;
    }
    return k > 0 ? k + 1 : 1u;
}

void llama_context::hga_swap_ensure(uint32_t n_tokens) {
    /* MTP has no weight swap, but it must use the same generate ubatch as the
     * trunk (K+1). Otherwise process() n=3 and draft() n=1 rebuild the leftover
     * prefill n_ubatch=512 graph every spec step, and MTP attention never sees
     * the other tokens in the verify window. */
    if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
        g_hga_mtp_context = this;
        if (cparams.warmup) {
            return;
        }
        if (cparams.hga_n_ubatch_orig == 0) {
            cparams.hga_n_ubatch_orig = cparams.n_ubatch;
        }
        const uint32_t spec_max = hga_spec_max_tokens(cparams);
        if (n_tokens > spec_max) {
            /* A member may access another llama_context's private state.  Drop
             * only its scheduler/graphs; target model tensors are untouched. */
            llama_context * target = g_hga_target_context;
            if (target && target != this && target->cparams.hga_swap &&
                    target->cparams.hga_seen_large_prefill && target->sched) {
                auto * target_sw = (hga_weight_swap *) target->cparams.hga_swap;
                if (hga_weight_swap_phase(target_sw) == HGA_SWAP_PREFILL) {
                    target->hga_vram_log("MTP before release target prefill", 0);
                    hga_swap_log("MTP catch-up: releasing target PREFILL compute graph; lm_head remains CUDA-resident");
                    setenv("HGA_VMM_UNMAP", "1", 1);
                    target->synchronize();
                    target->hga_graph_cache_detach();
                    target->sched.reset();
                    hga_trim_prefill_vmm();
                    unsetenv("HGA_VMM_UNMAP");
                    target->sched_need_reserve = true;
                    target->hga_vram_log("MTP after release target prefill", 0);
                }
            }
            cparams.hga_seen_prefill = true;
            cparams.hga_seen_large_prefill = true;
            if (cparams.n_ubatch != cparams.hga_n_ubatch_orig) {
                cparams.n_ubatch = cparams.hga_n_ubatch_orig;
                sched_need_reserve = true;
            }
            return;
        }
        if (!cparams.hga_seen_large_prefill) {
            return;
        }
        const uint32_t want = hga_decode_ubatch(cparams);
        if (cparams.n_ubatch != want) {
            cparams.n_ubatch = want;
            sched_need_reserve = true;
            hga_swap_log("MTP generate n_tokens=%u n_ubatch=%u (match trunk VERIFY)",
                    n_tokens, want);
        }
        return;
    }

    if (!cparams.hga_swap) {
        static bool once = false;
        if (!once) {
            once = true;
            hga_swap_log("ensure n_tokens=%u skipped (no swap state)", n_tokens);
        }
        return;
    }

    auto * sw_early = (hga_weight_swap *) cparams.hga_swap;
    hga_swap_log("ensure enter n_tokens=%u n_ubatch=%u n_rs_seq=%u warmup=%d phase=%d ctx=%d",
            n_tokens, cparams.n_ubatch, cparams.n_rs_seq,
            (int) cparams.warmup, (int) hga_weight_swap_phase(sw_early),
            (int) cparams.ctx_type);
    hga_vram_log("ensure enter", n_tokens);

    /* Graph warmup (often n=1 or n=2) must not stage lm_head. Spec used to
     * DECODE after that, PREFILL the real prompt, then fail to restage the
     * 995 MiB output.weight (CUDA OOM) and spend decode on a host lm_head. */
    if (cparams.warmup) {
        static bool logged_warmup = false;
        if (!logged_warmup) {
            logged_warmup = true;
            hga_swap_log("skip n_tokens=%u (graph warmup)", n_tokens);
        }
        return;
    }

    const uint32_t spec_max = hga_spec_max_tokens(cparams);
    auto * sw = (hga_weight_swap *) cparams.hga_swap;
    const hga_swap_phase cur = hga_weight_swap_phase(sw);
    hga_swap_phase want;
    if (n_tokens > spec_max) {
        want = HGA_SWAP_PREFILL;
        cparams.hga_seen_prefill = true;
        cparams.hga_seen_large_prefill = true;
    } else if (!cparams.hga_seen_large_prefill) {
        /* Stay PREFILL until a real prompt (n > spec_max). speculative-simple
         * evals a dummy n=2 batch, then creates MTP, then the 2K/4K prompt.
         * DECODE packing loads extra FFN and used to drop the 128-token graph;
         * either one leaves too little VRAM for the real prefill next to MTP.
         * Short prompts never exceed spec_max and keep the PREFILL graph. */
        if (n_tokens <= 1 && !cparams.hga_seen_prefill) {
            return;
        }
        want = HGA_SWAP_PREFILL;
        if (n_tokens > 1) {
            cparams.hga_seen_prefill = true;
        }
    } else {
        /* Generate is always VERIFY (CPU q/k-norm + Q-cont), even for n=1. */
        want = HGA_SWAP_VERIFY;
    }

    /* llama-server alternates target and MTP catch-up for each physical
     * prompt ubatch.  If the preceding MTP catch-up released our PREFILL
     * graph—or this is a VERIFY -> PREFILL continuation—drop MTP's idle
     * compute graph before reserving the large target graph.  This is the
     * reverse half of the graph-only handoff above; neither side owns model
     * weights, so lm_head and the resident layer tensors are not reloaded. */
    if (want == HGA_SWAP_PREFILL && g_hga_mtp_context &&
            g_hga_mtp_context != this && g_hga_mtp_context->sched &&
            (!sched || cur != want)) {
        llama_context * mtp = g_hga_mtp_context;
        mtp->hga_vram_log("target before release MTP graph", 0);
        hga_swap_log("target PREFILL: releasing MTP compute graph; lm_head remains CUDA-resident");
        setenv("HGA_VMM_UNMAP", "1", 1);
        mtp->synchronize();
        mtp->hga_graph_cache_detach();
        mtp->sched.reset();
        hga_trim_prefill_vmm();
        unsetenv("HGA_VMM_UNMAP");
        mtp->sched_need_reserve = true;
        mtp->hga_vram_log("target after release MTP graph", 0);
    }

    if (cur == want) {
        if (want == HGA_SWAP_VERIFY) {
            static bool logged_verify = false;
            if (!logged_verify) {
                logged_verify = true;
                hga_swap_log("VERIFY n_tokens=%u (decode graph, CPU q/k-norm; n_ubatch=%u n_rs_seq=%u spec_max=%u)",
                        n_tokens, cparams.n_ubatch, cparams.n_rs_seq, spec_max);
            }
        }
        hga_stream_begin_ubatch(sw);
        hga_stream_bind_eval(cparams, sw, sched.get());
        return;
    }

    synchronize();

    cparams.hga_phase = (int32_t) want;
    if (cparams.hga_n_ubatch_orig == 0) {
        cparams.hga_n_ubatch_orig = cparams.n_ubatch;
    }
    /* Decode pipeline: op_offload off so the CPU HGA op does not expand
     * through GPU neighbors. Prefill keeps op_offload and ubatch
     * min(orig, HGA_PREFILL_UBATCH) (default 768).
     * lm_head and non-exchange QKV stay CUDA; only exchange slots H2D.
     * VERIFY shares DECODE packing and the DECODE activation graph. */
    if (want == HGA_SWAP_DECODE || want == HGA_SWAP_VERIFY) {
        if (hga_decode_pack((int32_t) cur)) {
            cparams.op_offload = false;
            cparams.n_ubatch = hga_decode_ubatch(cparams);
            if (!hga_weight_swap_set_phase(sw, want, /*stage_output=*/true)) {
                hga_swap_log("failed to switch %s", want == HGA_SWAP_VERIFY ? "VERIFY" : "DECODE");
                return;
            }
            hga_stream_bind_eval(cparams, sw, sched.get());
            /* Prefill graph is already gone; re-reserve for VERIFY Q-cont. */
            sched_need_reserve = true;
            hga_vram_step(sw, "DECODE re-reserve before", n_tokens, cparams.n_ubatch, cparams.n_rs_seq);
            sched_reserve();
            hga_vram_step(sw, "DECODE re-reserve after", n_tokens, cparams.n_ubatch, cparams.n_rs_seq);
            sched_need_reserve = false;
            hga_swap_log("%s n_tokens=%u n_ubatch=%u (decode packing; CPU q/k-norm, Q-cont=%d)",
                    want == HGA_SWAP_VERIFY ? "VERIFY" : "DECODE",
                    n_tokens, cparams.n_ubatch, want == HGA_SWAP_VERIFY ? 1 : 0);
            return;
        }
        ggml_backend_t cpu_be = nullptr;
        ggml_backend_t gpu_be = nullptr;
        for (auto & b : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(b.get());
            if (!dev) {
                continue;
            }
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                cpu_be = b.get();
            } else if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                gpu_be = b.get();
            }
        }
        cparams.op_offload = false;
        const uint32_t decode_ub = hga_decode_ubatch(cparams);
        cparams.n_ubatch = decode_ub;
        if (!hga_weight_swap_set_phase(sw, want, /*stage_output=*/true)) {
            hga_swap_log("failed to enter DECODE");
            return;
        }
        /* speculative-simple evals a dummy n=2 batch before the real prompt
         * (and before/after MTP ctx init). Dropping the 128-token PREFILL
         * graph there, then reallocating it for a 2K/4K prompt with MTP
         * already resident, OOMs. Keep the prefill graph until a real
         * prompt (n_tokens > spec_max) has run. */
        if (!cparams.hga_seen_large_prefill) {
            hga_stream_bind_eval(cparams, sw, sched.get());
            hga_vram_step(sw, "DECODE keep prefill graph", n_tokens, cparams.n_ubatch, cparams.n_rs_seq);
            hga_swap_log("DECODE n_tokens=%u n_ubatch=%u keep prefill graph (no prompt > spec_max=%u yet)",
                    n_tokens, cparams.n_ubatch, spec_max);
            return;
        }
        /* Destroy the PREFILL ggml graph (n_ubatch=128 compute/scratch) and
         * reserve only the decode/verify graph (n_ubatch=K+1). The CUDA VMM
         * pool unmaps when that scheduler is freed (see ggml-cuda.cu). */
        hga_vram_step(sw, "DECODE before drop prefill graphs", n_tokens, cparams.n_ubatch, cparams.n_rs_seq);
        hga_vram_log("DECODE before drop prefill graphs", n_tokens);
        hga_swap_log("dropping prefill graphs  n_ubatch_orig=%u -> n_ubatch=%u  n_tokens=%u n_ctx=%u n_omax=%u spec_max=%u  (replace 512-token sched with K+1)",
                cparams.hga_n_ubatch_orig, cparams.n_ubatch, n_tokens,
                cparams.n_ctx, cparams.n_outputs_max, spec_max);
        /* Free the 512-token graph (and VMM pages) BEFORE leftover pin, and
         * do not allocate the 3-token graph until the 5 pairs are resident. */
        setenv("HGA_VMM_UNMAP", "1", 1);
        synchronize();
        hga_graph_cache_detach();
        sched.reset();
        /* sched.reset() releases graph ownership, but VMM arenas can already
         * have pool_used==0 by then and therefore do not pass through free().
         * Explicitly trim those stale physical mappings before pinning the
         * five A+B VERIFY pairs (they need roughly 2.25 GiB together). */
        hga_trim_prefill_vmm();
        unsetenv("HGA_VMM_UNMAP");
        if (buf_output) {
            hga_swap_log("releasing buf_output %.1f MiB before leftover pin",
                    hga_mib(ggml_backend_buffer_get_size(buf_output.get())));
            buf_output.reset();
        }
        hga_vram_log("DECODE after drop prefill graphs", n_tokens);
        hga_stream_pin_leftover_verify(sw);
        hga_weight_swap_try_split(sw);
        hga_pin_census(sw, "after leftover VERIFY pins");
        sched_need_reserve = true;
        sched_reserve();
        output_reserve((int32_t) std::max(1u, cparams.n_outputs_max));
        sched_need_reserve = false;
        hga_vram_log("after leftover VERIFY pins", n_tokens);
        /* New scheduler: op profiler first, then stream eval wraps it. */
        hga_ops_hook_cparams(cparams, sched.get(), cpu_be, gpu_be);
        hga_stream_bind_eval(cparams, sw, sched.get());
        return;
    }

    cparams.op_offload = true;
    {
        uint32_t cap = hga_env_u32("HGA_PREFILL_UBATCH", 768);
        if (cap < 8) {
            cap = 8;
        }
        cparams.n_ubatch = std::min(cparams.hga_n_ubatch_orig, cap);
    }

    if (!hga_weight_swap_set_phase(sw, want)) {
        hga_swap_log("failed to enter PREFILL");
        cparams.hga_phase = HGA_SWAP_NONE;
        cparams.n_ubatch = cparams.hga_n_ubatch_orig;
        cparams.op_offload = false;
        return;
    }

    /* The exclusive op profiler used to be installed only after transition
     * to VERIFY, so HGA_PROF_OPS silently skipped the expensive PREFILL
     * graphs. Install it before the streamer wrapper when explicitly asked;
     * hga_stream_bind_eval preserves it as prev_eval and chains callbacks. */
    if (std::getenv("HGA_PROF_PREFILL")) {
        ggml_backend_t cpu_be = nullptr;
        ggml_backend_t gpu_be = nullptr;
        for (auto & b : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(b.get());
            if (!dev) {
                continue;
            }
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                cpu_be = b.get();
            } else if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                gpu_be = b.get();
            }
        }
        hga_ops_hook_cparams(cparams, sched.get(), cpu_be, gpu_be);
    }
    hga_stream_bind_eval(cparams, sw, sched.get());
    /* Must set the flag: sched_reserve() is a no-op otherwise. DECODE→PREFILL
     * (dummy prompt then the real 2K/4K prompt in spec) would keep the small
     * decode graph and OOM inside graph_compute of a 128-token ubatch. */
    sched_need_reserve = true;
    hga_vram_step(sw, "PREFILL before sched_reserve", n_tokens, cparams.n_ubatch, cparams.n_rs_seq);
    hga_swap_log("calling sched_reserve n_ubatch=%u n_tokens=%u warmup=%d",
            cparams.n_ubatch, n_tokens, (int) cparams.warmup);
    sched_reserve();
    hga_vram_step(sw, "PREFILL after sched_reserve", n_tokens, cparams.n_ubatch, cparams.n_rs_seq);
    hga_vram_log("PREFILL after sched_reserve", n_tokens);
    sched_need_reserve = false;
    hga_swap_log("re-reserve compute n_ubatch=%u n_tokens=%u phase=PREFILL op_offload=1",
            cparams.n_ubatch, n_tokens);
}

void llama_context::hga_swap_free() {
    if (g_hga_target_context == this) {
        g_hga_target_context = nullptr;
    }
    if (g_hga_mtp_context == this) {
        g_hga_mtp_context = nullptr;
    }
    if (!cparams.hga_swap) {
        return;
    }
    hga_weight_swap_free((hga_weight_swap *) cparams.hga_swap);
    cparams.hga_swap = nullptr;
    cparams.hga_phase = HGA_SWAP_NONE;
}
