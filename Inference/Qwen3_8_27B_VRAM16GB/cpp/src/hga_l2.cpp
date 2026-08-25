#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "hga_l2.h"
#include "hga.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__linux__)
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>
#endif

#if defined(__AVX512F__)
#define HGA_L2_AVX512 1
#else
#define HGA_L2_AVX512 0
#endif

namespace {

enum Cmd : int {
    CMD_NONE     = 0,
    CMD_BIND     = 1,
    CMD_PREFETCH = 2,
    CMD_GEMV     = 3,
    CMD_STOP     = 4,
};

static inline float hga_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) {
            out = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) { man <<= 1; exp--; }
            man &= 0x3ffu;
            out = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (man << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

static inline uint16_t hga_f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;
        man = (man | 0x800000u) >> (1 - exp);
        return (uint16_t) (sign | (man >> 13));
    }
    if (exp >= 31) return (uint16_t) (sign | 0x7c00u);
    return (uint16_t) (sign | ((uint32_t) exp << 10) | (man >> 13));
}

static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static inline float dot_f32(const float * a, const float * b, int n) {
    int i = 0;
    float s = 0.f;
#if HGA_L2_AVX512
    __m512 acc = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    }
    s = _mm512_reduce_add_ps(acc);
#endif
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

static inline void touch_bytes(const void * p, size_t n) {
    if (!p || n == 0) return;
    const uint8_t * b = (const uint8_t *) p;
    volatile uint64_t acc = 0;
    size_t i = 0;
    for (; i + 64 <= n; i += 64) acc += *(const uint64_t *) (b + i);
    (void) acc;
}

struct CpuCore {
    int cpu;
    int pkg;
};

static std::vector<CpuCore> physical_cores() {
    std::vector<CpuCore> out;
#if defined(__linux__)
    for (int cpu = 0; cpu < 256; ++cpu) {
        char path[160];
        std::snprintf(path, sizeof(path),
                      "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        FILE * f = std::fopen(path, "r");
        if (!f) break;
        int first = cpu;
        if (std::fscanf(f, "%d", &first) != 1) first = cpu;
        std::fclose(f);
        if (first != cpu) continue;
        int pkg = 0;
        std::snprintf(path, sizeof(path),
                      "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        f = std::fopen(path, "r");
        if (f) {
            if (std::fscanf(f, "%d", &pkg) != 1) pkg = 0;
            std::fclose(f);
        }
        out.push_back({cpu, pkg});
    }
    std::sort(out.begin(), out.end(), [](const CpuCore & a, const CpuCore & b) {
        return a.pkg < b.pkg || (a.pkg == b.pkg && a.cpu < b.cpu);
    });
#endif
    return out;
}

static void pin_to_cpu(int cpu) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void) cpu;
#endif
}

struct ThreadLayer {
    std::vector<uint8_t> k_w;
    std::vector<uint8_t> v_w;
    std::vector<int>     k_ids;
    std::vector<int>     v_ids;
    size_t k_blk_bytes = 0;
    size_t v_blk_bytes = 0;
    int    k_blk_n     = 256;
    int    v_blk_n     = 256;
    hga_l2_dequant_fn k_deq = nullptr;
    hga_l2_dequant_fn v_deq = nullptr;
};

} /* namespace */

void hga_q4k_dequant_block(const void * block, float * dst, int64_t n) {
    const auto * x = (const hga_q4k_block *) block;
    const int N = (n > 0) ? (int) n : HGA_QK_K;
    const float d   = hga_f16_to_f32(x->d);
    const float min = hga_f16_to_f32(x->dmin);
    const uint8_t * q = x->qs;
    int is = 0;
    int yi = 0;
    for (int j = 0; j < HGA_QK_K && yi < N; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, x->scales, &sc, &m);
        const float d1 = d * (float) sc, m1 = min * (float) m;
        get_scale_min_k4(is + 1, x->scales, &sc, &m);
        const float d2 = d * (float) sc, m2 = min * (float) m;
        for (int l = 0; l < 32 && yi < N; ++l) dst[yi++] = d1 * (float) (q[l] & 0xF) - m1;
        for (int l = 0; l < 32 && yi < N; ++l) dst[yi++] = d2 * (float) (q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

void hga_q4k_make_uniform(hga_q4k_block * b, float d, uint8_t nibble) {
    std::memset(b, 0, sizeof(*b));
    b->d    = hga_f32_to_f16(d);
    b->dmin = 0;
    b->scales[0] = 1; b->scales[1] = 1; b->scales[2] = 1; b->scales[3] = 1;
    b->scales[4] = 0; b->scales[5] = 0; b->scales[6] = 0; b->scales[7] = 0;
    b->scales[8] = 0x01; b->scales[9] = 0x01; b->scales[10] = 0x01; b->scales[11] = 0x01;
    const uint8_t packed = (uint8_t) ((nibble & 0xF) | ((nibble & 0xF) << 4));
    std::memset(b->qs, packed, sizeof(b->qs));
}

struct hga_l2_plan {
    int n_threads = 1;
    int n_layers  = 0;
    int n_embd    = 0;
    int n_out     = 0;
    int n_blocks  = 0;
    int n_k_th    = 1;
    int n_v_th    = 1;

    std::vector<CpuCore> cores;
    std::vector<std::vector<ThreadLayer>> tl; /* [layer][thread] */
    int bind_layer = 0;
    const uint8_t * bind_wk = nullptr;
    const uint8_t * bind_wv = nullptr;
    size_t bind_wk_rb = 0, bind_wv_rb = 0;
    size_t bind_wk_bb = 0, bind_wv_bb = 0;
    int bind_wk_bn = 256, bind_wv_bn = 256;
    hga_l2_dequant_fn bind_wk_deq = nullptr;
    hga_l2_dequant_fn bind_wv_deq = nullptr;

    int gemv_layer = 0;
    const float * gemv_x = nullptr;
    float * gemv_k = nullptr;
    float * gemv_v = nullptr;
    std::vector<std::vector<float>> partial_k;
    std::vector<std::vector<float>> partial_v;

    int pref_layer = 0;
    const hga_session * pref_sess = nullptr;
    const int * pref_keys = nullptr;
    int pref_n_keys = 0;
    int pref_workers = 1;
    bool pref_touch_weights = true;
    bool pref_touch_summaries = false;
    std::vector<int> pref_keys_buf[2];
    int pref_buf_i = 0;

    std::atomic<int> cmd{CMD_NONE};
    std::atomic<int> epoch{0};
    std::atomic<int> done{0};
    std::atomic<int> running{0};
    std::atomic<int> started{0};

#if defined(__linux__)
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t  cv = PTHREAD_COND_INITIALIZER;
    std::vector<pthread_t> th;
    int dma_fd = -1;
#endif
};

static void assign_ids(hga_l2_plan * p, int tid, std::vector<int> & k_ids, std::vector<int> & v_ids) {
    k_ids.clear();
    v_ids.clear();
    if (p->n_threads == 1) {
        for (int b = 0; b < p->n_blocks; ++b) {
            k_ids.push_back(b);
            v_ids.push_back(b);
        }
        return;
    }
    if (tid < p->n_k_th) {
        for (int b = 0; b < p->n_blocks; ++b) {
            if (b % p->n_k_th == tid) k_ids.push_back(b);
        }
    } else {
        const int vt = tid - p->n_k_th;
        for (int b = 0; b < p->n_blocks; ++b) {
            if (b % p->n_v_th == vt) v_ids.push_back(b);
        }
    }
}

static void do_bind(hga_l2_plan * p, int tid) {
    const int L = p->bind_layer;
    ThreadLayer & T = p->tl[(size_t) L][(size_t) tid];
    assign_ids(p, tid, T.k_ids, T.v_ids);
    T.k_blk_bytes = p->bind_wk_bb;
    T.v_blk_bytes = p->bind_wv_bb;
    T.k_blk_n     = p->bind_wk_bn;
    T.v_blk_n     = p->bind_wv_bn;
    T.k_deq       = p->bind_wk_deq;
    T.v_deq       = p->bind_wv_deq;

    const int n_out = p->n_out;
    auto pack_rows = [](std::vector<uint8_t> & dstw, const uint8_t * srcw,
                        size_t src_row_bytes, size_t blk_bytes,
                        const std::vector<int> & ids, int n_out_rows) {
        dstw.assign((size_t) n_out_rows * ids.size() * blk_bytes, 0);
        uint8_t * dst = dstw.data();
        const uint8_t * src_row = srcw;
        for (int r = 0; r < n_out_rows; ++r) {
            for (int b : ids) {
                std::memcpy(dst, src_row + (size_t) b * blk_bytes, blk_bytes);
                dst += blk_bytes;
            }
            src_row += src_row_bytes;
        }
    };
    if (!T.k_ids.empty() && p->bind_wk) {
        pack_rows(T.k_w, p->bind_wk, p->bind_wk_rb, T.k_blk_bytes, T.k_ids, n_out);
    } else {
        T.k_w.clear();
    }
    if (!T.v_ids.empty() && p->bind_wv) {
        pack_rows(T.v_w, p->bind_wv, p->bind_wv_rb, T.v_blk_bytes, T.v_ids, n_out);
    } else {
        T.v_w.clear();
    }
}

static void gemv_one_matrix(const ThreadLayer & T, const float * x, float * partial, int n_out, bool is_v) {
    const std::vector<int> & ids = is_v ? T.v_ids : T.k_ids;
    const std::vector<uint8_t> & w = is_v ? T.v_w : T.k_w;
    const size_t bb = is_v ? T.v_blk_bytes : T.k_blk_bytes;
    const int bn = is_v ? T.v_blk_n : T.k_blk_n;
    const hga_l2_dequant_fn deq = is_v ? T.v_deq : T.k_deq;
    if (ids.empty() || !deq || w.empty() || bn > 256) {
        std::memset(partial, 0, (size_t) n_out * sizeof(float));
        return;
    }
    alignas(64) float tmp[256];
    const int n_own = (int) ids.size();
    const int x_step = (n_own > 1) ? (ids[1] - ids[0]) * bn : bn;
    const float * x0 = x + ids[0] * bn;
    const uint8_t * wrow = w.data();
    const size_t wrow_bytes = (size_t) n_own * bb;
    for (int r = 0; r < n_out; ++r) {
        float s = 0.f;
        const uint8_t * blk = wrow;
        const float * xp = x0;
        for (int i = 0; i < n_own; ++i) {
            deq(blk, tmp, (int64_t) bn);
            s += dot_f32(tmp, xp, bn);
            blk += bb;
            xp += x_step;
        }
        partial[r] = s;
        wrow += wrow_bytes;
    }
}

static void do_gemv(hga_l2_plan * p, int tid) {
    const ThreadLayer & T = p->tl[(size_t) p->gemv_layer][(size_t) tid];
    float * pk = p->partial_k[(size_t) tid].data();
    float * pv = p->partial_v[(size_t) tid].data();
    if (!T.k_ids.empty()) gemv_one_matrix(T, p->gemv_x, pk, p->n_out, false);
    else std::memset(pk, 0, (size_t) p->n_out * sizeof(float));
    if (!T.v_ids.empty()) gemv_one_matrix(T, p->gemv_x, pv, p->n_out, true);
    else std::memset(pv, 0, (size_t) p->n_out * sizeof(float));
}

static void do_reduce(hga_l2_plan * p, int tid) {
    const int n_out = p->n_out;
    const int n_th  = p->n_threads;
    const int i0 = tid * n_out / n_th;
    const int i1 = (tid + 1) * n_out / n_th;
    const int n = i1 - i0;
    if (n <= 0) return;
    float * kout = p->gemv_k + i0;
    float * vout = p->gemv_v + i0;
    std::memcpy(kout, p->partial_k[0].data() + i0, (size_t) n * sizeof(float));
    std::memcpy(vout, p->partial_v[0].data() + i0, (size_t) n * sizeof(float));
    for (int t = 1; t < n_th; ++t) {
        const float * pk = p->partial_k[(size_t) t].data() + i0;
        const float * pv = p->partial_v[(size_t) t].data() + i0;
        int i = 0;
#if HGA_L2_AVX512
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_ps(kout + i, _mm512_add_ps(_mm512_loadu_ps(kout + i),
                                                     _mm512_loadu_ps(pk + i)));
            _mm512_storeu_ps(vout + i, _mm512_add_ps(_mm512_loadu_ps(vout + i),
                                                     _mm512_loadu_ps(pv + i)));
        }
#endif
        for (; i < n; ++i) {
            kout[i] += pk[i];
            vout[i] += pv[i];
        }
    }
}

static void do_prefetch(hga_l2_plan * p, int tid) {
    const int nw = std::max(1, std::min(p->pref_workers, p->n_threads));
    int logical_tid = -1;
    for (int i = 0; i < nw; ++i) {
        /* Spread the small warmer team over the already socket-sorted list:
         * two workers become tids 0 and n_threads/2, one per socket. */
        if (tid == i * p->n_threads / nw) {
            logical_tid = i;
            break;
        }
    }
    if (logical_tid < 0) return;
    const int L = p->pref_layer;
    if (L < 0 || L >= p->n_layers) return;
    const ThreadLayer & T = p->tl[(size_t) L][(size_t) tid];
    if (p->pref_touch_weights) {
        touch_bytes(T.k_w.data(), T.k_w.size());
        touch_bytes(T.v_w.data(), T.v_w.size());
    }
    if (!p->pref_sess) return;
    /* Snapshot before touch: kick_prefetch may flip the other buffer. */
    const int * keys = p->pref_keys;
    const int n_keys = p->pref_n_keys;
    /* hga_touch_kv_tile already skips keys the next layer has not appended. */
    if (keys && n_keys > 0) {
        hga_touch_kv_tile(p->pref_sess, L, keys, n_keys, logical_tid, nw);
    } else {
        int cap = 2048;
        int win[2048];
        const int nkv = hga_session_n_kv(p->pref_sess, L);
        const int n = hga_window_keys(p->pref_sess, L, std::max(0, nkv - 1), win, cap);
        if (n > 0) hga_touch_kv_tile(p->pref_sess, L, win, n, logical_tid, nw);
    }
    if (p->pref_touch_summaries)
        hga_touch_summary_tile(p->pref_sess, L, logical_tid, nw);
}

static void run_serial(hga_l2_plan * p, int c) {
    for (int t = 0; t < p->n_threads; ++t) {
        if (c == CMD_BIND) do_bind(p, t);
        else if (c == CMD_GEMV) do_gemv(p, t);
        else if (c == CMD_PREFETCH) do_prefetch(p, t);
    }
    if (c == CMD_GEMV) {
        for (int t = 0; t < p->n_threads; ++t) do_reduce(p, t);
    }
}

#if defined(__linux__)
struct WorkerLaunch {
    hga_l2_plan * p;
    int tid;
};

static void * hga_l2_worker_main(void * argp) {
    WorkerLaunch * launch = (WorkerLaunch *) argp;
    hga_l2_plan * p = launch->p;
    const int tid = launch->tid;
    if (!p->cores.empty()) {
        pin_to_cpu(p->cores[(size_t) tid % p->cores.size()].cpu);
    }
    p->started.fetch_add(1, std::memory_order_release);

    int seen = 0;
    for (;;) {
        pthread_mutex_lock(&p->mu);
        int e;
        while ((e = p->epoch.load(std::memory_order_acquire)) == seen &&
               p->cmd.load(std::memory_order_relaxed) != CMD_STOP) {
            pthread_cond_wait(&p->cv, &p->mu);
        }
        seen = p->epoch.load(std::memory_order_relaxed);
        const int c = p->cmd.load(std::memory_order_relaxed);
        pthread_mutex_unlock(&p->mu);

        if (c == CMD_STOP) break;

        if (c == CMD_BIND) {
            do_bind(p, tid);
            p->done.fetch_add(1, std::memory_order_acq_rel);
        } else if (c == CMD_GEMV) {
            do_gemv(p, tid);
            const int after_slice = (int) p->done.fetch_add(1, std::memory_order_acq_rel) + 1;
            (void) after_slice;
            while (p->done.load(std::memory_order_acquire) < p->n_threads) {
#if HGA_L2_AVX512
                _mm_pause();
#endif
            }
            do_reduce(p, tid);
            p->done.fetch_add(1, std::memory_order_acq_rel);
        } else if (c == CMD_PREFETCH) {
            /* One-shot: do not busy-spin on all 40 cores through the GPU FFN. */
            do_prefetch(p, tid);
        }
    }
    return nullptr;
}

static void wake(hga_l2_plan * p, int c) {
    pthread_mutex_lock(&p->mu);
    p->cmd.store(c, std::memory_order_release);
    p->epoch.fetch_add(1, std::memory_order_acq_rel);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
}
#endif

hga_l2_plan * hga_l2_plan_create(int n_threads, int n_layers, int n_embd, int n_out) {
    auto * p = new hga_l2_plan();
    p->n_threads = std::max(1, n_threads);
    p->n_layers  = std::max(0, n_layers);
    p->n_embd    = n_embd;
    p->n_out     = n_out;
    p->n_blocks  = (n_embd > 0) ? n_embd / HGA_QK_K : 0;
    p->n_k_th = std::max(1, p->n_threads / 2);
    p->n_v_th = std::max(1, p->n_threads - p->n_k_th);
    if (p->n_threads == 1) {
        p->n_k_th = 1;
        p->n_v_th = 1;
    }
    p->cores = physical_cores();
    p->tl.assign((size_t) p->n_layers, std::vector<ThreadLayer>((size_t) p->n_threads));
    p->partial_k.assign((size_t) p->n_threads, std::vector<float>((size_t) std::max(1, n_out), 0.f));
    p->partial_v.assign((size_t) p->n_threads, std::vector<float>((size_t) std::max(1, n_out), 0.f));
    return p;
}

void hga_l2_plan_free(hga_l2_plan * p) {
    if (!p) return;
    hga_l2_plan_stop(p);
    delete p;
}

int hga_l2_plan_start(hga_l2_plan * p) {
    if (!p || p->running.load()) return p ? (p->running.load() ? p->n_threads : 0) : 0;
#if defined(__linux__)
    if (p->dma_fd < 0) {
        p->dma_fd = open("/dev/cpu_dma_latency", O_RDWR);
        if (p->dma_fd >= 0) {
            int32_t lat = 0;
            ssize_t wr = write(p->dma_fd, &lat, sizeof(lat));
            (void) wr;
        }
    }
    p->th.assign((size_t) p->n_threads, pthread_t{});
    p->started.store(0);
    p->running.store(1);
    /* Stack launch args: live until started==n_threads (worker copies tid first). */
    std::vector<WorkerLaunch> args((size_t) p->n_threads);
    for (int t = 0; t < p->n_threads; ++t) {
        args[(size_t) t] = {p, t};
        if (pthread_create(&p->th[(size_t) t], nullptr, hga_l2_worker_main, &args[(size_t) t]) != 0) {
            p->cmd.store(CMD_STOP);
            p->epoch.fetch_add(1);
            pthread_cond_broadcast(&p->cv);
            for (int u = 0; u < t; ++u) pthread_join(p->th[(size_t) u], nullptr);
            p->running.store(0);
            std::fprintf(stderr, "hga-l2: pthread_create failed at tid=%d; serial GEMV\n", t);
            return 0;
        }
    }
    while (p->started.load(std::memory_order_acquire) < p->n_threads) {
#if HGA_L2_AVX512
        _mm_pause();
#endif
    }
    static bool logged = false;
    if (!logged) {
        logged = true;
        const int pkg0 = p->cores.empty() ? -1 : p->cores.front().pkg;
        const int pkg1 = p->cores.empty() ? -1 : p->cores.back().pkg;
        std::fprintf(stderr,
            "hga-l2: decode plan  threads=%d  K/V blocks=%d+%d  n_k_th=%d n_v_th=%d  "
            "slice≈%d KiB  phys_cores=%zu  pkg=%d..%d  dma_latency=%d\n",
            p->n_threads, p->n_blocks, p->n_blocks, p->n_k_th, p->n_v_th,
            (int) ((p->n_out * (int) sizeof(hga_q4k_block)) / 1024),
            p->cores.size(), pkg0, pkg1, p->dma_fd >= 0 ? 1 : 0);
    }
    return p->n_threads;
#else
    (void) p;
    return 0;
#endif
}

void hga_l2_plan_stop(hga_l2_plan * p) {
    if (!p) return;
#if defined(__linux__)
    if (p->running.load()) {
        pthread_mutex_lock(&p->mu);
        p->cmd.store(CMD_STOP, std::memory_order_release);
        p->epoch.fetch_add(1, std::memory_order_acq_rel);
        pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mu);
        for (int t = 0; t < (int) p->th.size(); ++t) {
            if (p->th[(size_t) t]) pthread_join(p->th[(size_t) t], nullptr);
        }
        p->th.clear();
        p->running.store(0);
    }
    if (p->dma_fd >= 0) {
        close(p->dma_fd);
        p->dma_fd = -1;
    }
#endif
}

void hga_l2_bind_weights(hga_l2_plan * p, int layer,
                         const void * wk, size_t wk_row_bytes, size_t wk_blk_bytes,
                         int wk_blk_n, hga_l2_dequant_fn wk_dequant,
                         const void * wv, size_t wv_row_bytes, size_t wv_blk_bytes,
                         int wv_blk_n, hga_l2_dequant_fn wv_dequant) {
    if (!p || layer < 0 || layer >= p->n_layers) return;
    p->bind_layer  = layer;
    p->bind_wk     = (const uint8_t *) wk;
    p->bind_wv     = (const uint8_t *) wv;
    p->bind_wk_rb  = wk_row_bytes;
    p->bind_wv_rb  = wv_row_bytes;
    p->bind_wk_bb  = wk_blk_bytes;
    p->bind_wv_bb  = wv_blk_bytes;
    p->bind_wk_bn  = wk_blk_n > 0 ? wk_blk_n : HGA_QK_K;
    p->bind_wv_bn  = wv_blk_n > 0 ? wv_blk_n : HGA_QK_K;
    p->bind_wk_deq = wk_dequant ? wk_dequant : hga_q4k_dequant_block;
    p->bind_wv_deq = wv_dequant ? wv_dequant : hga_q4k_dequant_block;
#if defined(__linux__)
    if (p->running.load()) {
        p->done.store(0, std::memory_order_release);
        wake(p, CMD_BIND);
        while (p->done.load(std::memory_order_acquire) < p->n_threads) {
#if HGA_L2_AVX512
            _mm_pause();
#endif
        }
        return;
    }
#endif
    run_serial(p, CMD_BIND);
}

void hga_l2_gemv(hga_l2_plan * p, int layer, const float * x, float * k_out, float * v_out) {
    if (!p || !x || !k_out || !v_out || layer < 0 || layer >= p->n_layers) return;
    p->gemv_layer = layer;
    p->gemv_x = x;
    p->gemv_k = k_out;
    p->gemv_v = v_out;
#if defined(__linux__)
    if (p->running.load()) {
        p->done.store(0, std::memory_order_release);
        wake(p, CMD_GEMV);
        /* slice barrier + reduce: done goes to 2 * n_threads */
        while (p->done.load(std::memory_order_acquire) < 2 * p->n_threads) {
#if HGA_L2_AVX512
            _mm_pause();
#endif
        }
        return;
    }
#endif
    run_serial(p, CMD_GEMV);
}

void hga_l2_kick_prefetch(hga_l2_plan * p, int layer, const hga_session * sess,
                          const int * keys, int n_keys) {
    if (!p || layer < 0 || layer >= p->n_layers) return;
    p->pref_layer  = layer;
    p->pref_sess   = sess;
    p->pref_workers = p->n_threads;
    p->pref_touch_weights = true;
    p->pref_touch_summaries = false;
    /* Copy into the inactive buffer so in-flight do_prefetch still has a
     * stable pointer into last_keys / the previous buffer. */
    const int bi = 1 - p->pref_buf_i;
    if (keys && n_keys > 0) {
        p->pref_keys_buf[bi].assign(keys, keys + n_keys);
        p->pref_keys   = p->pref_keys_buf[bi].data();
        p->pref_n_keys = n_keys;
    } else {
        p->pref_keys_buf[bi].clear();
        p->pref_keys   = nullptr;
        p->pref_n_keys = 0;
    }
    p->pref_buf_i = bi;
#if defined(__linux__)
    if (p->running.load()) {
        wake(p, CMD_PREFETCH);
        return;
    }
#endif
    run_serial(p, CMD_PREFETCH);
}

void hga_l3_kick_kv_prefetch(hga_l2_plan * p, int layer,
                             const hga_session * sess,
                             const int * keys, int n_keys, int n_workers) {
    if (!p || !sess || layer < 0 || layer >= p->n_layers || !keys ||
        n_keys <= 0) return;
    p->pref_layer = layer;
    p->pref_sess = sess;
    p->pref_workers = std::max(1, std::min(n_workers, p->n_threads));
    p->pref_touch_weights = false;
    p->pref_touch_summaries = true;
    const int bi = 1 - p->pref_buf_i;
    p->pref_keys_buf[bi].assign(keys, keys + n_keys);
    p->pref_keys = p->pref_keys_buf[bi].data();
    p->pref_n_keys = n_keys;
    p->pref_buf_i = bi;
#if defined(__linux__)
    if (p->running.load()) {
        wake(p, CMD_PREFETCH);
        return;
    }
#endif
    run_serial(p, CMD_PREFETCH);
}

int hga_l2_n_threads(const hga_l2_plan * p) { return p ? p->n_threads : 0; }
int hga_l2_n_blocks(const hga_l2_plan * p) { return p ? p->n_blocks : 0; }

size_t hga_l2_slice_bytes(const hga_l2_plan * p, int layer, int tid) {
    if (!p || layer < 0 || layer >= p->n_layers || tid < 0 || tid >= p->n_threads) return 0;
    const ThreadLayer & T = p->tl[(size_t) layer][(size_t) tid];
    return T.k_w.size() + T.v_w.size();
}
