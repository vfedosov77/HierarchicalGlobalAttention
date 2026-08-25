#include "hga.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__AVX512F__)
#define HGA_AVX512 1
#else
#define HGA_AVX512 0
#endif

#if defined(__AVX512BW__)
#define HGA_AVX512BW 1
#else
#define HGA_AVX512BW 0
#endif

namespace {

constexpr float NEG = -1.0e4f;

static inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man = (man | 0x800000u) >> (1 - exp);
        return (uint16_t)(sign | (man >> 13));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t man  = h & 0x3ffu;
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

static inline float load_f(const void * p, hga_dtype dt, size_t i) {
    if (dt == HGA_F32) return ((const float *)p)[i];
    return f16_to_f32(((const uint16_t *)p)[i]);
}

#if HGA_AVX512
static inline float hga_dot_f32(const float * a, const float * b, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

static inline void hga_axpy_f32(float * acc, const float * v, float a, int n) {
    const __m512 as = _mm512_set1_ps(a);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(acc + i);
        x = _mm512_fmadd_ps(as, _mm512_loadu_ps(v + i), x);
        _mm512_storeu_ps(acc + i, x);
    }
    for (; i < n; ++i) acc[i] += a * v[i];
}

static inline void hga_scale_f32(float * acc, float s, int n) {
    const __m512 ss = _mm512_set1_ps(s);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(acc + i, _mm512_mul_ps(_mm512_loadu_ps(acc + i), ss));
    }
    for (; i < n; ++i) acc[i] *= s;
}

static inline float hga_dot_f16k(const float * q, const uint16_t * k, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i h = _mm256_loadu_si256((const __m256i *)(k + i));
        __m512  kv = _mm512_cvtph_ps(h);
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(q + i), kv, acc);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) s += q[i] * f16_to_f32(k[i]);
    return s;
}

static inline void hga_axpy_f16(float * acc, const uint16_t * v, float a, int n) {
    const __m512 as = _mm512_set1_ps(a);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i h = _mm256_loadu_si256((const __m256i *)(v + i));
        __m512  vv = _mm512_cvtph_ps(h);
        __m512  x  = _mm512_fmadd_ps(as, vv, _mm512_loadu_ps(acc + i));
        _mm512_storeu_ps(acc + i, x);
    }
    for (; i < n; ++i) acc[i] += a * f16_to_f32(v[i]);
}
#else
static inline float hga_dot_f32(const float * a, const float * b, int n) {
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
static inline void hga_axpy_f32(float * acc, const float * v, float a, int n) {
    for (int i = 0; i < n; ++i) acc[i] += a * v[i];
}
static inline void hga_scale_f32(float * acc, float s, int n) {
    for (int i = 0; i < n; ++i) acc[i] *= s;
}
static inline float hga_dot_f16k(const float * q, const uint16_t * k, int n) {
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += q[i] * f16_to_f32(k[i]);
    return s;
}
static inline void hga_axpy_f16(float * acc, const uint16_t * v, float a, int n) {
    for (int i = 0; i < n; ++i) acc[i] += a * f16_to_f32(v[i]);
}
#endif

/* Symmetric INT8, clamp to [-127, 127] so maddubs abs never hits -128. */
static void quant_vec_i8(const float * x, int n, int8_t * q, float * scale) {
    float amax = 0.f;
    int i = 0;
#if HGA_AVX512
    __m512 vmax = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        vmax = _mm512_max_ps(vmax, _mm512_abs_ps(_mm512_loadu_ps(x + i)));
    }
    amax = _mm512_reduce_max_ps(vmax);
#endif
    for (; i < n; ++i) amax = std::max(amax, std::fabs(x[i]));
    if (amax < 1e-12f) {
        std::memset(q, 0, (size_t)n);
        *scale = 0.f;
        return;
    }
    *scale = amax / 127.f;
    const float inv = 127.f / amax;
    i = 0;
#if HGA_AVX512
    const __m512 vinv = _mm512_set1_ps(inv);
    const __m512 vlo  = _mm512_set1_ps(-127.f);
    const __m512 vhi  = _mm512_set1_ps(127.f);
    for (; i + 16 <= n; i += 16) {
        __m512 y = _mm512_mul_ps(_mm512_loadu_ps(x + i), vinv);
        y = _mm512_min_ps(vhi, _mm512_max_ps(vlo, y));
        __m128i packed = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(y));
        _mm_storeu_si128((__m128i *)(q + i), packed);
    }
#endif
    for (; i < n; ++i) {
        int v = (int)std::lrintf(x[i] * inv);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q[i] = (int8_t)v;
    }
}

static inline int32_t hga_dot_i8(const int8_t * a, const int8_t * b, int n) {
    int32_t sum = 0;
    int i = 0;
#if HGA_AVX512BW
    __m512i acc = _mm512_setzero_si512();
    for (; i + 64 <= n; i += 64) {
        const __m512i va = _mm512_loadu_si512((const void *)(a + i));
        const __m512i vb = _mm512_loadu_si512((const void *)(b + i));
        const __m512i pa = _mm512_madd_epi16(
            _mm512_cvtepi8_epi16(_mm512_castsi512_si256(va)),
            _mm512_cvtepi8_epi16(_mm512_castsi512_si256(vb)));
        const __m512i pb = _mm512_madd_epi16(
            _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(va, 1)),
            _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vb, 1)));
        acc = _mm512_add_epi32(acc, pa);
        acc = _mm512_add_epi32(acc, pb);
    }
    sum += _mm512_reduce_add_epi32(acc);
#endif
#if defined(__AVX2__)
    for (; i + 32 <= n; i += 32) {
        const __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        const __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        /* signed×signed via maddubs + sign, no VNNI (Skylake-SP). */
        const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(va, va),
                                                 _mm256_sign_epi8(vb, va));
        const __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(p32),
                                  _mm256_extracti128_si256(p32, 1));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
        sum += _mm_cvtsi128_si32(s);
    }
#endif
    for (; i < n; ++i) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

static inline void hga_axpy_i8(float * acc, const int8_t * v, float a, int n) {
    int i = 0;
#if HGA_AVX512
    const __m512 as = _mm512_set1_ps(a);
    for (; i + 16 <= n; i += 16) {
        const __m128i v8 = _mm_loadu_si128((const __m128i *)(v + i));
        const __m512 vf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v8));
        _mm512_storeu_ps(acc + i, _mm512_fmadd_ps(as, vf, _mm512_loadu_ps(acc + i)));
    }
#endif
    for (; i < n; ++i) acc[i] += a * (float)v[i];
}

static inline double now_ms() {
#if defined(_OPENMP)
    return omp_get_wtime() * 1e3;
#else
    return 0.0;
#endif
}

struct Span {
    int start;
    int len;
};

struct Layer {
    int n_kv = 0;
    int n_closed = 0;
    std::vector<uint16_t> k;      /* F16 [kvh, max_seq, dh] */
    std::vector<uint16_t> v;
    std::vector<uint16_t> k_raw;  /* pre-RoPE F16 (summaries) */
    std::vector<int8_t> k8;       /* I8  [kvh, max_seq, dh] */
    std::vector<int8_t> v8;
    std::vector<float> k_scale;   /* [kvh, max_seq] */
    std::vector<float> v_scale;
    std::vector<float> chunk_k;   /* [kvh, max_chunks, dh] */
    std::vector<float> group_k;   /* [kvh, max_chunks, G, dh] */
};

struct RopeTables {
    int rotary_dim = 0;
    int cutoff_pair = 0;
    std::vector<float> inv_freq; /* [rotary_dim/2] */
};

static int mixed_cutoff_pair(const hga_config & c) {
    const int half = c.rotary_dim / 2;
    int cutoff = 0;
    const float span = (float)std::max(1, c.chunk_size - 1);
    for (int i = 0; i < half; ++i) {
        const float inv = 1.f / std::pow(c.theta, (float)i / (float)half);
        if (span * inv > c.mixed_rope_threshold) cutoff = i + 1;
    }
    return cutoff;
}

static RopeTables make_rope(const hga_config & c) {
    RopeTables t;
    t.rotary_dim = c.rotary_dim;
    t.cutoff_pair = mixed_cutoff_pair(c);
    const int half = c.rotary_dim / 2;
    t.inv_freq.resize((size_t)half);
    for (int i = 0; i < half; ++i) {
        t.inv_freq[(size_t)i] = 1.f / std::pow(c.theta, (float)i / (float)half);
    }
    return t;
}

/* Match ChunkRouter._apply_partial_rotary: x1,x2 = x_rot[:half], x_rot[half:];
   rotated = x_rot * cos + cat(-x2, x1) * sin */
static void apply_partial_rotary_hf(float * x, int rd, const float * cos, const float * sin) {
    const int half = rd / 2;
    std::vector<float> out((size_t)rd);
    for (int i = 0; i < half; ++i) {
        const float x1 = x[i];
        const float x2 = x[half + i];
        out[(size_t)i]        = x1 * cos[i] + (-x2) * sin[i];
        out[(size_t)half + i] = x2 * cos[half + i] + x1 * sin[half + i];
    }
    for (int i = 0; i < rd; ++i) x[i] = out[(size_t)i];
}

static void rope_cos_sin(const RopeTables & r, int pos, float * cos, float * sin) {
    const int half = r.rotary_dim / 2;
    for (int i = 0; i < half; ++i) {
        const float ang = (float)pos * r.inv_freq[(size_t)i];
        const float c = std::cos(ang);
        const float s = std::sin(ang);
        cos[i] = c; cos[half + i] = c;
        sin[i] = s; sin[half + i] = s;
    }
}

static void mix_tokenwise_anchor(float * dst, const float * tokenwise, const float * anchor,
                                 int dh, int rd, int cutoff_pair) {
    const int half = rd / 2;
    for (int i = 0; i < dh; ++i) {
        bool use_tw = true;
        if (i < rd) {
            const int pair = (i < half) ? i : (i - half);
            use_tw = pair < cutoff_pair;
        }
        dst[i] = use_tw ? tokenwise[i] : anchor[i];
    }
}

/* Mean-pool mixed-RoPE summary over `len` tokens starting at `base` for one kv head.
 * Post-RoPE K is either F16 (k_rope_h) or INT8 (k8_h + k_sc_h). */
static void rope_summary(const hga_config & c, const RopeTables & rope,
                         const uint16_t * k_raw_h, const uint16_t * k_rope_h,
                         const int8_t * k8_h, const float * k_sc_h, bool use_i8,
                         int base, int len, int anchor_pos, float scale,
                         int max_seq, float * out) {
    const int dh = c.head_dim;
    const int rd = c.rotary_dim;
    std::vector<float> raw_sum((size_t)dh, 0.f), tok_sum((size_t)dh, 0.f);
    for (int t = 0; t < len; ++t) {
        const uint16_t * raw = k_raw_h + (size_t)(base + t) * (size_t)dh;
        for (int d = 0; d < dh; ++d) raw_sum[(size_t)d] += f16_to_f32(raw[d]);
        if (use_i8) {
            const int8_t * rp = k8_h + (size_t)(base + t) * (size_t)dh;
            const float sc = k_sc_h[base + t];
            for (int d = 0; d < dh; ++d) tok_sum[(size_t)d] += (float)rp[d] * sc;
        } else {
            const uint16_t * rp = k_rope_h + (size_t)(base + t) * (size_t)dh;
            for (int d = 0; d < dh; ++d) tok_sum[(size_t)d] += f16_to_f32(rp[d]);
        }
    }
    for (int d = 0; d < dh; ++d) {
        raw_sum[(size_t)d] *= scale;
        tok_sum[(size_t)d] *= scale;
    }
    std::vector<float> cos((size_t)rd), sin((size_t)rd), endpoint = raw_sum;
    rope_cos_sin(rope, anchor_pos, cos.data(), sin.data());
    apply_partial_rotary_hf(endpoint.data(), rd, cos.data(), sin.data());
    mix_tokenwise_anchor(out, tok_sum.data(), endpoint.data(), dh, rd, rope.cutoff_pair);
    (void)max_seq;
}

static size_t kv_index(int kvh, int t, int dh, int max_seq) {
    return ((size_t)kvh * (size_t)max_seq + (size_t)t) * (size_t)dh;
}

static void topk_idx(const float * scores, int n, int k, int * out) {
    k = std::min(k, n);
    if (k <= 0) return;
    std::vector<int> idx((size_t)n);
    for (int i = 0; i < n; ++i) idx[(size_t)i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return scores[a] > scores[b]; });
    for (int i = 0; i < k; ++i) out[i] = idx[(size_t)i];
}

} /* namespace */

struct hga_session {
    hga_config cfg{};
    int n_layers = 0;
    int max_chunks = 0;
    int groups_per_chunk = 0;
    float scale = 1.f;
    float group_scale = 1.f;
    int ubatch_start = 0;
    int ubatch_n = 0;
    RopeTables rope;
    std::vector<Layer> layers;
};

hga_config hga_config_qwen38_27b(int levels, int max_seq, int n_threads) {
    hga_config c{};
    c.n_q_heads = 24;
    c.n_kv_heads = 4;
    c.head_dim = 256;
    c.rotary_dim = 64;
    c.chunk_size = 64;
    c.group_size = 16;
    c.keep_first = 2;
    c.keep_last = 8;
    c.levels = (levels == 2) ? 2 : 1;
    c.frac_l1 = 0.08f;
    c.frac_l2 = 0.04f;
    c.theta = 1.0e6f;
    c.mixed_rope_threshold = 0.5f;
    c.n_threads = n_threads > 0 ? n_threads : 1;
    c.max_seq = max_seq > 0 ? max_seq : 32768;
    c.prec = HGA_PREC_I8;
    return c;
}

int hga_topk_chunks(const hga_config * cfg, int n_closed) {
    if (n_closed <= 0) return 0;
    const int C = cfg->chunk_size;
    const int n_kv = n_closed * C;
    const int n_win = std::min(cfg->keep_first + cfg->keep_last, n_closed);
    const int n_mid = n_closed - n_win;
    if (n_mid <= 0) return 0;
    /* 8 % of tokens, minus the always-on windows, in whole chunks. */
    int want = (int)std::lround((double)cfg->frac_l1 * (double)n_kv / (double)C) - n_win;
    if (want < 1 && n_mid > 0) want = 1;
    if (want < 0) want = 0;
    return std::min(want, n_mid);
}

int hga_topk_groups(const hga_config * cfg, int n_closed, int topk_chunks) {
    const int G = cfg->chunk_size / cfg->group_size;
    if (topk_chunks <= 0) return 0;
    if (cfg->levels != 2) return topk_chunks * G; /* one-level: open every group */
    /* Target ~4 % tokens including windows, but never fewer than one group per
     * routed chunk so the extra hierarchy always has work to do. */
    const int n_kv = n_closed * cfg->chunk_size;
    const int n_win_groups = std::min(cfg->keep_first + cfg->keep_last, n_closed) * G;
    int want = (int)std::lround((double)cfg->frac_l2 * (double)n_kv / (double)cfg->group_size) - n_win_groups;
    const int floor_g = topk_chunks; /* ≥1 group in each selected chunk */
    const int cap = topk_chunks * G;
    want = std::max(want, floor_g);
    return std::min(std::max(want, 1), cap);
}

hga_session * hga_session_create(const hga_config * cfg, int n_layers) {
    auto * s = new hga_session();
    s->cfg = *cfg;
    if (s->cfg.group_size <= 0 || s->cfg.chunk_size % s->cfg.group_size != 0) {
        s->cfg.group_size = s->cfg.chunk_size;
    }
    s->n_layers = n_layers;
    s->groups_per_chunk = s->cfg.chunk_size / s->cfg.group_size;
    s->max_chunks = (s->cfg.max_seq + s->cfg.chunk_size - 1) / s->cfg.chunk_size;
    s->scale = 1.f / std::sqrt((float)s->cfg.head_dim);
    s->group_scale = 1.f / (s->cfg.group_size + std::sqrt((float)s->cfg.group_size));
    s->rope = make_rope(s->cfg);
    s->layers.resize((size_t)n_layers);
    const int kvh = s->cfg.n_kv_heads;
    const int dh = s->cfg.head_dim;
    const int ms = s->cfg.max_seq;
    const int G = s->groups_per_chunk;
    const bool i8 = s->cfg.prec == HGA_PREC_I8;
    for (auto & L : s->layers) {
        if (i8) {
            L.k8.assign((size_t)kvh * (size_t)ms * (size_t)dh, 0);
            L.v8.assign((size_t)kvh * (size_t)ms * (size_t)dh, 0);
            L.k_scale.assign((size_t)kvh * (size_t)ms, 0.f);
            L.v_scale.assign((size_t)kvh * (size_t)ms, 0.f);
        } else {
            L.k.assign((size_t)kvh * (size_t)ms * (size_t)dh, 0);
            L.v.assign((size_t)kvh * (size_t)ms * (size_t)dh, 0);
        }
        L.k_raw.assign((size_t)kvh * (size_t)ms * (size_t)dh, 0);
        L.chunk_k.assign((size_t)kvh * (size_t)s->max_chunks * (size_t)dh, 0.f);
        L.group_k.assign((size_t)kvh * (size_t)s->max_chunks * (size_t)G * (size_t)dh, 0.f);
    }
    return s;
}

void hga_session_free(hga_session * s) { delete s; }

void hga_session_reset(hga_session * s) {
    for (auto & L : s->layers) {
        L.n_kv = 0;
        L.n_closed = 0;
    }
}

const hga_config * hga_session_config(const hga_session * s) { return &s->cfg; }
int hga_session_n_layers(const hga_session * s) { return s->n_layers; }
int hga_session_n_kv(const hga_session * s, int layer) { return s->layers[(size_t)layer].n_kv; }

void hga_set_ubatch(hga_session * s, int start_pos, int n_tokens) {
    s->ubatch_start = start_pos;
    s->ubatch_n = n_tokens;
}
int hga_ubatch_start(const hga_session * s) { return s->ubatch_start; }
int hga_ubatch_n(const hga_session * s) { return s->ubatch_n; }

static void close_chunk_if_full(hga_session * s, Layer & L, int layer) {
    (void)layer;
    const int C = s->cfg.chunk_size;
    if (L.n_kv % C != 0 || L.n_kv == 0) return;
    const int n = L.n_kv / C - 1; /* just-closed chunk id */
    if (n < L.n_closed) return;
    if (n >= s->max_chunks) return;
    const int kvh = s->cfg.n_kv_heads;
    const int dh = s->cfg.head_dim;
    const int gs = s->cfg.group_size;
    const int G = s->groups_per_chunk;
    const int ms = s->cfg.max_seq;
    const int base = n * C;
    const bool i8 = s->cfg.prec == HGA_PREC_I8;
    for (int h = 0; h < kvh; ++h) {
        const uint16_t * kr = i8 ? nullptr : L.k.data() + kv_index(h, 0, dh, ms);
        const int8_t * k8 = i8 ? L.k8.data() + kv_index(h, 0, dh, ms) : nullptr;
        const float * ksc = i8 ? L.k_scale.data() + (size_t)h * (size_t)ms : nullptr;
        const uint16_t * raw = L.k_raw.data() + kv_index(h, 0, dh, ms);
        float * ck = L.chunk_k.data() + ((size_t)h * (size_t)s->max_chunks + (size_t)n) * (size_t)dh;
        rope_summary(s->cfg, s->rope, raw, kr, k8, ksc, i8, base, C, base + (C - 1) / 2, 1.f, ms, ck);
        for (int g = 0; g < G; ++g) {
            float * gk = L.group_k.data() +
                ((((size_t)h * (size_t)s->max_chunks + (size_t)n) * (size_t)G) + (size_t)g) * (size_t)dh;
            rope_summary(s->cfg, s->rope, raw, kr, k8, ksc, i8, base + g * gs, gs,
                         base + g * gs + (gs - 1) / 2, s->group_scale, ms, gk);
        }
    }
    L.n_closed = n + 1;
}

void hga_append(hga_session * s, int layer, int start_pos, int n_new,
                const void * k_rope, const void * k_raw, const void * v, hga_dtype dtype) {
    Layer & L = s->layers[(size_t)layer];
    const int kvh = s->cfg.n_kv_heads;
    const int dh = s->cfg.head_dim;
    const int ms = s->cfg.max_seq;
    if (L.n_kv != start_pos) {
        /* Allow append only in order. If start_pos < n_kv this is a rewind: truncate. */
        if (start_pos < L.n_kv) {
            L.n_kv = start_pos;
            L.n_closed = start_pos / s->cfg.chunk_size;
        }
    }
    const bool i8 = s->cfg.prec == HGA_PREC_I8;
    std::vector<float> tmpk((size_t)dh), tmpv((size_t)dh);
    for (int t = 0; t < n_new; ++t) {
        const int pos = start_pos + t;
        if (pos >= ms) break;
        for (int h = 0; h < kvh; ++h) {
            const size_t src = ((size_t)h * (size_t)n_new + (size_t)t) * (size_t)dh;
            uint16_t * rd = L.k_raw.data() + kv_index(h, pos, dh, ms);
            for (int d = 0; d < dh; ++d) {
                tmpk[(size_t)d] = load_f(k_rope, dtype, src + (size_t)d);
                tmpv[(size_t)d] = load_f(v, dtype, src + (size_t)d);
                const float rf = k_raw ? load_f(k_raw, dtype, src + (size_t)d) : tmpk[(size_t)d];
                rd[d] = f32_to_f16(rf);
            }
            if (i8) {
                quant_vec_i8(tmpk.data(), dh, L.k8.data() + kv_index(h, pos, dh, ms),
                             &L.k_scale[(size_t)h * (size_t)ms + (size_t)pos]);
                quant_vec_i8(tmpv.data(), dh, L.v8.data() + kv_index(h, pos, dh, ms),
                             &L.v_scale[(size_t)h * (size_t)ms + (size_t)pos]);
            } else {
                uint16_t * kd = L.k.data() + kv_index(h, pos, dh, ms);
                uint16_t * vd = L.v.data() + kv_index(h, pos, dh, ms);
                for (int d = 0; d < dh; ++d) {
                    kd[d] = f32_to_f16(tmpk[(size_t)d]);
                    vd[d] = f32_to_f16(tmpv[(size_t)d]);
                }
            }
        }
        L.n_kv = pos + 1;
    }
}

void hga_close_full_chunks(hga_session * s, int layer) {
    close_chunk_if_full(s, s->layers[(size_t)layer], layer);
}

struct RouteSet {
    std::vector<int> chunks; /* absolute closed-chunk ids, routed middle */
    std::vector<int> g_chunk;
    std::vector<int> g_id;
};

static void route_layer(const hga_session * s, const Layer & L, const float * q_pool /* [n_q_heads, dh] */,
                        int n_closed_view, RouteSet & rs, int * n_sel_chunks, int * n_open_groups) {
    const hga_config & c = s->cfg;
    const int H = c.n_q_heads;
    const int kvh = c.n_kv_heads;
    const int dh = c.head_dim;
    const int C = c.chunk_size;
    const int G = s->groups_per_chunk;
    const int rep = H / kvh;
    const int n_closed = std::min(n_closed_view, L.n_closed);
    const int f_hi = std::min(c.keep_first, n_closed);
    const int l_lo = std::max(f_hi, n_closed - c.keep_last);
    const int mid_lo = f_hi;
    const int mid_hi = l_lo;
    const int n_mid = std::max(0, mid_hi - mid_lo);
    const int Kc = std::min(hga_topk_chunks(&c, n_closed), n_mid);
    rs.chunks.clear();
    rs.g_chunk.clear();
    rs.g_id.clear();
    *n_sel_chunks = Kc;
    *n_open_groups = 0;
    if (Kc <= 0 || n_mid <= 0) return;

    /* Pool scores over query heads that share a kv head (max). */
    std::vector<float> sc((size_t)H * (size_t)n_mid, 0.f);
    for (int h = 0; h < H; ++h) {
        const int kh = h / rep;
        const float * qh = q_pool + (size_t)h * (size_t)dh;
        for (int m = 0; m < n_mid; ++m) {
            const float * ck = L.chunk_k.data() +
                ((size_t)kh * (size_t)s->max_chunks + (size_t)(mid_lo + m)) * (size_t)dh;
            sc[(size_t)h * (size_t)n_mid + (size_t)m] = hga_dot_f32(qh, ck, dh) * s->scale;
        }
    }
    /* Union of per-head top-k (GQA: take max over heads of each kv group, then top-k). */
    std::vector<float> pooled((size_t)n_mid, -1e30f);
    for (int h = 0; h < H; ++h) {
        for (int m = 0; m < n_mid; ++m) {
            pooled[(size_t)m] = std::max(pooled[(size_t)m], sc[(size_t)h * (size_t)n_mid + (size_t)m]);
        }
    }
    std::vector<int> rel((size_t)Kc);
    topk_idx(pooled.data(), n_mid, Kc, rel.data());
    /* Force-include previous chunk if it sits in the mid pool. */
    const int prev = n_closed - 1;
    if (mid_lo <= prev && prev < mid_hi) {
        const int prev_rel = prev - mid_lo;
        bool have = false;
        int worst = 0;
        for (int i = 0; i < Kc; ++i) {
            if (rel[(size_t)i] == prev_rel) have = true;
            if (pooled[(size_t)rel[(size_t)i]] < pooled[(size_t)rel[(size_t)worst]]) worst = i;
        }
        if (!have) rel[(size_t)worst] = prev_rel;
    }
    rs.chunks.resize((size_t)Kc);
    for (int i = 0; i < Kc; ++i) rs.chunks[(size_t)i] = mid_lo + rel[(size_t)i];

    const int Kg = hga_topk_groups(&c, n_closed, Kc);
    *n_open_groups = Kg;
    if (Kg <= 0) return;

    const int n_cand = Kc * G;
    std::vector<float> gsc((size_t)n_cand, -1e30f);
    for (int h = 0; h < H; ++h) {
        const int kh = h / rep;
        const float * qh = q_pool + (size_t)h * (size_t)dh;
        for (int i = 0; i < Kc; ++i) {
            const int cid = rs.chunks[(size_t)i];
            for (int g = 0; g < G; ++g) {
                const float * gk = L.group_k.data() +
                    ((((size_t)kh * (size_t)s->max_chunks + (size_t)cid) * (size_t)G) + (size_t)g) * (size_t)dh;
                const float s_g = hga_dot_f32(qh, gk, dh) * s->scale;
                float & slot = gsc[(size_t)i * (size_t)G + (size_t)g];
                slot = std::max(slot, s_g);
            }
        }
    }
    std::vector<int> grel((size_t)Kg);
    topk_idx(gsc.data(), n_cand, Kg, grel.data());
    rs.g_chunk.resize((size_t)Kg);
    rs.g_id.resize((size_t)Kg);
    for (int i = 0; i < Kg; ++i) {
        const int t = grel[(size_t)i];
        rs.g_chunk[(size_t)i] = rs.chunks[(size_t)(t / G)];
        rs.g_id[(size_t)i] = t % G;
    }
}

static void collect_spans(const hga_session * s, const Layer & L, const RouteSet & rs,
                          int n_closed_view, std::vector<Span> & spans) {
    const hga_config & c = s->cfg;
    const int C = c.chunk_size;
    const int gs = c.group_size;
    const int n_closed = std::min(n_closed_view, L.n_closed);
    const int f_hi = std::min(c.keep_first, n_closed);
    const int l_lo = std::max(f_hi, n_closed - c.keep_last);
    spans.clear();
    if (f_hi > 0) spans.push_back({0, f_hi * C});
    if (l_lo < n_closed) spans.push_back({l_lo * C, (n_closed - l_lo) * C});
    if (c.levels == 2) {
        for (size_t i = 0; i < rs.g_chunk.size(); ++i) {
            const int cid = rs.g_chunk[i];
            const int gid = rs.g_id[i];
            if (cid < f_hi || cid >= l_lo) continue; /* already in a window */
            spans.push_back({cid * C + gid * gs, gs});
        }
    } else {
        for (int cid : rs.chunks) {
            if (cid < f_hi || cid >= l_lo) continue;
            spans.push_back({cid * C, C});
        }
    }
}

/* Flatten attended positions (windows + routed + active) into one key list. */
static void collect_keys(const hga_session * s, const Layer & L, const std::vector<Span> & spans,
                         int q_hi, std::vector<int> & keys) {
    keys.clear();
    for (const Span & sp : spans) {
        const int hi = std::min(sp.start + sp.len - 1, q_hi);
        for (int j = sp.start; j <= hi; ++j) keys.push_back(j);
    }
    const int act0 = L.n_closed * s->cfg.chunk_size;
    for (int j = act0; j < L.n_kv && j <= q_hi; ++j) keys.push_back(j);
}

/* 2D flash tiles so Q-tile + K-tile + acc fit in one core's L2 (~1 MB on 6148).
 * Default on 40 threads / 4 KV heads: 1/2 Q × 1/4 K → 8 tiles × 4 heads = 32 cores. */
static void pick_qk_tiles(int n_kvh, int n_q_rows, int n_keys, int dh,
                          int q_bpe, int kv_bpe, int n_threads, int * nq_tiles, int * nk_tiles) {
    const size_t l2 = 768u * 1024u;
    n_kvh = std::max(1, n_kvh);
    n_q_rows = std::max(1, n_q_rows);
    n_keys = std::max(1, n_keys);
    n_threads = std::max(1, n_threads);

    auto bytes = [&](int nqt, int nkt) -> size_t {
        const int nr = (n_q_rows + nqt - 1) / nqt;
        const int nk = (n_keys + nkt - 1) / nkt;
        return (size_t)nr * (size_t)dh * (size_t)(q_bpe + 4)
             + (size_t)nk * (size_t)dh * 2u * (size_t)kv_bpe;
    };
    auto ok = [&](int nqt, int nkt) {
        return nqt >= 1 && nkt >= 1 && nqt <= n_q_rows && nkt <= n_keys
            && (size_t)n_kvh * (size_t)nqt * (size_t)nkt <= (size_t)n_threads
            && bytes(nqt, nkt) <= l2;
    };

    /* Preferred: 2 Q-tiles × 4 K-tiles per KV head (32 cores when n_kvh=4). */
    if (ok(2, 4)) { *nq_tiles = 2; *nk_tiles = 4; }
    else {
        int nqt = 1, nkt = 1;
        while (!ok(nqt, nkt) || bytes(nqt, nkt) > l2) {
            const int nr = (n_q_rows + nqt - 1) / nqt;
            const int nk = (n_keys + nkt - 1) / nkt;
            const size_t qacc = (size_t)nr * dh * (size_t)(q_bpe + 4);
            const size_t kv = (size_t)nk * dh * 2u * (size_t)kv_bpe;
            if (kv >= qacc && nkt < n_keys) ++nkt;
            else if (nqt < n_q_rows) ++nqt;
            else if (nkt < n_keys) ++nkt;
            else break;
            if (n_kvh * nqt * nkt > n_threads) {
                if (nkt > 1) --nkt;
                else if (nqt > 1) --nqt;
                break;
            }
        }
        *nq_tiles = nqt;
        *nk_tiles = nkt;
    }

    if (n_q_rows >= 64 && n_keys >= 64) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            std::fprintf(stderr, "hga L2 tiles: kv_heads=%d  q_tiles=%d  k_tiles=%d  tasks=%d  threads=%d  (want ~32 on 40-core)\n",
                         n_kvh, *nq_tiles, *nk_tiles, n_kvh * (*nq_tiles) * (*nk_tiles), n_threads);
        }
    }

    /* Use leftover cores: split the heavier side until we fill n_threads. */
    for (;;) {
        const int nqt = *nq_tiles, nkt = *nk_tiles;
        const int nr = (n_q_rows + nqt - 1) / nqt;
        const int nk = (n_keys + nkt - 1) / nkt;
        const size_t qacc = (size_t)nr * dh * (size_t)(q_bpe + 4);
        const size_t kv = (size_t)nk * dh * 2u * (size_t)kv_bpe;
        if (n_kvh * (nqt + 1) * nkt <= n_threads && nqt < n_q_rows && qacc >= kv && ok(nqt + 1, nkt))
            *nq_tiles = nqt + 1;
        else if (n_kvh * nqt * (nkt + 1) <= n_threads && nkt < n_keys && ok(nqt, nkt + 1))
            *nk_tiles = nkt + 1;
        else if (n_kvh * (nqt + 1) * nkt <= n_threads && nqt < n_q_rows && ok(nqt + 1, nkt))
            *nq_tiles = nqt + 1;
        else
            break;
    }
}

static void merge_online_softmax(float * m, float * lse, float * acc,
                                 const float * m2, const float * lse2, const float * acc2,
                                 int n_rows, int dh) {
    const float ninf = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n_rows; ++i) {
        const float a = m[i], b = m2[i];
        if (b == ninf) continue;
        float * o = acc + (size_t)i * (size_t)dh;
        if (a == ninf) {
            m[i] = b;
            lse[i] = lse2[i];
            std::memcpy(o, acc2 + (size_t)i * (size_t)dh, (size_t)dh * sizeof(float));
            continue;
        }
        const float mx = std::max(a, b);
        const float ea = std::exp(a - mx);
        const float eb = std::exp(b - mx);
        hga_scale_f32(o, ea, dh);
        hga_axpy_f32(o, acc2 + (size_t)i * (size_t)dh, eb, dh);
        lse[i] = lse[i] * ea + lse2[i] * eb;
        m[i] = mx;
    }
}

static void flash_tile_f16(const hga_session * s, const Layer & L, int kh,
                           int start_pos, int n_q, int h0, int h1, int t0, int t1,
                           const int * keys, int k0, int k1, const float * qf,
                           float * m, float * lse, float * acc) {
    const int dh = s->cfg.head_dim;
    const int ms = s->cfg.max_seq;
    const uint16_t * khp = L.k.data() + kv_index(kh, 0, dh, ms);
    const uint16_t * vhp = L.v.data() + kv_index(kh, 0, dh, ms);
    const int n_t = t1 - t0;
    const int n_rows = (h1 - h0) * n_t;
    const float ninf = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n_rows; ++i) { m[i] = ninf; lse[i] = 0.f; }
    std::memset(acc, 0, (size_t)n_rows * (size_t)dh * sizeof(float));

    for (int ki = k0; ki < k1; ++ki) {
        const int j = keys[ki];
        const uint16_t * k = khp + (size_t)j * (size_t)dh;
        const uint16_t * v = vhp + (size_t)j * (size_t)dh;
        int li = 0;
        for (int h = h0; h < h1; ++h) {
            for (int t = t0; t < t1; ++t, ++li) {
                if (j > start_pos + t) continue;
                const float * q = qf + ((size_t)h * (size_t)n_q + (size_t)t) * (size_t)dh;
                float * o = acc + (size_t)li * (size_t)dh;
                const float score = hga_dot_f16k(q, k, dh) * s->scale;
                const float m2 = std::max(m[li], score);
                const float e1 = (m[li] == ninf) ? 0.f : std::exp(m[li] - m2);
                const float e2 = std::exp(score - m2);
                hga_scale_f32(o, e1, dh);
                hga_axpy_f16(o, v, e2, dh);
                lse[li] = lse[li] * e1 + e2;
                m[li] = m2;
            }
        }
    }
}

static void flash_tile_i8(const hga_session * s, const Layer & L, int kh,
                          int start_pos, int n_q, int h0, int h1, int t0, int t1,
                          const int * keys, int k0, int k1, const int8_t * q8,
                          const float * qsc, float * m, float * lse, float * acc) {
    const int dh = s->cfg.head_dim;
    const int ms = s->cfg.max_seq;
    const int8_t * khp = L.k8.data() + kv_index(kh, 0, dh, ms);
    const int8_t * vhp = L.v8.data() + kv_index(kh, 0, dh, ms);
    const float * ksc = L.k_scale.data() + (size_t)kh * (size_t)ms;
    const float * vsc = L.v_scale.data() + (size_t)kh * (size_t)ms;
    const int n_t = t1 - t0;
    const int n_rows = (h1 - h0) * n_t;
    const float ninf = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n_rows; ++i) { m[i] = ninf; lse[i] = 0.f; }
    std::memset(acc, 0, (size_t)n_rows * (size_t)dh * sizeof(float));

    for (int ki = k0; ki < k1; ++ki) {
        const int j = keys[ki];
        const int8_t * k = khp + (size_t)j * (size_t)dh;
        const int8_t * v = vhp + (size_t)j * (size_t)dh;
        const float ks = ksc[j];
        const float vs = vsc[j];
        int li = 0;
        for (int h = h0; h < h1; ++h) {
            const float * qsc_h = qsc + (size_t)h * (size_t)n_q;
            const int8_t * q8_h = q8 + (size_t)h * (size_t)n_q * (size_t)dh;
            for (int t = t0; t < t1; ++t, ++li) {
                if (j > start_pos + t) continue;
                float * o = acc + (size_t)li * (size_t)dh;
                const float score = (float)hga_dot_i8(q8_h + (size_t)t * (size_t)dh, k, dh)
                                    * qsc_h[t] * ks * s->scale;
                const float m2 = std::max(m[li], score);
                const float e1 = (m[li] == ninf) ? 0.f : std::exp(m[li] - m2);
                const float e2 = std::exp(score - m2);
                hga_scale_f32(o, e1, dh);
                hga_axpy_i8(o, v, e2 * vs, dh);
                lse[li] = lse[li] * e1 + e2;
                m[li] = m2;
            }
        }
    }
}

void hga_attend(hga_session * s, int layer, int start_pos, int n_q,
                const void * q, hga_dtype q_dtype, float * out, hga_stats * stats) {
    Layer & L = s->layers[(size_t)layer];
    const int H = s->cfg.n_q_heads;
    const int dh = s->cfg.head_dim;
    const double t0 = now_ms();

    /* Mean-pool queries in this block for routing (chunk-shared). */
    std::vector<float> qf((size_t)H * (size_t)n_q * (size_t)dh);
    for (int h = 0; h < H; ++h) {
        for (int t = 0; t < n_q; ++t) {
            for (int d = 0; d < dh; ++d) {
                qf[((size_t)h * (size_t)n_q + (size_t)t) * (size_t)dh + (size_t)d] =
                    load_f(q, q_dtype, ((size_t)h * (size_t)n_q + (size_t)t) * (size_t)dh + (size_t)d);
            }
        }
    }
    std::vector<float> q_pool((size_t)H * (size_t)dh, 0.f);
    for (int h = 0; h < H; ++h) {
        for (int t = 0; t < n_q; ++t) {
            const float * src = qf.data() + ((size_t)h * (size_t)n_q + (size_t)t) * (size_t)dh;
            float * dst = q_pool.data() + (size_t)h * (size_t)dh;
            for (int d = 0; d < dh; ++d) dst[d] += src[d];
        }
        if (n_q > 1) {
            const float inv = 1.f / (float)n_q;
            for (int d = 0; d < dh; ++d) q_pool[(size_t)h * (size_t)dh + (size_t)d] *= inv;
        }
    }

    const int q_last = start_pos + n_q - 1;
    const int n_closed_view = std::min(L.n_closed, q_last / s->cfg.chunk_size);
    RouteSet rs;
    int n_sel = 0, n_open = 0;
    route_layer(s, L, q_pool.data(), n_closed_view, rs, &n_sel, &n_open);
    std::vector<Span> spans;
    collect_spans(s, L, rs, n_closed_view, spans);

    const bool i8 = s->cfg.prec == HGA_PREC_I8;
    std::vector<int8_t> q8;
    std::vector<float> qsc;
    if (i8) {
        q8.resize((size_t)H * (size_t)n_q * (size_t)dh);
        qsc.resize((size_t)H * (size_t)n_q);
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < n_q; ++t) {
                const size_t off = ((size_t)h * (size_t)n_q + (size_t)t) * (size_t)dh;
                quant_vec_i8(qf.data() + off, dh, q8.data() + off,
                             &qsc[(size_t)h * (size_t)n_q + (size_t)t]);
            }
        }
    }
    const double t1 = now_ms();

    const int n_kvh = s->cfg.n_kv_heads;
    const int rep = H / n_kvh;
    const int nthr = std::max(1, s->cfg.n_threads);
    std::vector<int> keys;
    collect_keys(s, L, spans, q_last, keys);
    const int n_keys = (int)keys.size();
    const int n_q_rows = rep * n_q;
    int nqt = 1, nkt = 1;
    pick_qk_tiles(n_kvh, std::max(1, n_q_rows), std::max(1, n_keys), dh,
                  i8 ? 1 : 4, i8 ? 1 : 2, nthr, &nqt, &nkt);

    int n_h_tiles = std::min(nqt, std::max(1, rep));
    while (n_h_tiles > 1 && nqt % n_h_tiles != 0) --n_h_tiles;
    const int n_t_tiles = nqt / n_h_tiles;

    auto q_bounds = [&](int kh, int qt, int * h0, int * h1, int * t0, int * t1) {
        const int ht = qt / n_t_tiles;
        const int tt = qt - ht * n_t_tiles;
        const int hg0 = kh * rep;
        *h0 = hg0 + ht * rep / n_h_tiles;
        *h1 = hg0 + (ht + 1) * rep / n_h_tiles;
        *t0 = tt * n_q / n_t_tiles;
        *t1 = (tt + 1) * n_q / n_t_tiles;
    };

    int max_rows = 1;
    for (int qt = 0; qt < nqt; ++qt) {
        int h0, h1, t0, t1;
        q_bounds(0, qt, &h0, &h1, &t0, &t1);
        max_rows = std::max(max_rows, (h1 - h0) * (t1 - t0));
    }

    const int n_parts = n_kvh * nqt * nkt;
    std::vector<float> part_m((size_t)n_parts * (size_t)max_rows,
                              -std::numeric_limits<float>::infinity());
    std::vector<float> part_lse((size_t)n_parts * (size_t)max_rows, 0.f);
    std::vector<float> part_acc((size_t)n_parts * (size_t)max_rows * (size_t)dh, 0.f);

#if defined(_OPENMP)
#pragma omp parallel for collapse(3) num_threads(nthr) schedule(static)
#endif
    for (int kh = 0; kh < n_kvh; ++kh) {
        for (int qt = 0; qt < nqt; ++qt) {
            for (int kt = 0; kt < nkt; ++kt) {
                int h0, h1, t0, t1;
                q_bounds(kh, qt, &h0, &h1, &t0, &t1);
                const int k0 = n_keys ? kt * n_keys / nkt : 0;
                const int k1 = n_keys ? (kt + 1) * n_keys / nkt : 0;
                if (h1 <= h0 || t1 <= t0) continue;
                const int part = (kh * nqt + qt) * nkt + kt;
                float * m = part_m.data() + (size_t)part * (size_t)max_rows;
                float * ls = part_lse.data() + (size_t)part * (size_t)max_rows;
                float * ac = part_acc.data() + (size_t)part * (size_t)max_rows * (size_t)dh;
                if (i8) {
                    flash_tile_i8(s, L, kh, start_pos, n_q, h0, h1, t0, t1,
                                  keys.data(), k0, k1, q8.data(), qsc.data(), m, ls, ac);
                } else {
                    flash_tile_f16(s, L, kh, start_pos, n_q, h0, h1, t0, t1,
                                   keys.data(), k0, k1, qf.data(), m, ls, ac);
                }
            }
        }
    }

    const int n_merge = n_kvh * nqt;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(nthr) schedule(static)
#endif
    for (int u = 0; u < n_merge; ++u) {
        const int kh = u / nqt;
        const int qt = u - kh * nqt;
        int h0, h1, t0, t1;
        q_bounds(kh, qt, &h0, &h1, &t0, &t1);
        const int n_rows = (h1 - h0) * (t1 - t0);
        if (n_rows <= 0) continue;
        const int base = (kh * nqt + qt) * nkt;
        float * m = part_m.data() + (size_t)base * (size_t)max_rows;
        float * ls = part_lse.data() + (size_t)base * (size_t)max_rows;
        float * ac = part_acc.data() + (size_t)base * (size_t)max_rows * (size_t)dh;
        for (int kt = 1; kt < nkt; ++kt) {
            const int p = base + kt;
            merge_online_softmax(m, ls, ac,
                                 part_m.data() + (size_t)p * (size_t)max_rows,
                                 part_lse.data() + (size_t)p * (size_t)max_rows,
                                 part_acc.data() + (size_t)p * (size_t)max_rows * (size_t)dh,
                                 n_rows, dh);
        }
        int li = 0;
        for (int h = h0; h < h1; ++h) {
            for (int t = t0; t < t1; ++t, ++li) {
                float * o = ac + (size_t)li * (size_t)dh;
                if (ls[li] > 0.f) hga_scale_f32(o, 1.f / ls[li], dh);
                else std::memset(o, 0, (size_t)dh * sizeof(float));
                std::memcpy(out + ((size_t)h * (size_t)n_q + (size_t)t) * (size_t)dh,
                            o, (size_t)dh * sizeof(float));
            }
        }
    }
    const double t2 = now_ms();

    if (stats) {
        std::memset(stats, 0, sizeof(*stats));
        stats->n_kv = L.n_kv;
        stats->n_closed_chunks = L.n_closed;
        stats->n_selected_chunks = n_sel;
        stats->n_opened_groups = n_open;
        int att = 0;
        const int pos = q_last;
        for (const Span & sp : spans) {
            const int hi = std::min(sp.start + sp.len - 1, pos);
            if (hi >= sp.start) att += hi - sp.start + 1;
        }
        const int act0 = L.n_closed * s->cfg.chunk_size;
        if (L.n_kv > act0) att += std::min(L.n_kv - 1, pos) - act0 + 1;
        if (att < 1) att = 1;
        stats->n_attended_tokens = att;
        stats->sparsity = (L.n_kv > 0) ? (float)att / (float)L.n_kv : 1.f;
        stats->ms_route = t1 - t0;
        stats->ms_attn = t2 - t1;
    }
    (void)NEG;
}

void hga_forward(hga_session * s, int layer, int start_pos, int n_q,
                 const void * q, const void * k_rope, const void * k_raw, const void * v,
                 hga_dtype dtype, float * out, hga_stats * stats) {
    hga_append(s, layer, start_pos, n_q, k_rope, k_raw, v, dtype);
    hga_attend(s, layer, start_pos, n_q, q, dtype, out, stats);
    hga_close_full_chunks(s, layer);
}
