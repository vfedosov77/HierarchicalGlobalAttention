#include "hga.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

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
constexpr int HGA_MAX_DH = 256;
constexpr int HGA_MIN_ROUTED_CHUNKS = 3;
constexpr int HGA_MIN_OPEN_GROUPS = 6;
constexpr int HGA_MAX_ROUTED_CHUNKS = 20;
constexpr int HGA_MAX_OPEN_GROUPS = 32;
/* Six query heads share one Qwen KV head. Preserve the existing exact-token
 * staging budget of at most 3x one route's group budget. */
constexpr int HGA_KV_HEAD_UNION_MULTIPLIER = 3;
static volatile uint64_t hga_profile_touch_sink = 0;

template <typename T> static inline void grow(std::vector<T> &v, size_t n) {
  if (v.size() < n)
    v.resize(n);
}

/* Kept only for on-host A/B measurement.  The default is the batched verify
 * kernel; HGA_VERIFY_BATCH=0 selects the old per-token verify loop. */
static inline bool hga_verify_batch_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("HGA_VERIFY_BATCH");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

/* Experimental row splitting trades K/V cache reuse for more workers.  It is
 * useful for investigation but slower on the target host, so never enable it
 * in the normal key-tiled verify path without an explicit A/B request. */
static inline bool hga_verify_rows_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("HGA_VERIFY_ROWS");
    return value && value[0] && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

/* The normal small-batch path divides every KV head into disjoint key tiles.
 * Set HGA_VERIFY_TILES=0 only to retain the old four-worker fused kernel for
 * A/B measurements.  HGA_VERIFY_ROWS remains an explicit experimental override.
 */
static inline bool hga_verify_tiles_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("HGA_VERIFY_TILES");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

/* Large-prefill A/B mode: split both query rows and keys. Qwen3.8 has four KV
 * heads, so two Q tiles x five K tiles expose all 40 physical cores while
 * avoiding the very large partial accumulators of the old 1-Q-tile layout. */
static inline bool hga_prefill_k_tiles_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("HGA_PREFILL_K_TILES");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

static inline bool env_enabled(const char *name) {
  const char *value = std::getenv(name);
  return value && value[0] && std::strcmp(value, "0") != 0;
}

/* Turing1 SMT A/B: use both logical CPUs of a core for two disjoint GQA-head
 * groups that stream the same KV tile.  OMP_PLACES=cores + OMP_PROC_BIND=close
 * maps consecutive worker pairs to the two siblings of one physical core.
 * HGA_SMT enables both paths; the narrower names support attribution. */
static inline bool hga_verify_smt_enabled() {
  static const bool enabled =
      env_enabled("HGA_SMT") || env_enabled("HGA_VERIFY_SMT");
  return enabled;
}

static inline bool hga_prefill_smt_enabled() {
  static const bool enabled =
      env_enabled("HGA_SMT") || env_enabled("HGA_PREFILL_SMT");
  return enabled;
}

static inline uint16_t f32_to_f16(float f) {
#if defined(__F16C__)
  return (uint16_t)_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
#else
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
  uint32_t man = x & 0x7fffffu;
  if (exp <= 0) {
    if (exp < -10)
      return (uint16_t)sign;
    man = (man | 0x800000u) >> (1 - exp);
    return (uint16_t)(sign | (man >> 13));
  }
  if (exp >= 31)
    return (uint16_t)(sign | 0x7c00u);
  return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
#endif
}

static inline float f16_to_f32(uint16_t h) {
#if defined(__F16C__)
  return _cvtsh_ss(h);
#else
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t man = h & 0x3ffu;
  uint32_t out;
  if (exp == 0) {
    if (man == 0) {
      out = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        exp--;
      }
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
#endif
}

static inline void f32_to_f16_vec(const float *src, uint16_t *dst, int n) {
  int i = 0;
#if HGA_AVX512
  for (; i + 16 <= n; i += 16) {
    __m256i h =
        _mm512_cvtps_ph(_mm512_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT);
    _mm256_storeu_si256((__m256i *)(dst + i), h);
  }
#elif defined(__F16C__)
  for (; i + 8 <= n; i += 8) {
    __m128i h =
        _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128((__m128i *)(dst + i), h);
  }
#endif
  for (; i < n; ++i)
    dst[i] = f32_to_f16(src[i]);
}

static inline void i8_scale_to_f16_vec(const int8_t *src, float scale,
                                       uint16_t *dst, int n) {
  int i = 0;
#if HGA_AVX512
  const __m512 ss = _mm512_set1_ps(scale);
  for (; i + 16 <= n; i += 16) {
    const __m128i q = _mm_loadu_si128((const __m128i *)(src + i));
    const __m512 f =
        _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q)), ss);
    const __m256i h = _mm512_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT);
    _mm256_storeu_si256((__m256i *)(dst + i), h);
  }
#elif defined(__AVX2__)
  const __m256 ss = _mm256_set1_ps(scale);
  for (; i + 8 <= n; i += 8) {
    const __m128i q = _mm_loadl_epi64((const __m128i *)(src + i));
    const __m256 f =
        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)), ss);
#if defined(__F16C__)
    _mm_storeu_si128((__m128i *)(dst + i),
                     _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT));
#else
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, f);
    for (int j = 0; j < 8; ++j)
      dst[i + j] = f32_to_f16(tmp[j]);
#endif
  }
#endif
  for (; i < n; ++i)
    dst[i] = f32_to_f16((float)src[i] * scale);
}

static inline void prefetch_bytes(const void *p, int n) {
#if defined(__AVX512F__) || defined(__AVX2__)
  const char *c = (const char *)p;
  for (int b = 0; b < n; b += 64)
    _mm_prefetch(c + b, _MM_HINT_T0);
#else
  (void)p;
  (void)n;
#endif
}

static inline float load_f(const void *p, hga_dtype dt, size_t i) {
  if (dt == HGA_F32)
    return ((const float *)p)[i];
  return f16_to_f32(((const uint16_t *)p)[i]);
}

#if HGA_AVX512
static inline float hga_dot_f32(const float *a, const float *b, int n) {
  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    acc0 =
        _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
    acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16),
                           _mm512_loadu_ps(b + i + 16), acc1);
  }
  acc0 = _mm512_add_ps(acc0, acc1);
  for (; i + 16 <= n; i += 16) {
    acc0 =
        _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
  }
  float s = _mm512_reduce_add_ps(acc0);
  for (; i < n; ++i)
    s += a[i] * b[i];
  return s;
}

static inline void hga_axpy_f32(float *acc, const float *v, float a, int n) {
  const __m512 as = _mm512_set1_ps(a);
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm512_storeu_ps(acc + i, _mm512_fmadd_ps(as, _mm512_loadu_ps(v + i),
                                              _mm512_loadu_ps(acc + i)));
    _mm512_storeu_ps(acc + i + 16,
                     _mm512_fmadd_ps(as, _mm512_loadu_ps(v + i + 16),
                                     _mm512_loadu_ps(acc + i + 16)));
  }
  for (; i + 16 <= n; i += 16) {
    _mm512_storeu_ps(acc + i, _mm512_fmadd_ps(as, _mm512_loadu_ps(v + i),
                                              _mm512_loadu_ps(acc + i)));
  }
  for (; i < n; ++i)
    acc[i] += a * v[i];
}

static inline void hga_scale_f32(float *acc, float s, int n) {
  const __m512 ss = _mm512_set1_ps(s);
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm512_storeu_ps(acc + i, _mm512_mul_ps(_mm512_loadu_ps(acc + i), ss));
    _mm512_storeu_ps(acc + i + 16,
                     _mm512_mul_ps(_mm512_loadu_ps(acc + i + 16), ss));
  }
  for (; i + 16 <= n; i += 16) {
    _mm512_storeu_ps(acc + i, _mm512_mul_ps(_mm512_loadu_ps(acc + i), ss));
  }
  for (; i < n; ++i)
    acc[i] *= s;
}

static inline float hga_dot_f16k(const float *q, const uint16_t *k, int n) {
  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    acc0 = _mm512_fmadd_ps(
        _mm512_loadu_ps(q + i),
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(k + i))), acc0);
    acc1 = _mm512_fmadd_ps(
        _mm512_loadu_ps(q + i + 16),
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(k + i + 16))),
        acc1);
  }
  acc0 = _mm512_add_ps(acc0, acc1);
  for (; i + 16 <= n; i += 16) {
    acc0 = _mm512_fmadd_ps(
        _mm512_loadu_ps(q + i),
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(k + i))), acc0);
  }
  float s = _mm512_reduce_add_ps(acc0);
  for (; i < n; ++i)
    s += q[i] * f16_to_f32(k[i]);
  return s;
}

static inline void hga_axpy_f16(float *acc, const uint16_t *v, float a, int n) {
  const __m512 as = _mm512_set1_ps(a);
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    __m512 vv0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(v + i)));
    __m512 vv1 =
        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(v + i + 16)));
    _mm512_storeu_ps(acc + i,
                     _mm512_fmadd_ps(as, vv0, _mm512_loadu_ps(acc + i)));
    _mm512_storeu_ps(acc + i + 16,
                     _mm512_fmadd_ps(as, vv1, _mm512_loadu_ps(acc + i + 16)));
  }
  for (; i + 16 <= n; i += 16) {
    __m512 vv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(v + i)));
    _mm512_storeu_ps(acc + i,
                     _mm512_fmadd_ps(as, vv, _mm512_loadu_ps(acc + i)));
  }
  for (; i < n; ++i)
    acc[i] += a * f16_to_f32(v[i]);
}

static inline void hga_f16_to_f32_add(float *acc, const uint16_t *v, int n) {
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    _mm512_storeu_ps(acc + i, _mm512_add_ps(_mm512_loadu_ps(acc + i),
                                            _mm512_cvtph_ps(_mm256_loadu_si256(
                                                (const __m256i *)(v + i)))));
  }
  for (; i < n; ++i)
    acc[i] += f16_to_f32(v[i]);
}
#else
static inline float hga_dot_f32(const float *a, const float *b, int n) {
  float s = 0.f;
  for (int i = 0; i < n; ++i)
    s += a[i] * b[i];
  return s;
}
static inline void hga_axpy_f32(float *acc, const float *v, float a, int n) {
  for (int i = 0; i < n; ++i)
    acc[i] += a * v[i];
}
static inline void hga_scale_f32(float *acc, float s, int n) {
  for (int i = 0; i < n; ++i)
    acc[i] *= s;
}
static inline float hga_dot_f16k(const float *q, const uint16_t *k, int n) {
  float s = 0.f;
  for (int i = 0; i < n; ++i)
    s += q[i] * f16_to_f32(k[i]);
  return s;
}
static inline void hga_axpy_f16(float *acc, const uint16_t *v, float a, int n) {
  for (int i = 0; i < n; ++i)
    acc[i] += a * f16_to_f32(v[i]);
}
static inline void hga_f16_to_f32_add(float *acc, const uint16_t *v, int n) {
  for (int i = 0; i < n; ++i)
    acc[i] += f16_to_f32(v[i]);
}
#endif

/* Symmetric INT8, clamp to [-127, 127] so maddubs abs never hits -128. */
static void quant_vec_i8(const float *x, int n, int8_t *q, float *scale) {
  float amax = 0.f;
  int i = 0;
#if HGA_AVX512
  __m512 vmax = _mm512_setzero_ps();
  for (; i + 16 <= n; i += 16) {
    vmax = _mm512_max_ps(vmax, _mm512_abs_ps(_mm512_loadu_ps(x + i)));
  }
  amax = _mm512_reduce_max_ps(vmax);
#endif
  for (; i < n; ++i)
    amax = std::max(amax, std::fabs(x[i]));
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
  const __m512 vlo = _mm512_set1_ps(-127.f);
  const __m512 vhi = _mm512_set1_ps(127.f);
  for (; i + 16 <= n; i += 16) {
    __m512 y = _mm512_mul_ps(_mm512_loadu_ps(x + i), vinv);
    y = _mm512_min_ps(vhi, _mm512_max_ps(vlo, y));
    __m128i packed = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(y));
    _mm_storeu_si128((__m128i *)(q + i), packed);
  }
#endif
  for (; i < n; ++i) {
    int v = (int)std::lrintf(x[i] * inv);
    if (v > 127)
      v = 127;
    if (v < -127)
      v = -127;
    q[i] = (int8_t)v;
  }
}

/* Signed×signed INT8 MAC. No VNNI / no 512-bit PSIGN (Xeon Gold 6148):
 * widen i8→i16 then madd_epi16. Two independent accumulators hide latency. */
static inline int32_t hga_dot_i8(const int8_t *a, const int8_t *b, int n) {
  int32_t sum = 0;
  int i = 0;
#if HGA_AVX512BW
  __m512i acc0 = _mm512_setzero_si512();
  __m512i acc1 = _mm512_setzero_si512();
  for (; i + 128 <= n; i += 128) {
    const __m512i a0 = _mm512_loadu_si512((const void *)(a + i));
    const __m512i b0 = _mm512_loadu_si512((const void *)(b + i));
    const __m512i a1 = _mm512_loadu_si512((const void *)(a + i + 64));
    const __m512i b1 = _mm512_loadu_si512((const void *)(b + i + 64));
    acc0 = _mm512_add_epi32(
        acc0,
        _mm512_madd_epi16(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(a0)),
                          _mm512_cvtepi8_epi16(_mm512_castsi512_si256(b0))));
    acc1 = _mm512_add_epi32(
        acc1, _mm512_madd_epi16(
                  _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a0, 1)),
                  _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b0, 1))));
    acc0 = _mm512_add_epi32(
        acc0,
        _mm512_madd_epi16(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(a1)),
                          _mm512_cvtepi8_epi16(_mm512_castsi512_si256(b1))));
    acc1 = _mm512_add_epi32(
        acc1, _mm512_madd_epi16(
                  _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a1, 1)),
                  _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b1, 1))));
  }
  for (; i + 64 <= n; i += 64) {
    const __m512i va = _mm512_loadu_si512((const void *)(a + i));
    const __m512i vb = _mm512_loadu_si512((const void *)(b + i));
    acc0 = _mm512_add_epi32(
        acc0,
        _mm512_madd_epi16(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(va)),
                          _mm512_cvtepi8_epi16(_mm512_castsi512_si256(vb))));
    acc1 = _mm512_add_epi32(
        acc1, _mm512_madd_epi16(
                  _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(va, 1)),
                  _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vb, 1))));
  }
  sum += _mm512_reduce_add_epi32(_mm512_add_epi32(acc0, acc1));
#endif
#if defined(__AVX2__)
  const __m256i one16 = _mm256_set1_epi16(1);
  for (; i + 32 <= n; i += 32) {
    const __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
    const __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
    const __m256i p32 =
        _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(va, va),
                                               _mm256_sign_epi8(vb, va)),
                          one16);
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(p32),
                              _mm256_extracti128_si256(p32, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
    sum += _mm_cvtsi128_si32(s);
  }
#endif
  for (; i < n; ++i)
    sum += (int32_t)a[i] * (int32_t)b[i];
  return sum;
}

static inline void hga_axpy_i8(float *acc, const int8_t *v, float a, int n) {
  int i = 0;
#if HGA_AVX512
  const __m512 as = _mm512_set1_ps(a);
  for (; i + 32 <= n; i += 32) {
    const __m256i v8 = _mm256_loadu_si256((const __m256i *)(v + i));
    const __m512 vf0 =
        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_castsi256_si128(v8)));
    const __m512 vf1 = _mm512_cvtepi32_ps(
        _mm512_cvtepi8_epi32(_mm256_extracti128_si256(v8, 1)));
    _mm512_storeu_ps(acc + i,
                     _mm512_fmadd_ps(as, vf0, _mm512_loadu_ps(acc + i)));
    _mm512_storeu_ps(acc + i + 16,
                     _mm512_fmadd_ps(as, vf1, _mm512_loadu_ps(acc + i + 16)));
  }
  for (; i + 16 <= n; i += 16) {
    const __m128i v8 = _mm_loadu_si128((const __m128i *)(v + i));
    const __m512 vf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v8));
    _mm512_storeu_ps(acc + i,
                     _mm512_fmadd_ps(as, vf, _mm512_loadu_ps(acc + i)));
  }
#endif
  for (; i < n; ++i)
    acc[i] += a * (float)v[i];
}

/* Dequant INT8 → F32 * scale. Used once per key so GQA heads share the float V.
 */
static inline void hga_i8_to_f32_scale(float *dst, const int8_t *v, float s,
                                       int n) {
  int i = 0;
#if HGA_AVX512
  const __m512 ss = _mm512_set1_ps(s);
  for (; i + 32 <= n; i += 32) {
    const __m256i v8 = _mm256_loadu_si256((const __m256i *)(v + i));
    _mm512_storeu_ps(dst + i,
                     _mm512_mul_ps(ss, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                                           _mm256_castsi256_si128(v8)))));
    _mm512_storeu_ps(dst + i + 16,
                     _mm512_mul_ps(ss, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                                           _mm256_extracti128_si256(v8, 1)))));
  }
  for (; i + 16 <= n; i += 16) {
    const __m128i v8 = _mm_loadu_si128((const __m128i *)(v + i));
    _mm512_storeu_ps(dst + i, _mm512_mul_ps(ss, _mm512_cvtepi32_ps(
                                                    _mm512_cvtepi8_epi32(v8))));
  }
#endif
  for (; i < n; ++i)
    dst[i] = s * (float)v[i];
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

struct WaveIndex {
  int mid_lo = 0;
  int mid_hi = 0;
  int n_mid = 0;
  int n_cent = 0;
  int built_n_closed = 0;
  int max_cs = 0;
  std::vector<float> cent;  /* [kvh, n_cent, dh] */
  std::vector<float> vsum;  /* [kvh, n_cent, dh]  sum of V in cluster */
  std::vector<int> members; /* [kvh, n_cent, max_cs] token positions */
  std::vector<int> csize;   /* [kvh, n_cent] */
};

/* Contiguous attended KV for one layer. Layout [kvh, cap, dh] so a decode
 * token is an append along the n axis without reshuffling. */
struct PackedKV {
  std::vector<int> keys;
  std::vector<int8_t> k8, v8;
  std::vector<float> ksc, vsc;
  std::vector<uint16_t> k16, v16;
  int n = 0;
  int cap = 0;
};

struct Layer {
  int n_kv = 0;
  int n_closed = 0;
  std::vector<uint16_t> k; /* F16 [kvh, max_seq, dh] */
  std::vector<uint16_t> v;
  std::vector<uint16_t> k_raw; /* pre-RoPE F16 (summaries) */
  std::vector<int8_t> k8;      /* I8  [kvh, max_seq, dh] */
  std::vector<int8_t> v8;
  std::vector<float> k_scale; /* [kvh, max_seq] */
  std::vector<float> v_scale;
  std::vector<float> chunk_k; /* [kvh, max_chunks, dh] */
  std::vector<float> group_k; /* [kvh, max_chunks, G, dh] */
  PackedKV pack;
  WaveIndex wave;
};

struct RopeTables {
  int rotary_dim = 0;
  int cutoff_pair = 0;
  std::vector<float> inv_freq; /* [rotary_dim/2] */
};

static int mixed_cutoff_pair(const hga_config &c) {
  const int half = c.rotary_dim / 2;
  int cutoff = 0;
  const float span = (float)std::max(1, c.chunk_size - 1);
  for (int i = 0; i < half; ++i) {
    const float inv = 1.f / std::pow(c.theta, (float)i / (float)half);
    if (span * inv > c.mixed_rope_threshold)
      cutoff = i + 1;
  }
  return cutoff;
}

static RopeTables make_rope(const hga_config &c) {
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
static void apply_partial_rotary_hf(float *x, int rd, const float *cos,
                                    const float *sin) {
  const int half = rd / 2;
  float out[128];
  for (int i = 0; i < half; ++i) {
    const float x1 = x[i];
    const float x2 = x[half + i];
    out[i] = x1 * cos[i] + (-x2) * sin[i];
    out[half + i] = x2 * cos[half + i] + x1 * sin[half + i];
  }
  for (int i = 0; i < rd; ++i)
    x[i] = out[i];
}

static void rope_cos_sin(const RopeTables &r, int pos, float *cos, float *sin) {
  const int half = r.rotary_dim / 2;
  for (int i = 0; i < half; ++i) {
    const float ang = (float)pos * r.inv_freq[(size_t)i];
    const float c = std::cos(ang);
    const float s = std::sin(ang);
    cos[i] = c;
    cos[half + i] = c;
    sin[i] = s;
    sin[half + i] = s;
  }
}

static void mix_tokenwise_anchor(float *dst, const float *tokenwise,
                                 const float *anchor, int dh, int rd,
                                 int cutoff_pair) {
  const int half = rd / 2;
  const int cut =
      cutoff_pair < 0 ? 0 : (cutoff_pair > half ? half : cutoff_pair);
  int i = 0;
  for (; i < cut; ++i)
    dst[i] = tokenwise[i];
  for (; i < half; ++i)
    dst[i] = anchor[i];
  const int hi_tw = half + cut;
  for (; i < hi_tw; ++i)
    dst[i] = tokenwise[i];
  for (; i < rd; ++i)
    dst[i] = anchor[i];
  for (; i < dh; ++i)
    dst[i] = tokenwise[i];
}

/* Mean-pool mixed-RoPE summary over `len` tokens starting at `base` for one kv
 * head. Post-RoPE K is either F16 (k_rope_h) or INT8 (k8_h + k_sc_h). */
static void rope_summary(const hga_config &c, const RopeTables &rope,
                         const uint16_t *k_raw_h, const uint16_t *k_rope_h,
                         const int8_t *k8_h, const float *k_sc_h, bool use_i8,
                         int base, int len, int anchor_pos, float scale,
                         int max_seq, float *out) {
  const int dh = c.head_dim;
  const int rd = c.rotary_dim;
  alignas(64) float raw_sum[HGA_MAX_DH];
  alignas(64) float tok_sum[HGA_MAX_DH];
  std::memset(raw_sum, 0, (size_t)dh * sizeof(float));
  std::memset(tok_sum, 0, (size_t)dh * sizeof(float));
  const uint16_t *raw = k_raw_h + (size_t)base * (size_t)dh;
  const int8_t *rp8 = use_i8 ? k8_h + (size_t)base * (size_t)dh : nullptr;
  const uint16_t *rp16 =
      (!use_i8 && k_rope_h) ? k_rope_h + (size_t)base * (size_t)dh : nullptr;
  const float *sc_p = use_i8 ? k_sc_h + base : nullptr;
  for (int t = 0; t < len; ++t) {
    hga_f16_to_f32_add(raw_sum, raw, dh);
    raw += dh;
    if (use_i8) {
      hga_axpy_i8(tok_sum, rp8, sc_p[t], dh);
      rp8 += dh;
    } else {
      hga_f16_to_f32_add(tok_sum, rp16, dh);
      rp16 += dh;
    }
  }
  hga_scale_f32(raw_sum, scale, dh);
  hga_scale_f32(tok_sum, scale, dh);
  float cosv[128], sinv[128];
  rope_cos_sin(rope, anchor_pos, cosv, sinv);
  apply_partial_rotary_hf(raw_sum, rd, cosv, sinv);
  mix_tokenwise_anchor(out, tok_sum, raw_sum, dh, rd, rope.cutoff_pair);
  (void)max_seq;
}

static size_t kv_index(int kvh, int t, int dh, int max_seq) {
  return ((size_t)kvh * (size_t)max_seq + (size_t)t) * (size_t)dh;
}

static void topk_idx(const float *scores, int n, int k, int *out, int *idx) {
  k = std::min(k, n);
  if (k <= 0)
    return;
  for (int i = 0; i < n; ++i)
    idx[i] = i;
  std::partial_sort(idx, idx + k, idx + n,
                    [&](int a, int b) { return scores[a] > scores[b]; });
  for (int i = 0; i < k; ++i)
    out[i] = idx[i];
}

} /* namespace */

class HgaPackPool {
public:
  explicit HgaPackPool(int n_threads)
      : n_(std::max(1, n_threads)), cpu_ids_(physical_cpu_ids(n_)) {
    workers_.reserve((size_t)n_);
    for (int i = 0; i < n_; ++i)
      workers_.emplace_back([this, i] { worker_loop(i); });
  }

  ~HgaPackPool() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
      ++generation_;
    }
    work_cv_.notify_all();
    for (std::thread &worker : workers_)
      worker.join();
  }

  HgaPackPool(const HgaPackPool &) = delete;
  HgaPackPool &operator=(const HgaPackPool &) = delete;

  template <typename Fn> void parallel_for(int begin, int end, Fn fn) {
    if (end <= begin)
      return;
    std::unique_lock<std::mutex> lock(mu_);
    begin_ = begin;
    end_ = end;
    job_ = fn;
    pending_ = n_;
    ++generation_;
    work_cv_.notify_all();
    done_cv_.wait(lock, [&] { return pending_ == 0; });
    job_ = {};
  }

private:
  static std::vector<int> physical_cpu_ids(int wanted) {
    std::vector<int> ids;
#if defined(__linux__)
    const long ncpu = sysconf(_SC_NPROCESSORS_CONF);
    std::vector<std::pair<int, int>> seen;
    for (int cpu = 0; cpu < ncpu && (int)ids.size() < wanted; ++cpu) {
      int package = 0;
      int core = cpu;
      char path[160];
      std::snprintf(path, sizeof(path),
                    "/sys/devices/system/cpu/cpu%d/topology/physical_package_id",
                    cpu);
      if (FILE *f = std::fopen(path, "r")) {
        if (std::fscanf(f, "%d", &package) != 1)
          package = 0;
        std::fclose(f);
      }
      std::snprintf(path, sizeof(path),
                    "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
      if (FILE *f = std::fopen(path, "r")) {
        if (std::fscanf(f, "%d", &core) != 1)
          core = cpu;
        std::fclose(f);
      }
      const std::pair<int, int> key{package, core};
      if (std::find(seen.begin(), seen.end(), key) == seen.end()) {
        seen.push_back(key);
        ids.push_back(cpu);
      }
    }
#else
    (void)wanted;
#endif
    return ids;
  }

  void pin_worker(int worker) {
#if defined(__linux__)
    if (worker >= (int)cpu_ids_.size())
      return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_ids_[(size_t)worker], &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)worker;
#endif
  }

  void worker_loop(int worker) {
    pin_worker(worker);
    uint64_t seen_generation = 0;
    std::unique_lock<std::mutex> lock(mu_);
    for (;;) {
      work_cv_.wait(lock,
                    [&] { return stop_ || generation_ != seen_generation; });
      if (stop_)
        return;
      seen_generation = generation_;
      const int count = end_ - begin_;
      const int i0 = begin_ + count * worker / n_;
      const int i1 = begin_ + count * (worker + 1) / n_;
      lock.unlock();
      for (int i = i0; i < i1; ++i)
        job_(i);
      lock.lock();
      if (--pending_ == 0)
        done_cv_.notify_one();
    }
  }

  int n_ = 1;
  std::vector<int> cpu_ids_;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  std::function<void(int)> job_;
  uint64_t generation_ = 0;
  int begin_ = 0;
  int end_ = 0;
  int pending_ = 0;
  bool stop_ = false;
};

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
  /* Reused across tokens — decode must not malloc in the hot path. */
  std::vector<float> scratch_q_pool;
  std::vector<float> scratch_qsc;
  std::vector<float> scratch_qf;
  std::vector<int8_t> scratch_q8;
  std::vector<int> scratch_keys;
  std::vector<float> scratch_part_m;
  std::vector<float> scratch_part_lse;
  std::vector<float> scratch_part_acc;
  std::vector<float> scratch_scores;
  /* Verify n_q=2..8: one independent online-softmax state per KV head and
   * query/head row.  Kept separate from the tiled-prefill scratch so the
   * fixed-shape speculative graph never allocates in its hot path. */
  std::vector<float> scratch_verify_invz;
  std::vector<float> scratch_verify_acc;
  std::vector<float> scratch_route_sc;
  std::vector<float> scratch_route_pooled;
  std::vector<int> scratch_route_idx;
  std::vector<int> last_keys;
  std::vector<float> scratch_fixed_input;
  std::vector<float> scratch_fixed_out;
  int last_keys_layer = -1;
  void *l2 = nullptr;
  std::unique_ptr<HgaPackPool> pack_pool;
  hga_cache_metrics metrics{};
};

/* Keep one stable OpenMP team for the alternating route/pack phases.  Asking
 * libgomp for 24 workers in route_layer, then 12 workers here, made the next
 * routing region pay a large team resize/wakeup penalty on the target host.
 * The full routing team therefore enters the region. Packing workers are
 * selected evenly across that team (24/12 => tids 0,2,...,22), so with the
 * target's OMP_PLACES=threads ordering they occupy distinct physical cores
 * instead of the first six SMT sibling pairs. */
template <typename Fn>
static inline void hga_pack_parallel_for(hga_session *s, int begin,
                                         int end, int parallel_threshold,
                                         Fn fn) {
  const int count = end - begin;
  if (count <= 0)
    return;
  if (s->pack_pool && count >= parallel_threshold) {
    s->pack_pool->parallel_for(begin, end, fn);
    return;
  }
#if defined(_OPENMP)
  const hga_config &cfg = s->cfg;
  const int team = std::max(1, cfg.n_threads);
  const int active =
      std::min(count, std::max(1, std::min(cfg.n_pack_threads, team)));
  if (team > 1 && count >= parallel_threshold) {
#pragma omp parallel num_threads(team)
    {
      const int tid = omp_get_thread_num();
      int worker = -1;
      for (int w = 0; w < active; ++w) {
        if (tid == w * team / active) {
          worker = w;
          break;
        }
      }
      if (worker >= 0) {
        const int i0 = begin + count * worker / active;
        const int i1 = begin + count * (worker + 1) / active;
        for (int i = i0; i < i1; ++i)
          fn(i);
      }
    }
    return;
  }
#else
  (void)parallel_threshold;
#endif
  for (int i = begin; i < end; ++i)
    fn(i);
}

hga_config hga_config_qwen38_27b(int levels, int max_seq, int n_threads) {
  hga_config c{};
  c.n_q_heads = 24;
  c.n_kv_heads = 4;
  c.head_dim = 256;
  c.rotary_dim = 64;
  c.chunk_size = 64;
  c.group_size = 16;
  c.keep_first = 2;
  c.keep_last = 7;
  c.levels = (levels == 1) ? 1 : 2; /* default HGA-2 */
  c.frac_l1 = 0.08f;
  c.frac_l2 = 0.04f;
  c.theta = 1.0e6f;
  c.mixed_rope_threshold = 0.5f;
  c.n_threads = n_threads > 0 ? n_threads : 1;
  c.n_pack_threads = c.n_threads;
  c.max_seq = max_seq > 0 ? max_seq : 32768;
  c.prec = HGA_PREC_I8;
  c.router = HGA_ROUTER_HIER;
  c.frac_retr = 0.018f;
  c.frac_est = 0.232f;
  c.wave_cluster = 16;
  c.wave_seg = 8192;
  c.wave_iters = 3;
  c.wave_update = 1024;
  return c;
}

int hga_routing_base_chunks(const hga_config *cfg, int n_closed, int n_q) {
  if (n_closed <= 0)
    return 0;
  const int C = std::max(1, cfg->chunk_size);
  const int sink = std::min(std::max(0, cfg->keep_first), n_closed);
  const int local = std::max(0, cfg->keep_last);
  const int q_chunks = std::min(local + 1, (std::max(1, n_q) + C - 1) / C);
  const int local_absorbed_by_query = std::max(0, q_chunks - 1);
  const int extra_local = std::max(0, local - local_absorbed_by_query);
  return std::max(0, n_closed - sink - std::min(extra_local, n_closed - sink));
}

int hga_topk_chunks_for_query(const hga_config *cfg, int n_closed, int n_q) {
  if (n_closed <= 0)
    return 0;
  const int n_win = std::min(cfg->keep_first + cfg->keep_last, n_closed);
  const int n_mid = n_closed - n_win;
  if (n_mid <= 0)
    return 0;
  const int n_base = hga_routing_base_chunks(cfg, n_closed, n_q);
  int want = (int)std::lround((double)cfg->frac_l1 * (double)n_base);
  want = std::max(want, HGA_MIN_ROUTED_CHUNKS);
  return std::min({want, n_mid, HGA_MAX_ROUTED_CHUNKS});
}

int hga_topk_chunks(const hga_config *cfg, int n_closed) {
  return hga_topk_chunks_for_query(cfg, n_closed, 1);
}

int hga_topk_groups_for_query(const hga_config *cfg, int n_closed,
                              int topk_chunks, int n_q) {
  const int G = cfg->chunk_size / cfg->group_size;
  if (topk_chunks <= 0)
    return 0;
  if (cfg->levels != 2)
    return topk_chunks * G; /* one-level: open every group */
  const int n_base = hga_routing_base_chunks(cfg, n_closed, n_q);
  int want =
      (int)std::lround((double)cfg->frac_l2 * (double)n_base *
                       (double)cfg->chunk_size / (double)cfg->group_size);
  /* Preserve several retrieval alternatives at short context while still
   * opening at least one group in every selected chunk. */
  const int floor_g = std::max(topk_chunks, HGA_MIN_OPEN_GROUPS);
  const int cap = topk_chunks * G;
  want = std::max(want, floor_g);
  return std::min({want, cap, HGA_MAX_OPEN_GROUPS});
}

int hga_topk_groups(const hga_config *cfg, int n_closed, int topk_chunks) {
  return hga_topk_groups_for_query(cfg, n_closed, topk_chunks, 1);
}

hga_session *hga_session_create(const hga_config *cfg, int n_layers) {
  auto *s = new hga_session();
  s->cfg = *cfg;
  if (s->cfg.n_pack_threads <= 0)
    s->cfg.n_pack_threads = std::max(1, s->cfg.n_threads);
  if (s->cfg.n_pack_threads != s->cfg.n_threads &&
      s->cfg.n_pack_threads > 1)
    s->pack_pool = std::make_unique<HgaPackPool>(s->cfg.n_pack_threads);
  if (s->cfg.group_size <= 0 || s->cfg.chunk_size % s->cfg.group_size != 0) {
    s->cfg.group_size = s->cfg.chunk_size;
  }
  s->n_layers = n_layers;
  s->groups_per_chunk = s->cfg.chunk_size / s->cfg.group_size;
  s->max_chunks = (s->cfg.max_seq + s->cfg.chunk_size - 1) / s->cfg.chunk_size;
  s->scale = 1.f / std::sqrt((float)s->cfg.head_dim);
  s->group_scale =
      1.f / (s->cfg.group_size + std::sqrt((float)s->cfg.group_size));
  s->rope = make_rope(s->cfg);
  s->layers.resize((size_t)n_layers);
  const int kvh = s->cfg.n_kv_heads;
  const int dh = s->cfg.head_dim;
  const int ms = s->cfg.max_seq;
  const int G = s->groups_per_chunk;
  const bool i8 = s->cfg.prec == HGA_PREC_I8;
  for (auto &L : s->layers) {
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
    L.group_k.assign(
        (size_t)kvh * (size_t)s->max_chunks * (size_t)G * (size_t)dh, 0.f);
  }
  return s;
}

void hga_session_free(hga_session *s) { delete s; }

void hga_session_reset(hga_session *s) {
  for (auto &L : s->layers) {
    L.n_kv = 0;
    L.n_closed = 0;
    L.pack = PackedKV{};
    L.wave = WaveIndex{};
  }
  s->metrics = hga_cache_metrics{};
}

const hga_config *hga_session_config(const hga_session *s) { return &s->cfg; }
int hga_session_n_layers(const hga_session *s) { return s->n_layers; }
int hga_session_n_kv(const hga_session *s, int layer) {
  return s->layers[(size_t)layer].n_kv;
}

void hga_session_cache_metrics(const hga_session *s, hga_cache_metrics *out) {
  if (out)
    *out = s->metrics;
}

int hga_fixed_batch_validate(const hga_fixed_batch *batch) {
  if (!batch || batch->real_count == 0 ||
      batch->real_count > batch->physical_count || batch->physical_count > 64 ||
      (batch->mode != HGA_MODE_PREFILL && batch->mode != HGA_MODE_VERIFY))
    return 0;
  const uint32_t expected = batch->mode == HGA_MODE_PREFILL ? 512u : 3u;
  if (batch->physical_count != expected)
    return 0;
  const uint64_t prefix = batch->real_count == 64
                              ? ~UINT64_C(0)
                              : (UINT64_C(1) << batch->real_count) - 1;
  return batch->valid_mask == prefix;
}

void hga_set_ubatch(hga_session *s, int start_pos, int n_tokens) {
  s->ubatch_start = start_pos;
  s->ubatch_n = n_tokens;
}
int hga_ubatch_start(const hga_session *s) { return s ? s->ubatch_start : 0; }
int hga_ubatch_n(const hga_session *s) { return s ? s->ubatch_n : 0; }

static void summarize_closed_chunk(hga_session *s, Layer &L, int n) {
  const int C = s->cfg.chunk_size;
  const int kvh = s->cfg.n_kv_heads;
  const int dh = s->cfg.head_dim;
  const int gs = s->cfg.group_size;
  const int G = s->groups_per_chunk;
  const int ms = s->cfg.max_seq;
  const int base = n * C;
  const bool i8 = s->cfg.prec == HGA_PREC_I8;
  for (int h = 0; h < kvh; ++h) {
    const uint16_t *kr = i8 ? nullptr : L.k.data() + kv_index(h, 0, dh, ms);
    const int8_t *k8 = i8 ? L.k8.data() + kv_index(h, 0, dh, ms) : nullptr;
    const float *ksc = i8 ? L.k_scale.data() + (size_t)h * (size_t)ms : nullptr;
    const uint16_t *raw = L.k_raw.data() + kv_index(h, 0, dh, ms);
    float *ck = L.chunk_k.data() +
                ((size_t)h * (size_t)s->max_chunks + (size_t)n) * (size_t)dh;
    rope_summary(s->cfg, s->rope, raw, kr, k8, ksc, i8, base, C,
                 base + (C - 1) / 2, 1.f, ms, ck);
    float *gk =
        L.group_k.data() + ((size_t)h * (size_t)s->max_chunks + (size_t)n) *
                               (size_t)G * (size_t)dh;
    int gbase = base;
    int ganchor = base + (gs - 1) / 2;
    for (int g = 0; g < G; ++g) {
      rope_summary(s->cfg, s->rope, raw, kr, k8, ksc, i8, gbase, gs, ganchor,
                   s->group_scale, ms, gk);
      gk += dh;
      gbase += gs;
      ganchor += gs;
    }
  }
}

/* Close every newly-full chunk, not only n_kv/C - 1. A multi-chunk append
 * (ubatch > chunk_size, or one-shot hga_forward) would otherwise leave
 * skipped chunk_k/group_k at 0 while n_closed jumps over them. */
static void close_chunk_if_full(hga_session *s, Layer &L, int layer) {
  (void)layer;
  const int C = s->cfg.chunk_size;
  if (C <= 0 || L.n_kv == 0)
    return;
  const int last = L.n_kv / C - 1;
  const int old_closed = L.n_closed;
  for (int n = old_closed; n <= last && n < s->max_chunks; ++n)
    summarize_closed_chunk(s, L, n);
  if (last + 1 > L.n_closed)
    L.n_closed = std::min(last + 1, s->max_chunks);
  s->metrics.chunk_closures += (uint64_t)(L.n_closed - old_closed);
}

static void hga_append_f32_strided(hga_session *s, int layer, int start_pos,
                                   int n_new, const float *k_rope, int k_hs,
                                   int k_ts, const float *k_raw, int kr_hs,
                                   int kr_ts, const float *v, int v_hs,
                                   int v_ts) {
  Layer &L = s->layers[(size_t)layer];
  const int kvh = s->cfg.n_kv_heads;
  const int dh = s->cfg.head_dim;
  const int ms = s->cfg.max_seq;
  if (start_pos < L.n_kv) {
    L.n_kv = start_pos;
    L.n_closed = start_pos / s->cfg.chunk_size;
    /* Packed KV and the wave index hold absolute positions; a cache roll
     * back would otherwise reuse stale rows on a prefix-matching key list. */
    L.pack = PackedKV{};
    L.wave = WaveIndex{};
  }
  const bool i8 = s->cfg.prec == HGA_PREC_I8;
  int n_ok = n_new;
  if (start_pos + n_ok > ms)
    n_ok = ms - start_pos;
  if (n_ok <= 0)
    return;
  ++s->metrics.append_calls;
  s->metrics.appended_tokens += (uint64_t)n_ok;

  const size_t kv_hstride = (size_t)ms * (size_t)dh;
  hga_pack_parallel_for(s, 0, n_ok, 8, [&](int t) {
    const int pos = start_pos + t;
    const float *src_k = k_rope + (size_t)t * (size_t)k_ts;
    const float *src_v = v + (size_t)t * (size_t)v_ts;
    const float *src_kr = k_raw ? k_raw + (size_t)t * (size_t)kr_ts : src_k;
    uint16_t *rd = L.k_raw.data() + (size_t)pos * (size_t)dh;
    if (i8) {
      int8_t *k8d = L.k8.data() + (size_t)pos * (size_t)dh;
      int8_t *v8d = L.v8.data() + (size_t)pos * (size_t)dh;
      float *ksc = L.k_scale.data() + (size_t)pos;
      float *vsc = L.v_scale.data() + (size_t)pos;
      int8_t *k8d_end = k8d + (size_t)kvh * kv_hstride;
      for (; k8d < k8d_end; k8d += kv_hstride, v8d += kv_hstride, ksc += ms,
                            vsc += ms, rd += kv_hstride, src_k += k_hs,
                            src_v += v_hs) {
        f32_to_f16_vec(src_kr, rd, dh);
        quant_vec_i8(src_k, dh, k8d, ksc);
        quant_vec_i8(src_v, dh, v8d, vsc);
        if (k_raw)
          src_kr += kr_hs;
        else
          src_kr = src_k + k_hs;
      }
    } else {
      uint16_t *kd = L.k.data() + (size_t)pos * (size_t)dh;
      uint16_t *vd = L.v.data() + (size_t)pos * (size_t)dh;
      uint16_t *kd_end = kd + (size_t)kvh * kv_hstride;
      for (; kd < kd_end; kd += kv_hstride, vd += kv_hstride, rd += kv_hstride,
                          src_k += k_hs, src_v += v_hs) {
        f32_to_f16_vec(src_kr, rd, dh);
        f32_to_f16_vec(src_k, kd, dh);
        f32_to_f16_vec(src_v, vd, dh);
        if (k_raw)
          src_kr += kr_hs;
        else
          src_kr = src_k + k_hs;
      }
    }
  });
  L.n_kv = start_pos + n_ok;
}

/* ggml Q8_0 stores one F16 scale plus 32 signed bytes per block. Collapse its
 * eight scales for a 256-value head vector into HGA's one-scale-per-vector
 * cache without expanding K/V back to F32. */
static void q8_0_to_hga_i8_vec(const uint8_t *src, int block_stride, int dh,
                               int8_t *dst, float *dst_scale) {
  float amax = 0.f;
  for (int b = 0; b < dh / 32; ++b) {
    const uint8_t *block = src + (size_t)b * (size_t)block_stride;
    uint16_t scale_bits;
    std::memcpy(&scale_bits, block, sizeof(scale_bits));
    const float scale = f16_to_f32(scale_bits);
    const int8_t *q = (const int8_t *)(block + sizeof(scale_bits));
    for (int d = 0; d < 32; ++d)
      amax = std::max(amax, std::fabs(scale * (float)q[d]));
  }
  if (amax < 1e-12f) {
    std::memset(dst, 0, (size_t)dh);
    *dst_scale = 0.f;
    return;
  }
  *dst_scale = amax / 127.f;
  const float inv = 127.f / amax;
  for (int b = 0; b < dh / 32; ++b) {
    const uint8_t *block = src + (size_t)b * (size_t)block_stride;
    uint16_t scale_bits;
    std::memcpy(&scale_bits, block, sizeof(scale_bits));
    const float mul = f16_to_f32(scale_bits) * inv;
    const int8_t *q = (const int8_t *)(block + sizeof(scale_bits));
    for (int d = 0; d < 32; ++d) {
      int value = (int)std::lrintf((float)q[d] * mul);
      value = std::max(-127, std::min(127, value));
      dst[b * 32 + d] = (int8_t)value;
    }
  }
}

static void hga_append_q8_0_strided(
    hga_session *s, int layer, int start_pos, int n_new, const uint8_t *k_q8,
    int k_block_stride, int k_hs, int k_ts, const float *k_raw, int kr_hs,
    int kr_ts, const uint8_t *v_q8, int v_block_stride, int v_hs, int v_ts) {
  Layer &L = s->layers[(size_t)layer];
  const int kvh = s->cfg.n_kv_heads;
  const int dh = s->cfg.head_dim;
  const int ms = s->cfg.max_seq;
  if (start_pos < L.n_kv) {
    L.n_kv = start_pos;
    L.n_closed = start_pos / s->cfg.chunk_size;
    L.pack = PackedKV{};
    L.wave = WaveIndex{};
  }
  int n_ok = std::min(n_new, ms - start_pos);
  if (n_ok <= 0)
    return;
  ++s->metrics.append_calls;
  s->metrics.appended_tokens += (uint64_t)n_ok;
  const size_t cache_hstride = (size_t)ms * (size_t)dh;
  hga_pack_parallel_for(s, 0, n_ok, 8, [&](int t) {
    const int pos = start_pos + t;
    for (int kh = 0; kh < kvh; ++kh) {
      const uint8_t *ks = k_q8 + (size_t)kh * (size_t)k_hs +
                          (size_t)t * (size_t)k_ts;
      const uint8_t *vs = v_q8 + (size_t)kh * (size_t)v_hs +
                          (size_t)t * (size_t)v_ts;
      const float *krs = k_raw + (size_t)kh * (size_t)kr_hs +
                         (size_t)t * (size_t)kr_ts;
      const size_t cache = (size_t)kh * cache_hstride +
                           (size_t)pos * (size_t)dh;
      f32_to_f16_vec(krs, L.k_raw.data() + cache, dh);
      q8_0_to_hga_i8_vec(ks, k_block_stride, dh, L.k8.data() + cache,
                         L.k_scale.data() + (size_t)kh * (size_t)ms + pos);
      q8_0_to_hga_i8_vec(vs, v_block_stride, dh, L.v8.data() + cache,
                         L.v_scale.data() + (size_t)kh * (size_t)ms + pos);
    }
  });
  L.n_kv = start_pos + n_ok;
}

void hga_append(hga_session *s, int layer, int start_pos, int n_new,
                const void *k_rope, const void *k_raw, const void *v,
                hga_dtype dtype) {
  const int kvh = s->cfg.n_kv_heads;
  const int dh = s->cfg.head_dim;
  const int hs = n_new * dh;
  if (dtype == HGA_F32) {
    hga_append_f32_strided(s, layer, start_pos, n_new, (const float *)k_rope,
                           hs, dh, (const float *)k_raw, hs, dh,
                           (const float *)v, hs, dh);
    return;
  }
  /* F16 input: one convert pass into reused scratch, then INT8/F16 store. */
  const size_t n = (size_t)kvh * (size_t)n_new * (size_t)dh;
  s->scratch_qf.resize(n * 3);
  float *k_f = s->scratch_qf.data();
  float *kr_f = k_f + n;
  float *v_f = kr_f + n;
  for (size_t i = 0; i < n; ++i) {
    k_f[i] = load_f(k_rope, dtype, i);
    v_f[i] = load_f(v, dtype, i);
    kr_f[i] = k_raw ? load_f(k_raw, dtype, i) : k_f[i];
  }
  hga_append_f32_strided(s, layer, start_pos, n_new, k_f, hs, dh, kr_f, hs, dh,
                         v_f, hs, dh);
}

void hga_close_full_chunks(hga_session *s, int layer) {
  close_chunk_if_full(s, s->layers[(size_t)layer], layer);
}

int hga_truncate(hga_session *s, int layer, int new_length) {
  if (!s || layer < 0 || layer >= s->n_layers || new_length < 0 ||
      new_length > s->cfg.max_seq)
    return 0;
  Layer &L = s->layers[(size_t)layer];
  if (new_length > L.n_kv)
    return 0;
  if (new_length == L.n_kv)
    return 1;
  ++s->metrics.truncate_calls;
  s->metrics.truncated_tokens += (uint64_t)(L.n_kv - new_length);
  L.n_kv = new_length;
  /* A chunk is closed only when all of its tokens still exist. */
  L.n_closed = new_length / s->cfg.chunk_size;
  L.pack = PackedKV{};
  L.wave = WaveIndex{};
  return 1;
}

struct RouteSet {
  std::vector<int> chunks; /* absolute closed-chunk ids, routed middle */
  std::vector<int> g_chunk;
  std::vector<int> g_id;
  std::vector<std::vector<int>> retr; /* wave: cluster ids per kv head */
  std::vector<std::vector<int>> est;
};

struct RouteProfile {
  double chunk_scan = 0.0;
  double chunk_topk = 0.0;
  double group_scan = 0.0;
  double group_topk = 0.0;
  int n_mid = 0;
  int n_group_candidates = 0;
  uint64_t chunk_bytes_unique = 0;
  uint64_t chunk_bytes_logical = 0;
  uint64_t group_bytes_unique = 0;
  uint64_t group_bytes_logical = 0;
};

static void route_layer(hga_session *s, const Layer &L,
                        const float *q_pool /* [n_q_heads, dh] */,
                        int n_closed_view, int n_q, RouteSet &rs,
                        int *n_sel_chunks, int *n_open_groups,
                        RouteProfile *profile = nullptr) {
  const hga_config &c = s->cfg;
  const int H = c.n_q_heads;
  const int kvh = c.n_kv_heads;
  const int dh = c.head_dim;
  const int G = s->groups_per_chunk;
  const int rep = H / kvh;
  const int n_closed = std::min(n_closed_view, L.n_closed);
  const int f_hi = std::min(c.keep_first, n_closed);
  const int l_lo = std::max(f_hi, n_closed - c.keep_last);
  const int mid_lo = f_hi;
  const int mid_hi = l_lo;
  const int n_mid = std::max(0, mid_hi - mid_lo);
  const int Kc = std::min(hga_topk_chunks_for_query(&c, n_closed, n_q), n_mid);
  if (profile) {
    *profile = RouteProfile{};
    profile->n_mid = n_mid;
    profile->chunk_bytes_unique =
        (uint64_t)kvh * (uint64_t)n_mid * (uint64_t)dh * sizeof(float);
    profile->chunk_bytes_logical =
        (uint64_t)H * (uint64_t)n_mid * (uint64_t)dh * sizeof(float);
  }
  rs.chunks.clear();
  rs.g_chunk.clear();
  rs.g_id.clear();
  *n_sel_chunks = Kc;
  *n_open_groups = 0;
  if (Kc <= 0 || n_mid <= 0)
    return;

  const float attn_scale = s->scale;
  grow(s->scratch_route_pooled, (size_t)n_mid);
  float *pooled = s->scratch_route_pooled.data();
  for (int m = 0; m < n_mid; ++m)
    pooled[m] = -1e30f;

  /* Max-pool Q·chunk over GQA query heads. Walk mid chunks with +dh. */
  const double chunk_scan_t0 = now_ms();
  const float *qh = q_pool;
  for (int h = 0; h < H; ++h) {
    const int kh = h / rep;
    const float *ck =
        L.chunk_k.data() +
        ((size_t)kh * (size_t)s->max_chunks + (size_t)mid_lo) * (size_t)dh;
    for (int m = 0; m < n_mid; ++m) {
      const float s_c = hga_dot_f32(qh, ck, dh) * attn_scale;
      if (s_c > pooled[m])
        pooled[m] = s_c;
      ck += dh;
    }
    qh += dh;
  }
  const double chunk_scan_t1 = now_ms();
  if (profile)
    profile->chunk_scan = chunk_scan_t1 - chunk_scan_t0;
  grow(s->scratch_route_idx, (size_t)n_mid + (size_t)Kc);
  int *rel = s->scratch_route_idx.data();
  int *idxw = rel + Kc;
  const double chunk_topk_t0 = now_ms();
  topk_idx(pooled, n_mid, Kc, rel, idxw);
  /* Force-include previous chunk if it sits in the mid pool. */
  const int prev = n_closed - 1;
  if (mid_lo <= prev && prev < mid_hi) {
    const int prev_rel = prev - mid_lo;
    bool have = false;
    int worst = 0;
    for (int i = 0; i < Kc; ++i) {
      if (rel[i] == prev_rel)
        have = true;
      if (pooled[rel[i]] < pooled[rel[worst]])
        worst = i;
    }
    if (!have)
      rel[worst] = prev_rel;
  }
  rs.chunks.resize((size_t)Kc);
  for (int i = 0; i < Kc; ++i)
    rs.chunks[(size_t)i] = mid_lo + rel[i];
  const double chunk_topk_t1 = now_ms();
  if (profile)
    profile->chunk_topk = chunk_topk_t1 - chunk_topk_t0;

  const int Kg = hga_topk_groups_for_query(&c, n_closed, Kc, n_q);
  *n_open_groups = Kg;
  if (Kg <= 0)
    return;

  const int n_cand = Kc * G;
  if (profile) {
    profile->n_group_candidates = n_cand;
    profile->group_bytes_unique =
        (uint64_t)kvh * (uint64_t)n_cand * (uint64_t)dh * sizeof(float);
    profile->group_bytes_logical =
        (uint64_t)H * (uint64_t)n_cand * (uint64_t)dh * sizeof(float);
  }
  grow(s->scratch_route_sc, (size_t)n_cand);
  float *gsc = s->scratch_route_sc.data();
  for (int i = 0; i < n_cand; ++i)
    gsc[i] = -1e30f;
  const double group_scan_t0 = now_ms();
  qh = q_pool;
  for (int h = 0; h < H; ++h) {
    const int kh = h / rep;
    const float *gbase = L.group_k.data() + (size_t)kh * (size_t)s->max_chunks *
                                                (size_t)G * (size_t)dh;
    const size_t chunk_gstride = (size_t)G * (size_t)dh;
    for (int i = 0; i < Kc; ++i) {
      const float *gk = gbase + (size_t)rs.chunks[(size_t)i] * chunk_gstride;
      float *slot = gsc + (size_t)i * (size_t)G;
      for (int g = 0; g < G; ++g) {
        const float s_g = hga_dot_f32(qh, gk, dh) * attn_scale;
        if (s_g > slot[g])
          slot[g] = s_g;
        gk += dh;
      }
    }
    qh += dh;
  }
  const double group_scan_t1 = now_ms();
  if (profile)
    profile->group_scan = group_scan_t1 - group_scan_t0;
  grow(s->scratch_route_idx,
       (size_t)n_mid + (size_t)Kc + (size_t)n_cand + (size_t)Kg);
  int *grel = s->scratch_route_idx.data() + n_mid + Kc;
  int *gidxw = grel + Kg;
  const double group_topk_t0 = now_ms();
  topk_idx(gsc, n_cand, Kg, grel, gidxw);
  rs.g_chunk.resize((size_t)Kg);
  rs.g_id.resize((size_t)Kg);
  for (int i = 0; i < Kg; ++i) {
    const int t = grel[i];
    rs.g_chunk[(size_t)i] = rs.chunks[(size_t)(t / G)];
    rs.g_id[(size_t)i] = t % G;
  }
  const double group_topk_t1 = now_ms();
  if (profile)
    profile->group_topk = group_topk_t1 - group_topk_t0;
}

static void collect_spans(const hga_session *s, const Layer &L,
                          const RouteSet &rs, int n_closed_view,
                          std::vector<Span> &spans) {
  const hga_config &c = s->cfg;
  const int C = c.chunk_size;
  const int gs = c.group_size;
  const int n_closed = std::min(n_closed_view, L.n_closed);
  const int f_hi = std::min(c.keep_first, n_closed);
  const int l_lo = std::max(f_hi, n_closed - c.keep_last);
  spans.clear();
  if (f_hi > 0)
    spans.push_back({0, f_hi * C});
  if (l_lo < n_closed)
    spans.push_back({l_lo * C, (n_closed - l_lo) * C});
  if (c.levels == 2) {
    for (size_t i = 0; i < rs.g_chunk.size(); ++i) {
      const int cid = rs.g_chunk[i];
      const int gid = rs.g_id[i];
      if (cid < f_hi || cid >= l_lo)
        continue; /* already in a window */
      spans.push_back({cid * C + gid * gs, gs});
    }
  } else {
    for (int cid : rs.chunks) {
      if (cid < f_hi || cid >= l_lo)
        continue;
      spans.push_back({cid * C, C});
    }
  }
}

/* Flatten attended positions (windows + routed + active) into one key list. */
static void collect_keys(const hga_session *s, const Layer &L,
                         const std::vector<Span> &spans, int q_hi,
                         std::vector<int> &keys) {
  int n_est = 0;
  for (const Span &sp : spans) {
    const int hi = std::min(sp.start + sp.len - 1, q_hi);
    if (hi >= sp.start)
      n_est += hi - sp.start + 1;
  }
  const int act0 = L.n_closed * s->cfg.chunk_size;
  if (L.n_kv > act0)
    n_est += std::min(L.n_kv - 1, q_hi) - act0 + 1;
  keys.clear();
  if (n_est > 0)
    keys.reserve((size_t)n_est);
  for (const Span &sp : spans) {
    const int hi = std::min(sp.start + sp.len - 1, q_hi);
    for (int j = sp.start; j <= hi; ++j)
      keys.push_back(j);
  }
  for (int j = act0; j < L.n_kv && j <= q_hi; ++j)
    keys.push_back(j);
}

/* Upper bound for one query chunk after unioning the distinct routes of the
 * GQA query heads that share a KV head. KvRouter routes per query head; the
 * union avoids replicating K/V while the head-specific mask preserves the
 * exact per-head visibility. */
static int gpu_prefill_chunk_capacity(const hga_config &c, int n_closed,
                                      int n_q_chunk, int active_len) {
  const int C = std::max(1, c.chunk_size);
  const int G = std::max(1, C / std::max(1, c.group_size));
  const int rep = std::max(1, c.n_q_heads / std::max(1, c.n_kv_heads));
  const int f_hi = std::min(c.keep_first, n_closed);
  const int l_lo = std::max(f_hi, n_closed - c.keep_last);
  const int n_mid = std::max(0, l_lo - f_hi);
  const int n_fixed = (f_hi + n_closed - l_lo) * C;
  const int kc = hga_topk_chunks_for_query(&c, n_closed, n_q_chunk);
  const int kg = hga_topk_groups_for_query(&c, n_closed, kc, n_q_chunk);
  const int groups_per_head = c.levels == 2 ? kg : kc * G;
  const int union_groups = std::min(
      n_mid * G, std::min(rep * groups_per_head,
                          HGA_KV_HEAD_UNION_MULTIPLIER * groups_per_head));
  return n_fixed + union_groups * c.group_size + active_len;
}

int hga_gpu_prefill_capacity(const hga_session *s, int layer, int n_q) {
  if (!s || layer < 0 || layer >= s->n_layers || n_q <= 0)
    return 0;
  const hga_config &c = s->cfg;
  if (c.router != HGA_ROUTER_HIER)
    return 0;
  const int C = std::max(1, c.chunk_size);
  const int max_closed = (std::max(0, c.max_seq - 1)) / C;
  int result = 0;
  for (int n_closed = 0; n_closed <= max_closed; ++n_closed)
    result = std::max(result, gpu_prefill_chunk_capacity(c, n_closed, C, C));
  return result;
}

static int gpu_prefill_chunk_history_capacity(const hga_config &c, int n_closed,
                                              int n_q_chunk,
                                              int ubatch_start);

int hga_gpu_prefill_current_capacity(const hga_session *s, int layer,
                                     int start_pos, int n_q) {
  if (!s || layer < 0 || layer >= s->n_layers || start_pos < 0 || n_q <= 0)
    return 0;
  const hga_config &c = s->cfg;
  if (c.router != HGA_ROUTER_HIER)
    return 0;
  const int C = std::max(1, c.chunk_size);
  const int n_after = std::min(c.max_seq, start_pos + n_q);
  int pos = start_pos;
  int result = 0;
  while (pos < n_after) {
    const int in_chunk = pos % C;
    const int take = std::min(C - in_chunk, n_after - pos);
    const int selected =
        gpu_prefill_chunk_capacity(c, pos / C, take, in_chunk + take);
    const int direct_len = pos + take - start_pos;
    const int direct_reuse = gpu_prefill_chunk_history_capacity(
                                 c, pos / C, take, start_pos) +
                             direct_len;
    result = std::max(result, std::max(selected, direct_reuse));
    pos += take;
  }
  return result;
}

static int gpu_prefill_chunk_history_capacity(const hga_config &c, int n_closed,
                                              int n_q_chunk, int ubatch_start) {
  const int C = std::max(1, c.chunk_size);
  const int G = std::max(1, C / std::max(1, c.group_size));
  const int rep = std::max(1, c.n_q_heads / std::max(1, c.n_kv_heads));
  const int f_hi = std::min(c.keep_first, n_closed);
  const int l_lo = std::max(f_hi, n_closed - c.keep_last);
  const int n_mid = std::max(0, l_lo - f_hi);
  const auto old_len = [ubatch_start](int lo, int hi) {
    return std::max(0, std::min(hi, ubatch_start) - lo);
  };
  const int old_fixed = old_len(0, f_hi * C) + old_len(l_lo * C, n_closed * C);
  const int old_active = std::max(0, ubatch_start - n_closed * C);
  const int kc = hga_topk_chunks_for_query(&c, n_closed, n_q_chunk);
  const int kg = hga_topk_groups_for_query(&c, n_closed, kc, n_q_chunk);
  const int groups_per_head = c.levels == 2 ? kg : kc * G;
  const int union_groups = std::min(
      n_mid * G, std::min(rep * groups_per_head,
                          HGA_KV_HEAD_UNION_MULTIPLIER * groups_per_head));
  return std::min(ubatch_start,
                  old_fixed + old_active + union_groups * c.group_size);
}

int hga_gpu_prefill_current_history_capacity(const hga_session *s, int layer,
                                             int start_pos, int n_q) {
  if (!s || layer < 0 || layer >= s->n_layers || start_pos < 0 || n_q <= 0)
    return 0;
  const hga_config &c = s->cfg;
  if (c.router != HGA_ROUTER_HIER)
    return 0;
  const int C = std::max(1, c.chunk_size);
  const int n_after = std::min(c.max_seq, start_pos + n_q);
  int pos = start_pos;
  int result = 0;
  while (pos < n_after) {
    const int in_chunk = pos % C;
    const int take = std::min(C - in_chunk, n_after - pos);
    result = std::max(result, gpu_prefill_chunk_history_capacity(
                                  c, pos / C, take, start_pos));
    pos += take;
  }
  return result;
}

int hga_gpu_prefill_segment_history_capacity(
    const hga_session *s, int layer, int ubatch_start, int seg_start,
    int seg_n, int total_capacity) {
  if (!s || layer < 0 || layer >= s->n_layers || ubatch_start < 0 ||
      seg_start < ubatch_start || seg_n <= 0 || total_capacity <= 0)
    return 0;
  const hga_config &c = s->cfg;
  if (c.router != HGA_ROUTER_HIER)
    return 0;
  const int C = std::max(1, c.chunk_size);
  const int direct_len = seg_start + seg_n - ubatch_start;
  const int history_need = gpu_prefill_chunk_history_capacity(
      c, seg_start / C, seg_n, ubatch_start);
  return std::max(1, std::max(history_need, total_capacity - direct_len));
}

int hga_gpu_prefill_ubatch_history_capacity(
    const hga_session *s, int layer, int ubatch_start, int n_q,
    int total_capacity) {
  if (!s || n_q <= 0 || total_capacity <= 0)
    return 0;
  (void)ubatch_start;
  /* A united 512-token block needs more than one routed chunk's old 1552-key
   * envelope. Two envelopes retain the measured cross-chunk/head union at 8K
   * while keeping the INT8 image several times smaller than eight independent
   * chunk images. Keep this independent of start_pos so one graph is reusable
   * throughout prefill; a dynamic GPU mask hides the not-yet-populated prefix. */
  const int block_need = 2 * hga_gpu_prefill_capacity(s, layer, n_q);
  return std::max(1, std::min(total_capacity, block_need));
}

struct PrefillHeadRoute {
  std::vector<int> group_ids;      /* absolute chunk*G + group */
  std::vector<int> visible_from;   /* query offset, cumulative visibility */
  std::vector<float> group_scores; /* max requested routing logit */
  int n_chunks = 0;
};

static void route_prefill_head(const hga_session *s, const Layer &L,
                               const float *q, int q_ts, int h, int n_q,
                               int n_closed, PrefillHeadRoute &out) {
  const hga_config &c = s->cfg;
  const int G = s->groups_per_chunk;
  const int dh = c.head_dim;
  const int kh = h / std::max(1, c.n_q_heads / c.n_kv_heads);
  const int f_hi = std::min(c.keep_first, n_closed);
  const int l_lo = std::max(f_hi, n_closed - c.keep_last);
  const int n_mid = std::max(0, l_lo - f_hi);
  const int Kc = std::min(hga_topk_chunks_for_query(&c, n_closed, n_q), n_mid);
  out.group_ids.clear();
  out.visible_from.clear();
  out.group_scores.clear();
  out.n_chunks = Kc;
  if (Kc <= 0)
    return;

  std::vector<float> chunk_sc((size_t)n_q * (size_t)n_mid);
  for (int t = 0; t < n_q; ++t) {
    const float *qt = q + (size_t)t * (size_t)q_ts;
    float *row = chunk_sc.data() + (size_t)t * (size_t)n_mid;
    for (int m = 0; m < n_mid; ++m) {
      const int cid = f_hi + m;
      // TODO: dh is const - replace multiplication byinitialization and
      // increase. s->max_chunks and kh - const - precalculate.
      const float *ck =
          L.chunk_k.data() +
          ((size_t)kh * (size_t)s->max_chunks + (size_t)cid) * (size_t)dh;
      row[m] = hga_dot_f32(qt, ck, dh) * s->scale;
    }
  }
  /* Match vectorized.py exactly: every query token first requests its own
   * top-K chunks, then the materialized set is selected from the max scores
   * of those requests only.  Max-pooling every candidate directly is not
   * equivalent: it can materialize a route that no token requested. */
  std::vector<float> chunk_pool((size_t)n_mid, NEG);
  std::vector<int> chunk_req((size_t)n_q * (size_t)Kc, -1);
  std::vector<int> selected((size_t)Kc), work((size_t)n_mid);
  std::vector<int> req((size_t)Kc);
  for (int t = 0; t < n_q; ++t) {
    float *row = chunk_sc.data() + (size_t)t * (size_t)n_mid;
    topk_idx(row, n_mid, Kc, req.data(), work.data());
    for (int j = 0; j < Kc; ++j) {
      const int r = req[(size_t)j];
      chunk_req[(size_t)t * (size_t)Kc + (size_t)j] = r;
      chunk_pool[(size_t)r] = std::max(chunk_pool[(size_t)r], row[r]);
    }
  }
  topk_idx(chunk_pool.data(), n_mid, Kc, selected.data(), work.data());

  const int prev_rel = n_closed - 1 - f_hi;
  if (0 <= prev_rel && prev_rel < n_mid) {
    bool have = false;
    int worst = 0;
    for (int i = 0; i < Kc; ++i) {
      have = have || selected[(size_t)i] == prev_rel;
      if (chunk_pool[(size_t)selected[(size_t)i]] <
          chunk_pool[(size_t)selected[(size_t)worst]])
        worst = i;
    }
    if (!have)
      selected[(size_t)worst] = prev_rel;
  }

  std::vector<int> chunk_visible((size_t)Kc, n_q);
  for (int t = 0; t < n_q; ++t) {
    for (int j = 0; j < Kc; ++j) {
      // TODO: replace (size_t)t * (size_t)Kc + (size_t)j with initialization
      // and increase
      const int r = chunk_req[(size_t)t * (size_t)Kc + (size_t)j];
      for (int i = 0; i < Kc; ++i)
        if (selected[(size_t)i] == r)
          chunk_visible[(size_t)i] = std::min(chunk_visible[(size_t)i], t);
    }
  }
  if (0 <= prev_rel && prev_rel < n_mid)
    for (int i = 0; i < Kc; ++i)
      if (selected[(size_t)i] == prev_rel)
        chunk_visible[(size_t)i] = 0;

  if (c.levels != 2) {
    for (int i = 0; i < Kc; ++i)
      for (int g = 0; g < G; ++g) {
        // TODO: precalculate (f_hi + selected[(size_t)i]) * G in the outher
        // loop and then increment
        out.group_ids.push_back((f_hi + selected[(size_t)i]) * G + g);
        out.visible_from.push_back(chunk_visible[(size_t)i]);
        out.group_scores.push_back(chunk_pool[(size_t)selected[(size_t)i]]);
      }
    return;
  }

  const int n_cand = Kc * G;
  const int Kg =
      std::min(hga_topk_groups_for_query(&c, n_closed, Kc, n_q), n_cand);
  const int Kg_req = std::min(Kg / 2, n_cand);
  if (Kg <= 0 || Kg_req <= 0)
    return;

  std::vector<float> group_sc((size_t)n_q * (size_t)n_cand, NEG);
  std::vector<float> group_pool((size_t)n_cand, NEG);
  std::vector<int> group_req((size_t)n_q * (size_t)Kg_req, -1);
  std::vector<int> group_work((size_t)n_cand);
  std::vector<int> req_tmp((size_t)Kg_req);
  for (int t = 0; t < n_q; ++t) {
    const float *qt = q + (size_t)t * (size_t)q_ts;
    float *row = group_sc.data() + (size_t)t * (size_t)n_cand;
    for (int i = 0; i < Kc; ++i) {
      if (chunk_visible[(size_t)i] > t)
        continue;
      const int cid = f_hi + selected[(size_t)i];
      const float *gk =
          L.group_k.data() + ((size_t)kh * (size_t)s->max_chunks * (size_t)G +
                              (size_t)cid * (size_t)G) *
                                 (size_t)dh;
      for (int g = 0; g < G; ++g)
        row[i * G + g] = hga_dot_f32(qt, gk + (size_t)g * dh, dh) * s->scale;
    }
    topk_idx(row, n_cand, Kg_req, req_tmp.data(), group_work.data());
    for (int j = 0; j < Kg_req; ++j) {
      const int r = req_tmp[(size_t)j];
      if (row[r] <= NEG)
        continue;
      group_req[(size_t)t * (size_t)Kg_req + (size_t)j] = r;
      group_pool[(size_t)r] = std::max(group_pool[(size_t)r], row[r]);
    }
  }
  int n_valid = 0;
  for (float x : group_pool)
    n_valid += x > NEG;
  const int Kg_actual = std::min(Kg, n_valid);
  if (Kg_actual <= 0)
    return;
  std::vector<int> groups((size_t)Kg_actual);
  topk_idx(group_pool.data(), n_cand, Kg_actual, groups.data(),
           group_work.data());
  for (int r : groups) {
    int first = n_q;
    for (int t = 0; t < n_q; ++t)
      for (int j = 0; j < Kg_req; ++j)
        if (group_req[(size_t)t * (size_t)Kg_req + (size_t)j] == r)
          first = std::min(first, t);
    if (first >= n_q)
      continue;
    const int parent = r / G;
    const int cid = f_hi + selected[(size_t)parent];
    out.group_ids.push_back(cid * G + r % G);
    out.visible_from.push_back(std::max(first, chunk_visible[(size_t)parent]));
    out.group_scores.push_back(group_pool[(size_t)r]);
  }
}

int hga_gpu_prefill_i8_image_layout(const hga_session *s,
                                    int history_capacity, int graph_n_q,
                                    hga_gpu_prefill_i8_layout *out) {
  if (!s || !out || s->cfg.prec != HGA_PREC_I8 || history_capacity <= 0 ||
      graph_n_q <= 0)
    return 0;
  const size_t kv_elems = (size_t)s->cfg.n_kv_heads *
                          (size_t)history_capacity * (size_t)s->cfg.head_dim;
  const size_t scale_elems =
      (size_t)s->cfg.n_kv_heads * (size_t)history_capacity;
  (void)graph_n_q;
  out->k_offset = 0;
  out->v_offset = kv_elems;
  out->k_scale_offset = (2 * kv_elems + alignof(float) - 1) &
                        ~(size_t)(alignof(float) - 1);
  out->v_scale_offset = out->k_scale_offset +
                        scale_elems * sizeof(float);
  out->n_bytes = out->v_scale_offset + scale_elems * sizeof(float);
  out->visibility_offset = out->n_bytes;
  out->direct_visibility_offset = out->n_bytes;
  return 1;
}

int hga_gpu_verify_capacity(const hga_session *s, int layer, int graph_n_q) {
  if (!s || layer < 0 || layer >= s->n_layers || graph_n_q <= 0 ||
      graph_n_q > 8 || s->cfg.router != HGA_ROUTER_HIER)
    return 0;
  const hga_config &c = s->cfg;
  const int C = std::max(1, c.chunk_size);
  const int G = std::max(1, C / std::max(1, c.group_size));
  const int max_closed = (std::max(0, c.max_seq - 1)) / C;
  int result = 0;
  for (int n_closed = 0; n_closed <= max_closed; ++n_closed) {
    const int f_hi = std::min(c.keep_first, n_closed);
    const int l_lo = std::max(f_hi, n_closed - c.keep_last);
    const int n_mid = std::max(0, l_lo - f_hi);
    const int n_fixed = (f_hi + n_closed - l_lo) * C;
    const int kc = hga_topk_chunks_for_query(&c, n_closed, graph_n_q);
    const int kg = hga_topk_groups_for_query(&c, n_closed, kc, graph_n_q);
    const int n_routed = c.levels == 2
        ? std::min(n_mid * G, kg) * c.group_size
        : std::min(n_mid, kc) * C;
    /* At most one not-yet-closed chunk is historical at the start of a
     * short verify batch. Direct verify tokens are supplied by llama.cpp. */
    result = std::max(result, n_fixed + n_routed + C);
  }
  return std::min(result, c.max_seq);
}

int hga_gpu_verify_i8_image_layout(const hga_session *s,
                                   int history_capacity, int graph_n_q,
                                   hga_gpu_verify_i8_layout *out) {
  if (!out || graph_n_q <= 0 || graph_n_q > 8)
    return 0;
  hga_gpu_prefill_i8_layout kv{};
  if (!hga_gpu_prefill_i8_image_layout(s, history_capacity, graph_n_q, &kv))
    return 0;
  const size_t mask_offset = (kv.n_bytes + alignof(uint16_t) - 1) &
                             ~(size_t)(alignof(uint16_t) - 1);
  const size_t n_keys = (size_t)history_capacity + (size_t)graph_n_q;
  out->k_offset = kv.k_offset;
  out->v_offset = kv.v_offset;
  out->k_scale_offset = kv.k_scale_offset;
  out->v_scale_offset = kv.v_scale_offset;
  out->mask_offset = mask_offset;
  out->n_bytes = mask_offset + n_keys * (size_t)graph_n_q * sizeof(uint16_t);
  return 1;
}

static int hga_prepare_gpu_prefill_strided_impl(
    hga_session *s, int layer, int start_pos, int n_q, const float *q,
    int q_head_stride, int q_tok_stride, const float *k_rope, int k_head_stride,
    int k_tok_stride, const float *k_raw, int kr_head_stride, int kr_tok_stride,
    const float *v, int v_head_stride, int v_tok_stride, void *image,
    size_t image_bytes, int total_capacity, bool raw_i8, hga_stats *stats) {
  if (!s || layer < 0 || layer >= s->n_layers || n_q <= 0 || !q || !k_rope ||
      !v || !image || total_capacity <= 0)
    return 0;

  if (raw_i8 && s->cfg.prec != HGA_PREC_I8)
    return 0;

  const int H = s->cfg.n_q_heads;
  const int kvh = s->cfg.n_kv_heads;
  const int dh = s->cfg.head_dim;
  const int C = s->cfg.chunk_size;
  const int n_segments = (start_pos % C + n_q + C - 1) / C;
  const size_t direct_visibility_elems = (size_t)H * (size_t)n_q;
  size_t required_bytes = 0;
  int shape_done = 0;
  for (int seg = 0; seg < n_segments; ++seg) {
    const int seg_start = start_pos + shape_done;
    const int seg_n = std::min(C - seg_start % C, n_q - shape_done);
    const int capacity = hga_gpu_prefill_segment_history_capacity(
        s, layer, start_pos, seg_start, seg_n, total_capacity);
    if (capacity <= 0)
      return 0;
    if (raw_i8) {
      hga_gpu_prefill_i8_layout layout{};
      if (!hga_gpu_prefill_i8_image_layout(s, capacity, n_q, &layout))
        return 0;
      required_bytes += layout.n_bytes;
    } else {
      const size_t kv_elems =
          (size_t)kvh * (size_t)capacity * (size_t)dh;
      const size_t visibility_elems = (size_t)H * (size_t)capacity;
      required_bytes += (2 * kv_elems + visibility_elems +
                         direct_visibility_elems) * sizeof(uint16_t);
    }
    shape_done += seg_n;
  }
  if (image_bytes < required_bytes)
    return 0;
  const int estimated =
      hga_gpu_prefill_current_capacity(s, layer, start_pos, n_q);
  if (estimated > total_capacity)
    return 0;

  const double t0 = now_ms();
  Layer &L = s->layers[(size_t)layer];
  const bool i8 = s->cfg.prec == HGA_PREC_I8;
  const int ms = s->cfg.max_seq;
  const size_t cache_hstride = (size_t)ms * (size_t)dh;
  const uint16_t invisible = f32_to_f16((float)C);
  uint8_t *image_bytes_out = (uint8_t *)image;
  uint16_t *image_f16 = (uint16_t *)image;
  /* The legacy F16 path has no separate scale which can safely neutralize
   * untouched payload. Production INT8 staging below never clears K/V. */
  if (!raw_i8)
    std::fill(image_f16, image_f16 + required_bytes / sizeof(uint16_t),
              (uint16_t)0);
  size_t image_offset_bytes = 0;
  int done = 0;
  int max_keys = 0;
  int max_materialized = 0;
  int max_chunks = 0;
  int max_groups = 0;
  double route_ms = 0.0;
  for (int seg = 0; seg < n_segments; ++seg) {
    const int seg_start = start_pos + done;
    const int seg_n = std::min(C - seg_start % C, n_q - done);
    const int capacity = hga_gpu_prefill_segment_history_capacity(
        s, layer, start_pos, seg_start, seg_n, total_capacity);
    const size_t kv_elems = (size_t)kvh * (size_t)capacity * (size_t)dh;
    const size_t visibility_elems = (size_t)H * (size_t)capacity;
    hga_gpu_prefill_i8_layout i8_layout{};
    if (raw_i8 &&
        !hga_gpu_prefill_i8_image_layout(s, capacity, n_q, &i8_layout))
      return 0;
    const size_t segment_bytes = raw_i8
        ? i8_layout.n_bytes
        : (2 * kv_elems + visibility_elems + direct_visibility_elems) *
              sizeof(uint16_t);
    hga_append_f32_strided(
        s, layer, seg_start, seg_n,
        k_rope + (size_t)done * (size_t)k_tok_stride, k_head_stride,
        k_tok_stride,
        k_raw ? k_raw + (size_t)done * (size_t)kr_tok_stride : nullptr,
        kr_head_stride, kr_tok_stride, v + (size_t)done * (size_t)v_tok_stride,
        v_head_stride, v_tok_stride);

    const int n_closed = L.n_closed;
    const int f_hi = std::min(s->cfg.keep_first, n_closed);
    const int l_lo = std::max(f_hi, n_closed - s->cfg.keep_last);
    const int act0 = n_closed * C;
    const int q_last = seg_start + seg_n - 1;
    const int G = s->groups_per_chunk;
    const int rep = H / kvh;
    const double tr0 = now_ms();
    std::vector<PrefillHeadRoute> routes((size_t)H);
#if defined(_OPENMP)
#pragma omp parallel for num_threads(std::min(s->cfg.n_threads, H))            \
    schedule(static)
#endif
    for (int h = 0; h < H; ++h)
      route_prefill_head(s, L,
                         q + (size_t)h * (size_t)q_head_stride +
                             (size_t)done * (size_t)q_tok_stride,
                         q_tok_stride, h, seg_n, n_closed, routes[(size_t)h]);
    route_ms += now_ms() - tr0;
    for (const PrefillHeadRoute &hr : routes) {
      max_chunks = std::max(max_chunks, hr.n_chunks);
    }

    /* One materialized KV union serves the `rep` query heads of each KV head.
     * If their distinct routes exceed the assumed 50%-overlap budget, keep
     * the groups with the largest sum of per-head normalized routing weights.
     * Head-specific first-visible thresholds below still decide which of the
     * retained groups each query head may actually attend to. */
    std::vector<std::vector<int>> selected_groups((size_t)kvh);
    std::vector<std::vector<uint8_t>> group_selected(
        (size_t)kvh,
        std::vector<uint8_t>((size_t)s->max_chunks * (size_t)G, 0));
    const int groups_total = s->max_chunks * G;
    const int per_head_group_budget = hga_topk_groups_for_query(
        &s->cfg, n_closed, routes.empty() ? 0 : routes[0].n_chunks, seg_n);
    const int union_group_cap =
        HGA_KV_HEAD_UNION_MULTIPLIER * per_head_group_budget;
    for (int kh = 0; kh < kvh; ++kh) {
      std::vector<float> summed_attention((size_t)groups_total, 0.f);
      std::vector<uint8_t> seen((size_t)groups_total, 0);
      for (int h = kh * rep; h < (kh + 1) * rep; ++h) {
        const PrefillHeadRoute &hr = routes[(size_t)h];
        if (hr.group_ids.empty())
          continue;
        float row_max = -1e30f;
        for (float score : hr.group_scores)
          row_max = std::max(row_max, score);
        float denom = 0.f;
        for (float score : hr.group_scores)
          denom += std::exp(std::max(-80.f, score - row_max));
        denom = std::max(denom, 1e-30f);
        for (size_t i = 0; i < hr.group_ids.size(); ++i) {
          const int gid = hr.group_ids[i];
          if (gid < 0 || gid >= groups_total)
            continue;
          seen[(size_t)gid] = 1;
          summed_attention[(size_t)gid] +=
              std::exp(std::max(-80.f, hr.group_scores[i] - row_max)) / denom;
        }
      }
      std::vector<int> candidates;
      candidates.reserve((size_t)groups_total);
      for (int gid = 0; gid < groups_total; ++gid)
        if (seen[(size_t)gid])
          candidates.push_back(gid);
      const int keep = std::min((int)candidates.size(), union_group_cap);
      std::partial_sort(candidates.begin(), candidates.begin() + keep,
                        candidates.end(), [&](int a, int b) {
                          return summed_attention[(size_t)a] >
                                 summed_attention[(size_t)b];
                        });
      candidates.resize((size_t)keep);
      selected_groups[(size_t)kh].swap(candidates);
      for (int gid : selected_groups[(size_t)kh])
        group_selected[(size_t)kh][(size_t)gid] = 1;
      max_groups = std::max(max_groups, keep);
    }

    std::vector<std::vector<int>> keys((size_t)kvh);
    std::vector<std::vector<int>> full_keys((size_t)kvh);
    for (int kh = 0; kh < kvh; ++kh) {
      std::vector<int> &kk = keys[(size_t)kh];
      for (int p = 0; p < f_hi * C; ++p)
        kk.push_back(p);
      for (int p = l_lo * C; p < n_closed * C; ++p)
        kk.push_back(p);
      for (int gid : selected_groups[(size_t)kh]) {
        const int base = (gid / G) * C + (gid % G) * s->cfg.group_size;
        for (int p = base; p < base + s->cfg.group_size; ++p)
          kk.push_back(p);
      }
      for (int p = act0; p <= q_last; ++p)
        kk.push_back(p);
      std::sort(kk.begin(), kk.end());
      kk.erase(std::unique(kk.begin(), kk.end()), kk.end());
      full_keys[(size_t)kh] = kk;
      kk.erase(std::lower_bound(kk.begin(), kk.end(), start_pos), kk.end());
      if ((int)kk.size() > capacity)
        return 0;
      max_keys = std::max(max_keys, (int)kk.size());
      max_materialized =
          std::max(max_materialized, (int)kk.size() + q_last - start_pos + 1);
    }

    uint8_t *seg_bytes = image_bytes_out + image_offset_bytes;
    uint16_t *seg_f16 = (uint16_t *)seg_bytes;
    int8_t *kout_i8 = raw_i8 ? (int8_t *)(seg_bytes + i8_layout.k_offset)
                             : nullptr;
    int8_t *vout_i8 = raw_i8 ? (int8_t *)(seg_bytes + i8_layout.v_offset)
                             : nullptr;
    float *kscale_out = raw_i8
        ? (float *)(seg_bytes + i8_layout.k_scale_offset) : nullptr;
    float *vscale_out = raw_i8
        ? (float *)(seg_bytes + i8_layout.v_scale_offset) : nullptr;
    uint16_t *kout = raw_i8 ? nullptr : seg_f16;
    uint16_t *vout = raw_i8 ? nullptr : seg_f16 + kv_elems;
    uint16_t *first_visible = raw_i8
        ? (uint16_t *)(seg_bytes + i8_layout.visibility_offset)
        : seg_f16 + 2 * kv_elems;
    uint16_t *direct_first_visible = raw_i8
        ? (uint16_t *)(seg_bytes + i8_layout.direct_visibility_offset)
        : first_visible + visibility_elems;
    if (raw_i8) {
      const size_t scale_elems = (size_t)kvh * (size_t)capacity;
      /* Zero scales, not the multi-megabyte payload. Garbage INT8 values are
       * finite and become exact zero after dequantization, so masked padding
       * cannot inject a NaN into softmax. */
      std::fill(kscale_out, kscale_out + scale_elems, 0.f);
      std::fill(vscale_out, vscale_out + scale_elems, 0.f);
    }
    std::fill(first_visible, first_visible + visibility_elems, invisible);
    std::fill(direct_first_visible,
              direct_first_visible + direct_visibility_elems, invisible);
/* Four KV heads already use the same four-worker team as before the split. */
#if defined(_OPENMP)
#pragma omp parallel for num_threads(std::min(s->cfg.n_threads, kvh)) schedule(static)
#endif
    for (int kh = 0; kh < kvh; ++kh) {
      const std::vector<int> &kk = keys[(size_t)kh];
      for (int x = 0; x < (int)kk.size(); ++x) {
        const int pos = kk[(size_t)x];
        const size_t out_vec =
            ((size_t)kh * (size_t)capacity + (size_t)x) * (size_t)dh;
        uint16_t *kd = raw_i8 ? nullptr : kout + out_vec;
        if (raw_i8) {
          const int8_t *ks = L.k8.data() + (size_t)kh * cache_hstride +
                             (size_t)pos * (size_t)dh;
          const int8_t *vs = L.v8.data() + (size_t)kh * cache_hstride +
                             (size_t)pos * (size_t)dh;
          std::memcpy(kout_i8 + out_vec, ks, (size_t)dh);
          std::memcpy(vout_i8 + out_vec, vs, (size_t)dh);
          const size_t scale_out =
              (size_t)kh * (size_t)capacity + (size_t)x;
          kscale_out[scale_out] =
              L.k_scale[(size_t)kh * (size_t)ms + (size_t)pos];
          vscale_out[scale_out] =
              L.v_scale[(size_t)kh * (size_t)ms + (size_t)pos];
        } else if (i8) {
          const int8_t *ks = L.k8.data() + (size_t)kh * cache_hstride +
                             (size_t)pos * (size_t)dh;
          const int8_t *vs = L.v8.data() + (size_t)kh * cache_hstride +
                             (size_t)pos * (size_t)dh;
          const float kscale = L.k_scale[(size_t)kh * (size_t)ms + (size_t)pos];
          const float vscale = L.v_scale[(size_t)kh * (size_t)ms + (size_t)pos];
          i8_scale_to_f16_vec(ks, kscale, kd, dh);
          for (int d = 0; d < dh; ++d)
            vout[((size_t)kh * (size_t)dh + (size_t)d) * (size_t)capacity +
                 (size_t)x] = f32_to_f16((float)vs[d] * vscale);
        } else {
          const uint16_t *ks = L.k.data() + (size_t)kh * cache_hstride +
                               (size_t)pos * (size_t)dh;
          const uint16_t *vs = L.v.data() + (size_t)kh * cache_hstride +
                               (size_t)pos * (size_t)dh;
          std::memcpy(kd, ks, (size_t)dh * sizeof(uint16_t));
          for (int d = 0; d < dh; ++d)
            vout[((size_t)kh * (size_t)dh + (size_t)d) * (size_t)capacity +
                 (size_t)x] = vs[d];
        }
      }
    }

    hga_pack_parallel_for(s, 0, H, 1, [&](int h) {
      const int kh = h / rep;
      const std::vector<int> &kk = keys[(size_t)kh];
      std::vector<int> first_by_group((size_t)s->max_chunks * (size_t)G, seg_n);
      const PrefillHeadRoute &hr = routes[(size_t)h];
      for (size_t i = 0; i < hr.group_ids.size(); ++i)
        first_by_group[(size_t)hr.group_ids[i]] = hr.visible_from[i];
      uint16_t *head_visibility = first_visible + (size_t)h * (size_t)capacity;
      for (int x = 0; x < (int)kk.size(); ++x) {
        const int pos = kk[(size_t)x];
        const bool fixed =
            pos < f_hi * C || (pos >= l_lo * C && pos < n_closed * C);
        int first = C;
        if (fixed) {
          first = 0;
        } else if (pos >= act0 && pos <= q_last) {
          first = std::max(0, pos - seg_start);
        } else if (pos < act0) {
          first = first_by_group[(size_t)(pos / C * G +
                                          (pos % C) / s->cfg.group_size)];
        }
        head_visibility[x] = f32_to_f16((float)std::min(C, first));
      }

      uint16_t *head_direct = direct_first_visible + (size_t)h * (size_t)n_q;
      for (int p = start_pos; p <= q_last; ++p) {
        const bool fixed = p < f_hi * C || (p >= l_lo * C && p < n_closed * C);
        int first = C;
        if (fixed) {
          first = 0;
        } else if (p >= act0) {
          first = std::max(0, p - seg_start);
        } else {
          const int gid = p / C * G + (p % C) / s->cfg.group_size;
          if (gid >= 0 && gid < (int)group_selected[(size_t)kh].size() &&
              group_selected[(size_t)kh][(size_t)gid])
            first = first_by_group[(size_t)gid];
        }
        head_direct[(size_t)(p - start_pos)] =
            f32_to_f16((float)std::min(C, first));
      }
    });
    s->last_keys = full_keys.empty() ? std::vector<int>{} : full_keys[0];
    s->last_keys_layer = layer;
    hga_close_full_chunks(s, layer);
    image_offset_bytes += segment_bytes;
    done += seg_n;
  }
  const double t_pack = now_ms();

  if (stats) {
    std::memset(stats, 0, sizeof(*stats));
    stats->n_kv = L.n_kv;
    stats->n_closed_chunks = L.n_closed;
    stats->n_selected_chunks = max_chunks;
    stats->n_opened_groups = max_groups;
    stats->n_attended_tokens = max_materialized;
    stats->sparsity =
        L.n_kv > 0 ? (float)max_materialized / (float)L.n_kv : 1.f;
    stats->ms_route = route_ms;
    stats->ms_pack = t_pack - t0 - route_ms;
  }
  return max_materialized;
}

/* United prefill staging: route every logical chunk as before, then
 * materialize one historical KV union for the complete physical ubatch.
 * Production uses INT8; F16 is retained as a controlled prefill A/B. */
static int hga_prepare_gpu_prefill_ubatch_strided(
    hga_session *s, int layer, int start_pos, int n_q, const float *q,
    int q_head_stride, int q_tok_stride, const float *k_rope,
    int k_head_stride, int k_tok_stride, const float *k_raw,
    int kr_head_stride, int kr_tok_stride, const float *v,
    int v_head_stride, int v_tok_stride, const uint8_t *k_q8,
    int k_q8_block_stride, int k_q8_head_stride, int k_q8_tok_stride,
    const uint8_t *v_q8, int v_q8_block_stride, int v_q8_head_stride,
    int v_q8_tok_stride, void *image, size_t image_bytes, int history_capacity,
    bool stage_i8, hga_stats *stats) {
  const bool input_q8 = k_q8 && v_q8;
  if (!s || layer < 0 || layer >= s->n_layers || start_pos < 0 || n_q <= 0 ||
      !q || (!input_q8 && (!k_rope || !v)) || (input_q8 && !k_raw) || !image ||
      history_capacity <= 0 ||
      (stage_i8 ? s->cfg.prec != HGA_PREC_I8
                : s->cfg.prec != HGA_PREC_F16))
    return 0;

  const int H = s->cfg.n_q_heads;
  const int kvh = s->cfg.n_kv_heads;
  const int dh = s->cfg.head_dim;
  const int C = std::max(1, s->cfg.chunk_size);
  const int G = s->groups_per_chunk;
  const int rep = H / kvh;
  const int capacity = history_capacity;
  hga_gpu_prefill_i8_layout layout{};
  const size_t f16_kv_elems =
      2 * (size_t)kvh * (size_t)capacity * (size_t)dh;
  const size_t required_bytes = stage_i8
      ? (hga_gpu_prefill_i8_image_layout(s, capacity, n_q, &layout)
             ? layout.n_bytes
             : 0)
      : f16_kv_elems * sizeof(uint16_t);
  if (capacity <= 0 || required_bytes == 0 || image_bytes < required_bytes)
    return 0;

  const double t0 = now_ms();
  Layer &L = s->layers[(size_t)layer];
  std::vector<std::vector<PrefillHeadRoute>> segment_routes;
  std::vector<uint8_t> fixed((size_t)start_pos, 0);
  int done = 0;
  int max_chunks = 0;
  double route_ms = 0.0;
  double append_ms = 0.0;
  double close_ms = 0.0;
  while (done < n_q) {
    const int seg_start = start_pos + done;
    const int seg_n = std::min(C - seg_start % C, n_q - done);
    const double ta0 = now_ms();
    if (input_q8) {
      hga_append_q8_0_strided(
          s, layer, seg_start, seg_n,
          k_q8 + (size_t)done * (size_t)k_q8_tok_stride,
          k_q8_block_stride, k_q8_head_stride, k_q8_tok_stride,
          k_raw + (size_t)done * (size_t)kr_tok_stride, kr_head_stride,
          kr_tok_stride, v_q8 + (size_t)done * (size_t)v_q8_tok_stride,
          v_q8_block_stride, v_q8_head_stride, v_q8_tok_stride);
    } else {
      hga_append_f32_strided(
          s, layer, seg_start, seg_n,
          k_rope + (size_t)done * (size_t)k_tok_stride, k_head_stride,
          k_tok_stride,
          k_raw ? k_raw + (size_t)done * (size_t)kr_tok_stride : nullptr,
          kr_head_stride, kr_tok_stride,
          v + (size_t)done * (size_t)v_tok_stride, v_head_stride,
          v_tok_stride);
    }
    append_ms += now_ms() - ta0;

    const int n_closed = L.n_closed;
    const int f_hi = std::min(s->cfg.keep_first, n_closed);
    const int l_lo = std::max(f_hi, n_closed - s->cfg.keep_last);
    const int act0 = n_closed * C;
    const auto mark_fixed = [&](int lo, int hi) {
      lo = std::max(0, lo);
      hi = std::min(start_pos, hi);
      for (int p = lo; p < hi; ++p)
        fixed[(size_t)p] = 1;
    };
    mark_fixed(0, f_hi * C);
    mark_fixed(l_lo * C, n_closed * C);
    mark_fixed(act0, seg_start + seg_n);

    segment_routes.emplace_back((size_t)H);
    std::vector<PrefillHeadRoute> &routes = segment_routes.back();
    const double tr0 = now_ms();
#if defined(_OPENMP)
#pragma omp parallel for num_threads(std::min(s->cfg.n_threads, H))            \
    schedule(static)
#endif
    for (int h = 0; h < H; ++h)
      route_prefill_head(s, L,
                         q + (size_t)h * (size_t)q_head_stride +
                             (size_t)done * (size_t)q_tok_stride,
                         q_tok_stride, h, seg_n, n_closed, routes[(size_t)h]);
    route_ms += now_ms() - tr0;
    for (const PrefillHeadRoute &hr : routes)
      max_chunks = std::max(max_chunks, hr.n_chunks);

    const double tc0 = now_ms();
    hga_close_full_chunks(s, layer);
    close_ms += now_ms() - tc0;
    done += seg_n;
  }

  const double tclear0 = now_ms();
  int8_t *kout = stage_i8
      ? (int8_t *)((uint8_t *)image + layout.k_offset) : nullptr;
  int8_t *vout = stage_i8
      ? (int8_t *)((uint8_t *)image + layout.v_offset) : nullptr;
  float *kscale_out = stage_i8
      ? (float *)((uint8_t *)image + layout.k_scale_offset) : nullptr;
  float *vscale_out = stage_i8
      ? (float *)((uint8_t *)image + layout.v_scale_offset) : nullptr;
  uint16_t *kout16 = stage_i8 ? nullptr : (uint16_t *)image;
  uint16_t *vout16 = stage_i8
      ? nullptr
      : kout16 + (size_t)kvh * (size_t)capacity * (size_t)dh;
  const size_t scale_elems = (size_t)kvh * (size_t)capacity;
  /* Do not clear the multi-megabyte payload. A zero scale turns unused raw
   * bytes into exact zero after the GPU cast/multiply. */
  if (stage_i8) {
    std::fill(kscale_out, kscale_out + scale_elems, 0.f);
    std::fill(vscale_out, vscale_out + scale_elems, 0.f);
  }
  const double scale_clear_ms = now_ms() - tclear0;

  const double tunion0 = now_ms();
  std::vector<std::vector<int>> keys((size_t)kvh);
  int requested_total = 0;
  int union_total = 0;
  int retained_total = 0;
  int head_uses_total = 0;
  int chunk_uses_total = 0;
  int max_request_fanout = 0;
  int max_head_fanout = 0;
  int max_chunk_fanout = 0;
  int selected_history_total = 0;
  int max_selected_history = 0;
  int min_uniform_limit = std::numeric_limits<int>::max();
  int max_groups = 0;

  for (int kh = 0; kh < kvh; ++kh) {
    std::vector<std::vector<int>> routes;
    routes.reserve(segment_routes.size() * (size_t)rep);
    int max_depth = 0;
    for (const auto &segment : segment_routes)
      for (int h = kh * rep; h < (kh + 1) * rep; ++h) {
        routes.emplace_back();
        std::vector<int> &historical = routes.back();
        for (int gid : segment[(size_t)h].group_ids) {
          const int base = (gid / G) * C +
                           (gid % G) * s->cfg.group_size;
          if (base < start_pos)
            historical.push_back(gid);
        }
        requested_total += (int)historical.size();
        max_depth = std::max(max_depth, (int)historical.size());
      }

    const int groups_total = s->max_chunks * G;
    std::vector<uint8_t> requested((size_t)groups_total, 0);
    std::vector<uint16_t> request_fanout((size_t)groups_total, 0);
    std::vector<uint64_t> head_mask((size_t)groups_total, 0);
    std::vector<uint64_t> chunk_mask((size_t)groups_total, 0);
    for (size_t r = 0; r < routes.size(); ++r) {
      const int head = (int)(r % (size_t)rep);
      const int chunk = (int)(r / (size_t)rep);
      for (int gid : routes[r])
        if (gid >= 0 && gid < groups_total) {
          requested[(size_t)gid] = 1;
          ++request_fanout[(size_t)gid];
          if (head < 64)
            head_mask[(size_t)gid] |= UINT64_C(1) << head;
          if (chunk < 64)
            chunk_mask[(size_t)gid] |= UINT64_C(1) << chunk;
        }
    }
    union_total += (int)std::count(requested.begin(), requested.end(),
                                  (uint8_t)1);
    for (int gid = 0; gid < groups_total; ++gid) {
      if (!requested[(size_t)gid])
        continue;
      const int heads = __builtin_popcountll(head_mask[(size_t)gid]);
      const int chunks = __builtin_popcountll(chunk_mask[(size_t)gid]);
      head_uses_total += heads;
      chunk_uses_total += chunks;
      max_request_fanout =
          std::max(max_request_fanout, (int)request_fanout[(size_t)gid]);
      max_head_fanout = std::max(max_head_fanout, heads);
      max_chunk_fanout = std::max(max_chunk_fanout, chunks);
    }

    const auto build_prefix = [&](int depth, std::vector<uint8_t> *groups,
                                  std::vector<uint8_t> *positions) {
      std::vector<uint8_t> local_groups((size_t)groups_total, 0);
      std::vector<uint8_t> local_positions = fixed;
      for (const std::vector<int> &route : routes) {
        const int take = std::min(depth, (int)route.size());
        for (int i = 0; i < take; ++i) {
          const int gid = route[(size_t)i];
          if (gid < 0 || gid >= groups_total)
            continue;
          local_groups[(size_t)gid] = 1;
          const int base = (gid / G) * C +
                           (gid % G) * s->cfg.group_size;
          for (int p = base;
               p < base + s->cfg.group_size && p < start_pos; ++p)
            if (p >= 0)
              local_positions[(size_t)p] = 1;
        }
      }
      const int count = (int)std::count(local_positions.begin(),
                                        local_positions.end(), (uint8_t)1);
      if (groups)
        groups->swap(local_groups);
      if (positions)
        positions->swap(local_positions);
      return count;
    };

    int lo = 0;
    int hi = max_depth + 1;
    while (lo + 1 < hi) {
      const int mid = (lo + hi) / 2;
      if (build_prefix(mid, nullptr, nullptr) <= capacity)
        lo = mid;
      else
        hi = mid;
    }
    int uniform_limit = lo;
    std::vector<uint8_t> selected;
    std::vector<uint8_t> positions;
    int used = build_prefix(uniform_limit, &selected, &positions);
    if (used > capacity)
      return 0;

    /* Spend any remainder round-robin. Each route may only advance to its
     * next locally ranked group, so incomparable route scores never meet. */
    std::vector<int> depth(routes.size(), uniform_limit);
    std::vector<uint8_t> blocked(routes.size(), 0);
    bool progress = true;
    while (progress) {
      progress = false;
      for (size_t r = 0; r < routes.size(); ++r) {
        if (blocked[r] || depth[r] >= (int)routes[r].size())
          continue;
        const int gid = routes[r][(size_t)depth[r]];
        if (gid < 0 || gid >= groups_total) {
          ++depth[r];
          progress = true;
          continue;
        }
        const int base = (gid / G) * C +
                         (gid % G) * s->cfg.group_size;
        int extra = 0;
        for (int p = base;
             p < base + s->cfg.group_size && p < start_pos; ++p)
          if (p >= 0 && !positions[(size_t)p])
            ++extra;
        if (used + extra > capacity) {
          blocked[r] = 1;
          continue;
        }
        selected[(size_t)gid] = 1;
        for (int p = base;
             p < base + s->cfg.group_size && p < start_pos; ++p)
          if (p >= 0)
            positions[(size_t)p] = 1;
        used += extra;
        ++depth[r];
        progress = true;
      }
    }

    selected_history_total += used;
    max_selected_history = std::max(max_selected_history, used);

    std::vector<int> &kk = keys[(size_t)kh];
    const int history_valid = std::min(start_pos, capacity);
    /* A common valid width lets one causal mask serve every KV/query head.
     * Fill rare spare slots with additional real history, which is safe for a
     * model trained with dense historical visibility. */
    for (int p = start_pos - 1; used < history_valid && p >= 0; --p) {
      if (!positions[(size_t)p]) {
        positions[(size_t)p] = 1;
        ++used;
      }
    }
    kk.reserve((size_t)history_valid);
    for (int p = 0; p < start_pos; ++p)
      if (positions[(size_t)p])
        kk.push_back(p);
    if ((int)kk.size() != history_valid)
      return 0;
    const int retained = (int)std::count(selected.begin(), selected.end(),
                                         (uint8_t)1);
    retained_total += retained;
    max_groups = std::max(max_groups, retained);
    min_uniform_limit = std::min(min_uniform_limit, uniform_limit);
  }
  const double union_ms = now_ms() - tunion0;

  const int ms = s->cfg.max_seq;
  const size_t cache_hstride = (size_t)ms * (size_t)dh;
  const double tcopy0 = now_ms();
#if defined(_OPENMP)
#pragma omp parallel for num_threads(std::min(s->cfg.n_threads, kvh)) schedule(static)
#endif
  for (int kh = 0; kh < kvh; ++kh) {
    const std::vector<int> &kk = keys[(size_t)kh];
    for (int x = 0; x < (int)kk.size(); ++x) {
      const int pos = kk[(size_t)x];
      const size_t out_vec =
          ((size_t)kh * (size_t)capacity + (size_t)x) * (size_t)dh;
      if (stage_i8) {
        const int8_t *ks = L.k8.data() + (size_t)kh * cache_hstride +
                           (size_t)pos * (size_t)dh;
        const int8_t *vs = L.v8.data() + (size_t)kh * cache_hstride +
                           (size_t)pos * (size_t)dh;
        std::memcpy(kout + out_vec, ks, (size_t)dh);
        std::memcpy(vout + out_vec, vs, (size_t)dh);
        const size_t scale_out =
            (size_t)kh * (size_t)capacity + (size_t)x;
        kscale_out[scale_out] =
            L.k_scale[(size_t)kh * (size_t)ms + (size_t)pos];
        vscale_out[scale_out] =
            L.v_scale[(size_t)kh * (size_t)ms + (size_t)pos];
      } else {
        const uint16_t *ks = L.k.data() + (size_t)kh * cache_hstride +
                             (size_t)pos * (size_t)dh;
        const uint16_t *vs = L.v.data() + (size_t)kh * cache_hstride +
                             (size_t)pos * (size_t)dh;
        std::memcpy(kout16 + out_vec, ks, (size_t)dh * sizeof(uint16_t));
        std::memcpy(vout16 + out_vec, vs, (size_t)dh * sizeof(uint16_t));
      }
    }
  }
  const double kv_copy_ms = now_ms() - tcopy0;

  s->last_keys = keys.empty() ? std::vector<int>{} : keys[0];
  s->last_keys_layer = layer;
  const int max_history = [&] {
    int n = 0;
    for (const auto &kk : keys)
      n = std::max(n, (int)kk.size());
    return n;
  }();
  const double t_pack = now_ms();
  if (stats) {
    std::memset(stats, 0, sizeof(*stats));
    stats->n_kv = L.n_kv;
    stats->n_closed_chunks = L.n_closed;
    stats->n_selected_chunks = max_chunks;
    stats->n_opened_groups = max_groups;
    stats->n_attended_tokens = max_history + n_q;
    stats->sparsity = L.n_kv > 0
        ? (float)(max_history + n_q) / (float)L.n_kv : 1.f;
    stats->ms_route = route_ms;
    stats->ms_pack = t_pack - t0 - route_ms;
    stats->ms_prefill_append = append_ms;
    stats->ms_prefill_close = close_ms;
    stats->ms_prefill_union = union_ms;
    stats->ms_prefill_scale_clear = scale_clear_ms;
    stats->ms_prefill_kv_copy = kv_copy_ms;
    stats->ms_prefill_pack_other = std::max(
        0.0, stats->ms_pack - append_ms - close_ms - union_ms -
                 scale_clear_ms - kv_copy_ms);
    stats->n_route_group_requests = requested_total;
    stats->n_route_group_union = union_total;
    stats->n_route_group_retained = retained_total;
    stats->n_route_topk_limit =
        min_uniform_limit == std::numeric_limits<int>::max()
            ? 0 : min_uniform_limit;
    stats->route_group_overlap = requested_total > 0
        ? 1.f - (float)union_total / (float)requested_total : 0.f;
    stats->n_route_group_head_uses = head_uses_total;
    stats->n_route_group_chunk_uses = chunk_uses_total;
    stats->n_route_group_max_requests = max_request_fanout;
    stats->n_route_group_max_heads = max_head_fanout;
    stats->n_route_group_max_chunks = max_chunk_fanout;
    stats->n_route_history_selected = selected_history_total;
    stats->n_route_history_max = max_selected_history;
  }
  return max_history + n_q;
}

int hga_prepare_gpu_prefill_i8_strided(
    hga_session *s, int layer, int start_pos, int n_q, const float *q,
    int q_head_stride, int q_tok_stride, const float *k_rope, int k_head_stride,
    int k_tok_stride, const float *k_raw, int kr_head_stride, int kr_tok_stride,
    const float *v, int v_head_stride, int v_tok_stride, void *image,
    size_t image_bytes, int history_capacity, hga_stats *stats) {
  return hga_prepare_gpu_prefill_ubatch_strided(
      s, layer, start_pos, n_q, q, q_head_stride, q_tok_stride, k_rope,
      k_head_stride, k_tok_stride, k_raw, kr_head_stride, kr_tok_stride, v,
      v_head_stride, v_tok_stride, nullptr, 0, 0, 0, nullptr, 0, 0, 0, image,
      image_bytes, history_capacity, true, stats);
}

int hga_prepare_gpu_prefill_i8_q8_0_strided(
    hga_session *s, int layer, int start_pos, int n_q, const float *q,
    int q_head_stride, int q_tok_stride, const void *k_q8,
    int k_block_stride, int k_head_stride, int k_tok_stride,
    const float *k_raw, int kr_head_stride, int kr_tok_stride,
    const void *v_q8, int v_block_stride, int v_head_stride, int v_tok_stride,
    void *image, size_t image_bytes, int history_capacity, hga_stats *stats) {
  return hga_prepare_gpu_prefill_ubatch_strided(
      s, layer, start_pos, n_q, q, q_head_stride, q_tok_stride, nullptr, 0, 0,
      k_raw, kr_head_stride, kr_tok_stride, nullptr, 0, 0,
      (const uint8_t *)k_q8, k_block_stride, k_head_stride, k_tok_stride,
      (const uint8_t *)v_q8, v_block_stride, v_head_stride, v_tok_stride,
      image, image_bytes, history_capacity, true, stats);
}

int hga_prepare_gpu_prefill_f16_ubatch_strided(
    hga_session *s, int layer, int start_pos, int n_q, const float *q,
    int q_head_stride, int q_tok_stride, const float *k_rope,
    int k_head_stride, int k_tok_stride, const float *k_raw,
    int kr_head_stride, int kr_tok_stride, const float *v,
    int v_head_stride, int v_tok_stride, uint16_t *image,
    size_t image_elems, int history_capacity, hga_stats *stats) {
  return hga_prepare_gpu_prefill_ubatch_strided(
      s, layer, start_pos, n_q, q, q_head_stride, q_tok_stride, k_rope,
      k_head_stride, k_tok_stride, k_raw, kr_head_stride, kr_tok_stride, v,
      v_head_stride, v_tok_stride, nullptr, 0, 0, 0, nullptr, 0, 0, 0, image,
      image_elems * sizeof(uint16_t), history_capacity, false, stats);
}

int hga_prepare_gpu_prefill_f16_strided(
    hga_session *s, int layer, int start_pos, int n_q, const float *q,
    int q_head_stride, int q_tok_stride, const float *k_rope, int k_head_stride,
    int k_tok_stride, const float *k_raw, int kr_head_stride, int kr_tok_stride,
    const float *v, int v_head_stride, int v_tok_stride, uint16_t *image,
    size_t image_elems, int total_capacity, hga_stats *stats) {
  return hga_prepare_gpu_prefill_strided_impl(
      s, layer, start_pos, n_q, q, q_head_stride, q_tok_stride, k_rope,
      k_head_stride, k_tok_stride, k_raw, kr_head_stride, kr_tok_stride, v,
      v_head_stride, v_tok_stride, image, image_elems * sizeof(uint16_t),
      total_capacity, false, stats);
}

enum PackOp { PACK_REBUILD = 0, PACK_APPEND = 1, PACK_REUSE = 2 };

static void pack_grow(PackedKV &P, int kvh, int dh, int need, bool i8) {
  int cap = P.cap;
  if (cap < 16)
    cap = 16;
  while (cap < need)
    cap *= 2;
  if (cap == P.cap)
    return;
  const size_t vec = (size_t)dh;
  if (i8) {
    std::vector<int8_t> nk8((size_t)kvh * (size_t)cap * vec);
    std::vector<int8_t> nv8((size_t)kvh * (size_t)cap * vec);
    std::vector<float> nks((size_t)kvh * (size_t)cap);
    std::vector<float> nvs((size_t)kvh * (size_t)cap);
    if (P.n > 0 && P.cap > 0) {
      for (int h = 0; h < kvh; ++h) {
        std::memcpy(nk8.data() + (size_t)h * cap * vec,
                    P.k8.data() + (size_t)h * P.cap * vec, (size_t)P.n * vec);
        std::memcpy(nv8.data() + (size_t)h * cap * vec,
                    P.v8.data() + (size_t)h * P.cap * vec, (size_t)P.n * vec);
        std::memcpy(nks.data() + (size_t)h * cap,
                    P.ksc.data() + (size_t)h * P.cap,
                    (size_t)P.n * sizeof(float));
        std::memcpy(nvs.data() + (size_t)h * cap,
                    P.vsc.data() + (size_t)h * P.cap,
                    (size_t)P.n * sizeof(float));
      }
    }
    P.k8.swap(nk8);
    P.v8.swap(nv8);
    P.ksc.swap(nks);
    P.vsc.swap(nvs);
  } else {
    std::vector<uint16_t> nk((size_t)kvh * (size_t)cap * vec);
    std::vector<uint16_t> nv((size_t)kvh * (size_t)cap * vec);
    if (P.n > 0 && P.cap > 0) {
      for (int h = 0; h < kvh; ++h) {
        std::memcpy(nk.data() + (size_t)h * cap * vec,
                    P.k16.data() + (size_t)h * P.cap * vec,
                    (size_t)P.n * vec * sizeof(uint16_t));
        std::memcpy(nv.data() + (size_t)h * cap * vec,
                    P.v16.data() + (size_t)h * P.cap * vec,
                    (size_t)P.n * vec * sizeof(uint16_t));
      }
    }
    P.k16.swap(nk);
    P.v16.swap(nv);
  }
  P.cap = cap;
}

static void pack_fill_range(Layer &L, PackedKV &P, const hga_config &c,
                            const int *keys, int i0, int i1, bool i8) {
  const int kvh = c.n_kv_heads;
  const int dh = c.head_dim;
  const int ms = c.max_seq;
  const int cap = P.cap;
  for (int i = i0; i < i1; ++i) {
    const int j = keys[i];
    for (int h = 0; h < kvh; ++h) {
      const size_t dst = ((size_t)h * (size_t)cap + (size_t)i) * (size_t)dh;
      const size_t src = kv_index(h, j, dh, ms);
      if (i8) {
        std::memcpy(P.k8.data() + dst, L.k8.data() + src, (size_t)dh);
        std::memcpy(P.v8.data() + dst, L.v8.data() + src, (size_t)dh);
        P.ksc[(size_t)h * (size_t)cap + (size_t)i] =
            L.k_scale[(size_t)h * (size_t)ms + (size_t)j];
        P.vsc[(size_t)h * (size_t)cap + (size_t)i] =
            L.v_scale[(size_t)h * (size_t)ms + (size_t)j];
      } else {
        std::memcpy(P.k16.data() + dst, L.k.data() + src,
                    (size_t)dh * sizeof(uint16_t));
        std::memcpy(P.v16.data() + dst, L.v.data() + src,
                    (size_t)dh * sizeof(uint16_t));
      }
    }
  }
}

static PackOp pack_sync(hga_session *s, Layer &L, const hga_config &c,
                        const int *keys, int n, int nthr) {
  (void)nthr;
  PackedKV &P = L.pack;
  const bool i8 = c.prec == HGA_PREC_I8;
  const int kvh = c.n_kv_heads;
  const int dh = c.head_dim;
  if (n <= 0) {
    P.n = 0;
    P.keys.clear();
    return PACK_REBUILD;
  }
  const int n_old = P.n;
  int p = 0;
  const int nlim = std::min(n_old, n);
  if ((int)P.keys.size() == n_old) {
    while (p < nlim && P.keys[(size_t)p] == keys[p])
      ++p;
  }
  if (p == n && p == n_old)
    return PACK_REUSE;
  pack_grow(P, kvh, dh, n, i8);
  if (p < n) {
    hga_pack_parallel_for(s, p, n, 64, [&](int i) {
      pack_fill_range(L, P, c, keys, i, i + 1, i8);
    });
  }
  P.keys.assign(keys, keys + n);
  P.n = n;
  if (p == 0)
    return PACK_REBUILD;
  return PACK_APPEND;
}

int hga_prepare_gpu_verify_i8_strided(
    hga_session *s, int layer, int start_pos, int n_q, int graph_n_q,
    const float *q, int q_head_stride, int q_tok_stride,
    const float *k_rope, int k_head_stride, int k_tok_stride,
    const float *k_raw, int kr_head_stride, int kr_tok_stride,
    const float *v, int v_head_stride, int v_tok_stride, void *image,
    size_t image_bytes, int history_capacity, hga_stats *stats) {
  if (!s || layer < 0 || layer >= s->n_layers || start_pos < 0 || n_q <= 0 ||
      n_q > graph_n_q || graph_n_q > 8 || !q || !k_rope || !v || !image ||
      history_capacity <= 0 || s->cfg.prec != HGA_PREC_I8 ||
      s->cfg.router != HGA_ROUTER_HIER)
    return -1;

  hga_gpu_verify_i8_layout layout{};
  if (!hga_gpu_verify_i8_image_layout(s, history_capacity, graph_n_q,
                                      &layout) ||
      image_bytes < layout.n_bytes)
    return -1;

  Layer &L = s->layers[(size_t)layer];
  const int H = s->cfg.n_q_heads;
  const int kvh = s->cfg.n_kv_heads;
  const int dh = s->cfg.head_dim;
  const int nthr = std::max(1, s->cfg.n_pack_threads);

  hga_append_f32_strided(s, layer, start_pos, n_q, k_rope, k_head_stride,
                         k_tok_stride, k_raw, kr_head_stride, kr_tok_stride, v,
                         v_head_stride, v_tok_stride);

  /* Match the CPU VERIFY router exactly: one mean-pooled query per Q head,
   * one shared route for the whole adjacent speculative batch. */
  grow(s->scratch_q_pool, (size_t)H * (size_t)dh);
  float *pooled = s->scratch_q_pool.data();
  const size_t tok_span = (size_t)n_q * (size_t)q_tok_stride;
  const float *qh = q;
  float *dst = pooled;
  const float *const qh_end = q + (size_t)H * (size_t)q_head_stride;
  if (n_q == 1) {
    for (; qh < qh_end; qh += q_head_stride, dst += dh)
      std::memcpy(dst, qh, (size_t)dh * sizeof(float));
  } else {
    for (; qh < qh_end; qh += q_head_stride, dst += dh) {
      std::memset(dst, 0, (size_t)dh * sizeof(float));
      const float *qt = qh;
      const float *const qt_end = qt + tok_span;
      for (; qt < qt_end; qt += q_tok_stride)
        hga_axpy_f32(dst, qt, 1.f, dh);
      hga_scale_f32(dst, 1.f / (float)n_q, dh);
    }
  }

  const double tr0 = now_ms();
  const int q_last = start_pos + n_q - 1;
  const int n_closed_view =
      std::min(L.n_closed, q_last / std::max(1, s->cfg.chunk_size));
  RouteSet rs;
  int n_sel = 0;
  int n_open = 0;
  route_layer(s, L, pooled, n_closed_view, n_q, rs, &n_sel, &n_open);
  std::vector<Span> spans;
  collect_spans(s, L, rs, n_closed_view, spans);
  std::vector<int> all_keys;
  collect_keys(s, L, spans, q_last, all_keys);
  std::vector<int> history;
  history.reserve(all_keys.size());
  for (int key : all_keys)
    if (key < start_pos)
      history.push_back(key);
  const double tr1 = now_ms();

  const int n_history = (int)history.size();
  if (n_history > history_capacity)
    return -1;

  PackOp pack_op = PACK_REBUILD;
  const double tp0 = now_ms();
  if (n_history > 0) {
    pack_op = pack_sync(s, L, s->cfg, history.data(), n_history, nthr);
    if (pack_op == PACK_REBUILD)
      ++s->metrics.packed_rebuilds;
    else if (pack_op == PACK_APPEND)
      ++s->metrics.packed_appends;
    else
      ++s->metrics.packed_reuses;
  }

  int8_t *const kout =
      (int8_t *)((uint8_t *)image + layout.k_offset);
  int8_t *const vout =
      (int8_t *)((uint8_t *)image + layout.v_offset);
  float *const kscale =
      (float *)((uint8_t *)image + layout.k_scale_offset);
  float *const vscale =
      (float *)((uint8_t *)image + layout.v_scale_offset);
  const size_t scale_elems = (size_t)kvh * (size_t)history_capacity;
  std::fill(kscale, kscale + scale_elems, 0.f);
  std::fill(vscale, vscale + scale_elems, 0.f);

  if (n_history > 0) {
    const size_t src_vec = (size_t)L.pack.cap * (size_t)dh;
    const size_t dst_vec = (size_t)history_capacity * (size_t)dh;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(std::min(nthr, kvh)) schedule(static)
#endif
    for (int kh = 0; kh < kvh; ++kh) {
      std::memcpy(kout + (size_t)kh * dst_vec,
                  L.pack.k8.data() + (size_t)kh * src_vec,
                  (size_t)n_history * (size_t)dh);
      std::memcpy(vout + (size_t)kh * dst_vec,
                  L.pack.v8.data() + (size_t)kh * src_vec,
                  (size_t)n_history * (size_t)dh);
      std::memcpy(kscale + (size_t)kh * (size_t)history_capacity,
                  L.pack.ksc.data() + (size_t)kh * (size_t)L.pack.cap,
                  (size_t)n_history * sizeof(float));
      std::memcpy(vscale + (size_t)kh * (size_t)history_capacity,
                  L.pack.vsc.data() + (size_t)kh * (size_t)L.pack.cap,
                  (size_t)n_history * sizeof(float));
    }
  }

  /* The mask travels with the packed image. This avoids rebuilding a CUDA
   * mask subgraph as context grows and masks unused fixed-capacity slots
   * before softmax (zero K/V alone would still alter its denominator). */
  uint16_t *const mask =
      (uint16_t *)((uint8_t *)image + layout.mask_offset);
  const int n_keys_graph = history_capacity + graph_n_q;
  const uint16_t visible = f32_to_f16(0.f);
  const uint16_t hidden = f32_to_f16(-10000.f);
  for (int t = 0; t < graph_n_q; ++t) {
    uint16_t *row = mask + (size_t)t * (size_t)n_keys_graph;
    for (int x = 0; x < history_capacity; ++x)
      row[x] = x < n_history ? visible : hidden;
    for (int x = 0; x < graph_n_q; ++x)
      row[history_capacity + x] = x <= t ? visible : hidden;
  }
  const double tp1 = now_ms();

  s->last_keys = history;
  s->last_keys_layer = layer;
  hga_close_full_chunks(s, layer);

  if (stats) {
    std::memset(stats, 0, sizeof(*stats));
    stats->n_kv = L.n_kv;
    stats->n_closed_chunks = L.n_closed;
    stats->n_selected_chunks = n_sel;
    stats->n_opened_groups = n_open;
    stats->n_attended_tokens = n_history + n_q;
    stats->sparsity = L.n_kv > 0
        ? (float)(n_history + n_q) / (float)L.n_kv : 1.f;
    stats->ms_route = tr1 - tr0;
    stats->ms_pack = tp1 - tp0;
    stats->pack_rebuild = pack_op == PACK_REBUILD;
    stats->pack_append = pack_op == PACK_APPEND;
    stats->pack_reuse = pack_op == PACK_REUSE;
  }
  return n_history;
}

static int hga_gcd(int a, int b) {
  a = std::abs(a);
  b = std::abs(b);
  while (b) {
    const int t = a % b;
    a = b;
    b = t;
  }
  return a == 0 ? 1 : a;
}

static int hga_lcm(int a, int b) {
  if (a <= 0 || b <= 0)
    return std::max(a, b);
  return a / hga_gcd(a, b) * b;
}

static void load_kv_f32(const Layer &L, const hga_config &c, int h, int tok,
                        float *kout, float *vout) {
  const int dh = c.head_dim;
  const int ms = c.max_seq;
  if (c.prec == HGA_PREC_I8) {
    const int8_t *k8 = L.k8.data() + kv_index(h, tok, dh, ms);
    const int8_t *v8 = L.v8.data() + kv_index(h, tok, dh, ms);
    const float ks = L.k_scale[(size_t)h * (size_t)ms + (size_t)tok];
    const float vs = L.v_scale[(size_t)h * (size_t)ms + (size_t)tok];
    const int8_t *k8_end = k8 + dh;
    float *ko = kout;
    if (vout) {
      float *vo = vout;
      for (; k8 < k8_end; ++k8, ++v8, ++ko, ++vo) {
        *ko = (float)*k8 * ks;
        *vo = (float)*v8 * vs;
      }
    } else {
      for (; k8 < k8_end; ++k8, ++ko)
        *ko = (float)*k8 * ks;
    }
  } else {
    const uint16_t *k = L.k.data() + kv_index(h, tok, dh, ms);
    const uint16_t *v = L.v.data() + kv_index(h, tok, dh, ms);
    const uint16_t *k_end = k + dh;
    float *ko = kout;
    if (vout) {
      float *vo = vout;
      for (; k < k_end; ++k, ++v, ++ko, ++vo) {
        *ko = f16_to_f32(*k);
        *vo = f16_to_f32(*v);
      }
    } else {
      for (; k < k_end; ++k, ++ko)
        *ko = f16_to_f32(*k);
    }
  }
}

/* RetroInfer segmented k-means on mid-context keys. New tokens after mid_hi
 * stay exact (steady/local) until the next rebuild. */
static bool wave_rebuild(hga_session *s, Layer &L) {
  const hga_config &c = s->cfg;
  const int C = c.chunk_size;
  const int n_closed = L.n_closed;
  const int f_hi = std::min(c.keep_first, n_closed);
  const int l_lo = std::max(f_hi, n_closed - c.keep_last);
  const int mid_lo = f_hi * C;
  const int mid_hi = l_lo * C;
  const int n_mid = std::max(0, mid_hi - mid_lo);
  const int kvh = c.n_kv_heads;
  const int dh = c.head_dim;
  const int cluster = c.wave_cluster > 0 ? c.wave_cluster : 16;
  const int seg_tok = c.wave_seg > 0 ? c.wave_seg : 8192;
  const int iters = c.wave_iters > 0 ? c.wave_iters : 3;
  const int nthr = std::max(1, c.n_threads);

  if (n_mid < cluster * 2) {
    L.wave = WaveIndex{};
    return false;
  }

  int n_seg = std::max(1, n_mid / seg_tok);
  const int factor = hga_lcm(8, n_seg);
  int n_cent = std::max(n_mid / cluster, factor);
  n_cent = (n_cent / factor) * factor;
  if (n_cent < factor)
    n_cent = factor;
  if (n_cent > n_mid)
    n_cent = (n_mid / factor) * factor;
  if (n_cent < n_seg || n_cent <= 0) {
    L.wave = WaveIndex{};
    return false;
  }

  const int n_tok_seg = n_mid / n_seg;
  const int n_cent_seg = n_cent / n_seg;
  if (n_tok_seg <= 0 || n_cent_seg <= 0) {
    L.wave = WaveIndex{};
    return false;
  }

  std::vector<float> data((size_t)kvh * (size_t)n_mid * (size_t)dh);
  std::vector<float> vals((size_t)kvh * (size_t)n_mid * (size_t)dh);
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) num_threads(nthr) schedule(static)
#endif
  for (int h = 0; h < kvh; ++h) {
    for (int i = 0; i < n_mid; ++i) {
      load_kv_f32(
          L, c, h, mid_lo + i,
          data.data() + ((size_t)h * (size_t)n_mid + (size_t)i) * (size_t)dh,
          vals.data() + ((size_t)h * (size_t)n_mid + (size_t)i) * (size_t)dh);
    }
  }

  std::vector<float> cent((size_t)kvh * (size_t)n_cent * (size_t)dh, 0.f);
  for (int h = 0; h < kvh; ++h) {
    for (int ci = 0; ci < n_cent; ++ci) {
      int idx = (int)(((double)ci + 0.5) * (double)n_mid / (double)n_cent);
      if (idx < 0)
        idx = 0;
      if (idx >= n_mid)
        idx = n_mid - 1;
      std::memcpy(
          cent.data() + ((size_t)h * (size_t)n_cent + (size_t)ci) * (size_t)dh,
          data.data() + ((size_t)h * (size_t)n_mid + (size_t)idx) * (size_t)dh,
          (size_t)dh * sizeof(float));
    }
  }

  std::vector<int> assign((size_t)kvh * (size_t)n_mid, 0);
  std::vector<int> cnt((size_t)kvh * (size_t)n_cent, 0);

  auto l2_normalize = [&](float *x) {
    float n2 = 0.f;
    for (int d = 0; d < dh; ++d)
      n2 += x[d] * x[d];
    if (n2 < 1e-12f)
      return;
    const float inv = 1.f / std::sqrt(n2);
    for (int d = 0; d < dh; ++d)
      x[d] *= inv;
  };

  /* Keep assignment inside each segment so a centroid cannot swallow the
   * whole sequence (the paper's last full-data pass does that on GPU with
   * structured keys; on random CPU keys it produced 2k–4k-token clusters). */
  for (int it = 0; it < std::max(1, iters); ++it) {
    const bool last = (it == iters - 1);
    std::vector<float> sums((size_t)kvh * (size_t)n_cent * (size_t)dh, 0.f);
    std::fill(cnt.begin(), cnt.end(), 0);

#if defined(_OPENMP)
#pragma omp parallel for collapse(2) num_threads(nthr) schedule(static)
#endif
    for (int h = 0; h < kvh; ++h) {
      for (int seg = 0; seg < n_seg; ++seg) {
        std::vector<float> lsum((size_t)n_cent_seg * (size_t)dh, 0.f);
        std::vector<int> lcnt((size_t)n_cent_seg, 0);
        const int t0 = seg * n_tok_seg;
        const int t1 = (seg + 1 == n_seg) ? n_mid : (seg + 1) * n_tok_seg;
        const float *c0 = cent.data() + ((size_t)h * (size_t)n_cent +
                                         (size_t)(seg * n_cent_seg)) *
                                            (size_t)dh;
        for (int t = t0; t < t1; ++t) {
          const float *x = data.data() +
                           ((size_t)h * (size_t)n_mid + (size_t)t) * (size_t)dh;
          float best = -1e30f;
          int bi = 0;
          for (int ci = 0; ci < n_cent_seg; ++ci) {
            const float sdot = hga_dot_f32(x, c0 + (size_t)ci * (size_t)dh, dh);
            if (sdot > best) {
              best = sdot;
              bi = ci;
            }
          }
          assign[(size_t)h * (size_t)n_mid + (size_t)t] = seg * n_cent_seg + bi;
          lcnt[(size_t)bi]++;
          hga_axpy_f32(lsum.data() + (size_t)bi * (size_t)dh, x, 1.f, dh);
        }
        std::memcpy(sums.data() + ((size_t)h * (size_t)n_cent +
                                   (size_t)(seg * n_cent_seg)) *
                                      (size_t)dh,
                    lsum.data(),
                    (size_t)n_cent_seg * (size_t)dh * sizeof(float));
        std::memcpy(cnt.data() + (size_t)h * (size_t)n_cent +
                        (size_t)(seg * n_cent_seg),
                    lcnt.data(), (size_t)n_cent_seg * sizeof(int));
      }
    }

    for (int h = 0; h < kvh; ++h) {
      for (int ci = 0; ci < n_cent; ++ci) {
        const int n = cnt[(size_t)h * (size_t)n_cent + (size_t)ci];
        float *cc = cent.data() +
                    ((size_t)h * (size_t)n_cent + (size_t)ci) * (size_t)dh;
        if (n <= 0)
          continue;
        const float inv = 1.f / (float)n;
        const float *sp =
            sums.data() +
            ((size_t)h * (size_t)n_cent + (size_t)ci) * (size_t)dh;
        for (int d = 0; d < dh; ++d)
          cc[d] = sp[d] * inv;
        if (!last)
          l2_normalize(cc);
      }
    }
  }

  int max_cs = 1;
  for (int i = 0; i < kvh * n_cent; ++i)
    max_cs = std::max(max_cs, cnt[(size_t)i]);

  std::vector<float> vsum((size_t)kvh * (size_t)n_cent * (size_t)dh, 0.f);
  std::vector<int> members((size_t)kvh * (size_t)n_cent * (size_t)max_cs, -1);
  std::vector<int> fill((size_t)kvh * (size_t)n_cent, 0);
  for (int h = 0; h < kvh; ++h) {
    for (int t = 0; t < n_mid; ++t) {
      const int ci = assign[(size_t)h * (size_t)n_mid + (size_t)t];
      const int slot = fill[(size_t)h * (size_t)n_cent + (size_t)ci]++;
      members[(((size_t)h * (size_t)n_cent + (size_t)ci) * (size_t)max_cs) +
              (size_t)slot] = mid_lo + t;
      hga_axpy_f32(
          vsum.data() + ((size_t)h * (size_t)n_cent + (size_t)ci) * (size_t)dh,
          vals.data() + ((size_t)h * (size_t)n_mid + (size_t)t) * (size_t)dh,
          1.f, dh);
    }
  }

  L.wave.mid_lo = mid_lo;
  L.wave.mid_hi = mid_hi;
  L.wave.n_mid = n_mid;
  L.wave.n_cent = n_cent;
  L.wave.built_n_closed = n_closed;
  L.wave.max_cs = max_cs;
  L.wave.cent.swap(cent);
  L.wave.vsum.swap(vsum);
  L.wave.members.swap(members);
  L.wave.csize.swap(cnt);
  std::fprintf(stderr,
               "hga wave: rebuilt n_mid=%d n_cent=%d max_cs=%d closed=%d "
               "(cluster=%d segs=%d iters=%d)\n",
               n_mid, n_cent, max_cs, n_closed, cluster, n_seg, iters);
  return true;
}

static bool wave_index_ready(hga_session *s, Layer &L) {
  const int C = s->cfg.chunk_size;
  const int upd = s->cfg.wave_update > 0 ? s->cfg.wave_update : 1024;
  const int upd_chunks = std::max(1, upd / std::max(1, C));
  if (L.wave.n_cent <= 0 || L.wave.built_n_closed <= 0)
    return wave_rebuild(s, L);
  if (L.n_closed - L.wave.built_n_closed >= upd_chunks)
    return wave_rebuild(s, L);
  return true;
}

static void route_wave(const hga_session *s, const Layer &L,
                       const float *q_pool, RouteSet &rs, int *n_sel,
                       int *n_open) {
  const int kvh = s->cfg.n_kv_heads;
  const int H = s->cfg.n_q_heads;
  const int dh = s->cfg.head_dim;
  const int rep = H / kvh;
  const int n_cent = L.wave.n_cent;
  /* Paper: retrieval_budget is a *token* fraction. nprobe is only a
   * starting cluster count; collect_keys_wave stops at the token cap. */
  int nprobe =
      std::max(1, (int)std::lround((double)n_cent * (double)s->cfg.frac_retr));
  nprobe = std::min(std::max(nprobe, 8),
                    n_cent); /* extra clusters if some are tiny */
  int n_est = (int)std::lround((double)n_cent * (double)s->cfg.frac_est);
  n_est = std::min(std::max(n_est, 0), n_cent - nprobe);
  *n_sel = nprobe;
  *n_open = n_est;
  rs.retr.assign((size_t)kvh, std::vector<int>());
  rs.est.assign((size_t)kvh, std::vector<int>((size_t)n_est, 0));
  if (n_cent <= 0)
    return;

  std::vector<float> dist((size_t)kvh * (size_t)n_cent, 0.f);
  for (int h = 0; h < H; ++h) {
    const int kh = h / rep;
    const float *qh = q_pool + (size_t)h * (size_t)dh;
    const float *c0 =
        L.wave.cent.data() + (size_t)kh * (size_t)n_cent * (size_t)dh;
    std::vector<float> logits((size_t)n_cent);
    float m = -1e30f;
    for (int ci = 0; ci < n_cent; ++ci) {
      if (L.wave.csize[(size_t)kh * (size_t)n_cent + (size_t)ci] <= 0) {
        logits[(size_t)ci] = -1e30f;
        continue;
      }
      logits[(size_t)ci] =
          hga_dot_f32(qh, c0 + (size_t)ci * (size_t)dh, dh) * s->scale;
      m = std::max(m, logits[(size_t)ci]);
    }
    float z = 0.f;
    for (int ci = 0; ci < n_cent; ++ci) {
      const float e = (logits[(size_t)ci] < -1e20f)
                          ? 0.f
                          : std::exp(logits[(size_t)ci] - m);
      logits[(size_t)ci] = e;
      z += e;
    }
    const float inv = z > 0.f ? 1.f / z : 0.f;
    for (int ci = 0; ci < n_cent; ++ci)
      dist[(size_t)kh * (size_t)n_cent + (size_t)ci] +=
          logits[(size_t)ci] * inv;
  }

  const int take = nprobe + n_est;
  for (int kh = 0; kh < kvh; ++kh) {
    std::vector<int> idx((size_t)n_cent);
    for (int i = 0; i < n_cent; ++i)
      idx[(size_t)i] = i;
    std::partial_sort(idx.begin(), idx.begin() + take, idx.end(),
                      [&](int a, int b) {
                        return dist[(size_t)kh * (size_t)n_cent + (size_t)a] >
                               dist[(size_t)kh * (size_t)n_cent + (size_t)b];
                      });
    rs.retr[(size_t)kh].assign(idx.begin(), idx.begin() + nprobe);
    for (int i = 0; i < n_est; ++i)
      rs.est[(size_t)kh][(size_t)i] = idx[(size_t)(nprobe + i)];
  }
}

static void collect_keys_wave(const hga_session *s, const Layer &L,
                              const RouteSet &rs, int q_hi,
                              std::vector<std::vector<int>> &keys_h) {
  const int kvh = s->cfg.n_kv_heads;
  const int n_cent = L.wave.n_cent;
  const int max_cs = L.wave.max_cs;
  const int mid_lo = L.wave.mid_lo;
  const int mid_hi = L.wave.mid_hi;
  keys_h.assign((size_t)kvh, {});
  for (int kh = 0; kh < kvh; ++kh) {
    auto &keys = keys_h[(size_t)kh];
    keys.reserve((size_t)mid_lo + (size_t)(L.n_kv - mid_hi) + 64 +
                 (rs.retr.empty() ? 0 : rs.retr[(size_t)kh].size() * 16));
    const int sink_hi = std::min(mid_lo - 1, q_hi);
    for (int j = 0; j <= sink_hi; ++j)
      keys.push_back(j);
    if (!rs.retr.empty()) {
      const int budget =
          std::max(1, (int)std::lround((double)std::max(1, L.wave.n_mid) *
                                       (double)s->cfg.frac_retr));
      int got = 0;
      for (int ci : rs.retr[(size_t)kh]) {
        if (got >= budget)
          break;
        if (ci < 0 || ci >= n_cent)
          continue;
        const int cs = L.wave.csize[(size_t)kh * (size_t)n_cent + (size_t)ci];
        const int *mem =
            L.wave.members.data() +
            ((size_t)kh * (size_t)n_cent + (size_t)ci) * (size_t)max_cs;
        for (int i = 0; i < cs && got < budget; ++i) {
          const int j = mem[i];
          if (j >= 0 && j <= q_hi) {
            keys.push_back(j);
            ++got;
          }
        }
      }
    }
    const int tail0 = mid_hi;
    for (int j = tail0; j < L.n_kv && j <= q_hi; ++j)
      keys.push_back(j);
  }
}

static void flash_est(const hga_session *s, const Layer &L, int kh,
                      int start_pos, int n_q, int h0, int h1, int t0, int t1,
                      const int *est_ids, int n_est, const float *q, int q_hs,
                      int q_ts, float *m, float *lse, float *acc) {
  const int dh = s->cfg.head_dim;
  const int n_cent = L.wave.n_cent;
  const float ninf = -std::numeric_limits<float>::infinity();
  for (int ei = 0; ei < n_est; ++ei) {
    const int ci = est_ids[ei];
    if (ci < 0 || ci >= n_cent)
      continue;
    if (L.wave.csize[(size_t)kh * (size_t)n_cent + (size_t)ci] <= 0)
      continue;
    const float *k = L.wave.cent.data() +
                     ((size_t)kh * (size_t)n_cent + (size_t)ci) * (size_t)dh;
    const float *v = L.wave.vsum.data() +
                     ((size_t)kh * (size_t)n_cent + (size_t)ci) * (size_t)dh;
    float *mp = m;
    float *lp = lse;
    float *o = acc;
    for (int h = h0; h < h1; ++h) {
      const float *qh =
          q + (size_t)h * (size_t)q_hs + (size_t)t0 * (size_t)q_ts;
      const float *qh_end = qh + (size_t)(t1 - t0) * (size_t)q_ts;
      int pos = start_pos + t0;
      for (; qh < qh_end; qh += q_ts, ++mp, ++lp, o += dh, ++pos) {
        if (pos < 0)
          continue;
        const float score = hga_dot_f32(qh, k, dh) * s->scale;
        const float m2 = std::max(*mp, score);
        const float e1 = (*mp == ninf) ? 0.f : std::exp(*mp - m2);
        const float e2 = std::exp(score - m2);
        hga_scale_f32(o, e1, dh);
        hga_axpy_f32(o, v, e2, dh);
        *lp = (*lp) * e1 + e2;
        *mp = m2;
      }
    }
  }
}

/* 2D flash tiles so Q-tile + K-tile + acc fit in one core's L2 (~1 MB on
 * 6148). The picker prefers 2 Q × 4 K tiles when that fits, but a production
 * 512-token prefill resolves to 8 Q × 1 K per KV head. The experimental
 * key-only mode below forces 1 Q × 8 K to measure shared-KV reuse against the
 * larger partial-accumulator footprint. */
static void pick_qk_tiles(int n_kvh, int n_q_rows, int n_keys, int dh,
                          int q_bpe, int kv_bpe, int n_threads, int *nq_tiles,
                          int *nk_tiles) {
  const size_t l2 = 768u * 1024u;
  n_kvh = std::max(1, n_kvh);
  n_q_rows = std::max(1, n_q_rows);
  n_keys = std::max(1, n_keys);
  n_threads = std::max(1, n_threads);

  const bool prefill_smt = hga_prefill_smt_enabled();
  const bool force_prefill_tiles = hga_prefill_k_tiles_enabled();
  const int forced_q_tiles = 2;
  const int forced_k_tiles = prefill_smt ? 10 : 5;
  if ((force_prefill_tiles || prefill_smt) && n_q_rows >= 64 &&
      n_keys >= forced_k_tiles &&
      n_kvh * forced_q_tiles * forced_k_tiles <=
          (prefill_smt ? 2 * n_threads : n_threads)) {
    *nq_tiles = forced_q_tiles;
    *nk_tiles = forced_k_tiles;
    static bool logged = false;
    if (!logged) {
      logged = true;
      std::fprintf(stderr,
                   "hga prefill K tiles: kv_heads=%d  q_tiles=%d  k_tiles=%d  "
                   "tasks=%d  physical_threads=%d  smt=%d\n",
                   n_kvh, *nq_tiles, *nk_tiles, n_kvh * *nq_tiles * *nk_tiles,
                   n_threads, (int)prefill_smt);
    }
    return;
  }

  auto bytes = [&](int nqt, int nkt) -> size_t {
    const int nr = (n_q_rows + nqt - 1) / nqt;
    const int nk = (n_keys + nkt - 1) / nkt;
    return (size_t)nr * (size_t)dh * (size_t)(q_bpe + 4) +
           (size_t)nk * (size_t)dh * 2u * (size_t)kv_bpe;
  };
  auto ok = [&](int nqt, int nkt) {
    return nqt >= 1 && nkt >= 1 && nqt <= n_q_rows && nkt <= n_keys &&
           (size_t)n_kvh * (size_t)nqt * (size_t)nkt <= (size_t)n_threads &&
           bytes(nqt, nkt) <= l2;
  };

  /* Preferred: 2 Q-tiles × 4 K-tiles per KV head (32 cores when n_kvh=4). */
  if (ok(2, 4)) {
    *nq_tiles = 2;
    *nk_tiles = 4;
  } else {
    int nqt = 1, nkt = 1;
    while (!ok(nqt, nkt) || bytes(nqt, nkt) > l2) {
      const int nr = (n_q_rows + nqt - 1) / nqt;
      const int nk = (n_keys + nkt - 1) / nkt;
      const size_t qacc = (size_t)nr * dh * (size_t)(q_bpe + 4);
      const size_t kv = (size_t)nk * dh * 2u * (size_t)kv_bpe;
      if (kv >= qacc && nkt < n_keys)
        ++nkt;
      else if (nqt < n_q_rows)
        ++nqt;
      else if (nkt < n_keys)
        ++nkt;
      else
        break;
      if (n_kvh * nqt * nkt > n_threads) {
        if (nkt > 1)
          --nkt;
        else if (nqt > 1)
          --nqt;
        break;
      }
    }
    *nq_tiles = nqt;
    *nk_tiles = nkt;
  }

  /* Once K/V alone is too large for the nominal L2 target, the balancing
   * fallback above can reduce the Q-tile count as n_keys grows.  That is the
   * wrong direction: it leaves cores idle and makes each Q/accumulator tile
   * larger without making the single K tile fit.  Keep at least eight Q tiles
   * for a genuinely large prefill when that exposes one task per physical
   * core group (4 KV heads x 8 = 32 tasks on Qwen3.8). */
  const int q_floor = std::min(8, n_threads / n_kvh);
  if (*nk_tiles == 1 && q_floor > 1 && n_q_rows >= 64 * q_floor &&
      *nq_tiles < q_floor) {
    *nq_tiles = q_floor;
  }

  if (n_q_rows >= 64 && n_keys >= 64) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      std::fprintf(stderr,
                   "hga L2 tiles: kv_heads=%d  q_tiles=%d  k_tiles=%d  "
                   "tasks=%d  threads=%d  (want ~32 on 40-core)\n",
                   n_kvh, *nq_tiles, *nk_tiles,
                   n_kvh * (*nq_tiles) * (*nk_tiles), n_threads);
    }
  }

  /* Use leftover cores: split the heavier side until we fill n_threads. */
  for (;;) {
    const int nqt = *nq_tiles, nkt = *nk_tiles;
    const int nr = (n_q_rows + nqt - 1) / nqt;
    const int nk = (n_keys + nkt - 1) / nkt;
    const size_t qacc = (size_t)nr * dh * (size_t)(q_bpe + 4);
    const size_t kv = (size_t)nk * dh * 2u * (size_t)kv_bpe;
    if (n_kvh * (nqt + 1) * nkt <= n_threads && nqt < n_q_rows && qacc >= kv &&
        ok(nqt + 1, nkt))
      *nq_tiles = nqt + 1;
    else if (n_kvh * nqt * (nkt + 1) <= n_threads && nkt < n_keys &&
             ok(nqt, nkt + 1))
      *nk_tiles = nkt + 1;
    else if (n_kvh * (nqt + 1) * nkt <= n_threads && nqt < n_q_rows &&
             ok(nqt + 1, nkt))
      *nq_tiles = nqt + 1;
    else
      break;
  }
}

static void merge_online_softmax(float *m, float *lse, float *acc,
                                 const float *m2, const float *lse2,
                                 const float *acc2, int n_rows, int dh) {
  const float ninf = -std::numeric_limits<float>::infinity();
  float *mp = m;
  const float *mp_end = m + n_rows;
  float *lp = lse;
  const float *lp2 = lse2;
  const float *m2p = m2;
  float *o = acc;
  const float *o2 = acc2;
  for (; mp < mp_end; ++mp, ++lp, ++m2p, ++lp2, o += dh, o2 += dh) {
    const float a = *mp, b = *m2p;
    if (b == ninf)
      continue;
    if (a == ninf) {
      *mp = b;
      *lp = *lp2;
      std::memcpy(o, o2, (size_t)dh * sizeof(float));
      continue;
    }
    const float mx = std::max(a, b);
    const float ea = expf(a - mx);
    const float eb = expf(b - mx);
    hga_scale_f32(o, ea, dh);
    hga_axpy_f32(o, o2, eb, dh);
    *lp = (*lp) * ea + (*lp2) * eb;
    *mp = mx;
  }
}

static void flash_tile_f16(const hga_session *s, const Layer &L, int kh,
                           int start_pos, int n_q, int h0, int h1, int t0,
                           int t1, const int *keys, int k0, int k1,
                           const float *qf, float *m, float *lse, float *acc,
                           float *scores, const uint16_t *pk,
                           const uint16_t *pv, int pack_cap) {
  const int dh = s->cfg.head_dim;
  const int ms = s->cfg.max_seq;
  const bool packed = pack_cap > 0 && pk && pv;
  const uint16_t *khp = packed
                            ? pk + (size_t)kh * (size_t)pack_cap * (size_t)dh
                            : L.k.data() + (size_t)kh * (size_t)ms * (size_t)dh;
  const uint16_t *vhp = packed
                            ? pv + (size_t)kh * (size_t)pack_cap * (size_t)dh
                            : L.v.data() + (size_t)kh * (size_t)ms * (size_t)dh;
  const int n_rows = (h1 - h0) * (t1 - t0);
  const int nk = k1 - k0;
  const float ninf = -std::numeric_limits<float>::infinity();
  const float attn_scale = s->scale;
  {
    float *mp = m, *lp = lse;
    for (float *me = m + n_rows; mp < me; ++mp, ++lp) {
      *mp = ninf;
      *lp = 0.f;
    }
  }
  std::memset(acc, 0, (size_t)n_rows * (size_t)dh * sizeof(float));
  if (!keys || nk <= 0 || n_rows <= 0)
    return;

  /* Scores are [n_rows][nk] so softmax is sequential (prefill n_rows~192). */
  const uint16_t *kwalk = packed ? khp + (size_t)k0 * (size_t)dh : nullptr;
  const int *kp = keys + k0;
  const int *kp_end = kp + nk;
  const size_t q_hstride = (size_t)n_q * (size_t)dh;
  const float *qh0 = qf + ((size_t)h0 * (size_t)n_q + (size_t)t0) * (size_t)dh;
  const float *qh_end = qh0 + (size_t)(h1 - h0) * q_hstride;
  const size_t tok_span = (size_t)(t1 - t0) * (size_t)dh;
  float *col = scores;
  for (; kp < kp_end; ++kp, ++col) {
    const int j = *kp;
    const uint16_t *k = packed ? kwalk : khp + (size_t)j * (size_t)dh;
    if (kp + 1 < kp_end)
      prefetch_bytes(packed ? kwalk + dh : khp + (size_t)kp[1] * (size_t)dh,
                     dh * 2);
    float *sc = col;
    const float *qh = qh0;
    for (; qh < qh_end; qh += q_hstride) {
      const float *qt = qh;
      const float *qt_end = qt + tok_span;
      int pos = start_pos + t0;
      for (; qt < qt_end; qt += dh, ++pos, sc += nk)
        *sc = (j > pos) ? ninf : hga_dot_f16k(qt, k, dh) * attn_scale;
    }
    if (packed)
      kwalk += dh;
  }

  float *srow = scores;
  const float *srow_end = scores + (size_t)n_rows * (size_t)nk;
  float *mp = m;
  float *lp = lse;
  for (; srow < srow_end; srow += nk, ++mp, ++lp) {
    float mx = ninf;
    const float *s = srow;
    const float *s_end = srow + nk;
    for (; s < s_end; ++s)
      if (*s > mx)
        mx = *s;
    *mp = mx;
    if (!(mx > -1e20f))
      continue;
    float z = 0.f;
    float *sp = srow;
    for (; sp < s_end; ++sp) {
      const float e = (*sp == ninf) ? 0.f : expf(*sp - mx);
      *sp = e;
      z += e;
    }
    *lp = z;
  }

  alignas(64) float vf[HGA_MAX_DH];
  const uint16_t *vwalk = packed ? vhp + (size_t)k0 * (size_t)dh : nullptr;
  const int *kpv = keys + k0;
  const int *kpv_end = kpv + nk;
  const float *wr0 = scores;
  for (; kpv < kpv_end; ++kpv, ++wr0) {
    const uint16_t *v = packed ? vwalk : vhp + (size_t)*kpv * (size_t)dh;
    if (kpv + 1 < kpv_end)
      prefetch_bytes(packed ? vwalk + dh : vhp + (size_t)kpv[1] * (size_t)dh,
                     dh * 2);
    const float *wr = wr0;
    const float *wr_end = wr0 + (size_t)n_rows * (size_t)nk;
    float *ar = acc;
    if (dh <= HGA_MAX_DH) {
      int d = 0;
#if HGA_AVX512
      for (; d + 16 <= dh; d += 16) {
        _mm512_storeu_ps(vf + d, _mm512_cvtph_ps(_mm256_loadu_si256(
                                     (const __m256i *)(v + d))));
      }
#endif
      for (; d < dh; ++d)
        vf[d] = f16_to_f32(v[d]);
      for (; wr < wr_end; wr += nk, ar += dh) {
        if (*wr != 0.f)
          hga_axpy_f32(ar, vf, *wr, dh);
      }
    } else {
      for (; wr < wr_end; wr += nk, ar += dh) {
        if (*wr != 0.f)
          hga_axpy_f16(ar, v, *wr, dh);
      }
    }
    if (packed)
      vwalk += dh;
  }
}

static void flash_tile_i8(const hga_session *s, const Layer &L, int kh,
                          int start_pos, int n_q, int h0, int h1, int t0,
                          int t1, const int *keys, int k0, int k1,
                          const int8_t *q8, const float *qsc, float *m,
                          float *lse, float *acc, float *scores,
                          const int8_t *pk, const int8_t *pv, const float *pks,
                          const float *pvs, int pack_cap) {
  const int dh = s->cfg.head_dim;
  const int ms = s->cfg.max_seq;
  const bool packed = pack_cap > 0 && pk && pv;
  const int8_t *khp = packed
                          ? pk + (size_t)kh * (size_t)pack_cap * (size_t)dh
                          : L.k8.data() + (size_t)kh * (size_t)ms * (size_t)dh;
  const int8_t *vhp = packed
                          ? pv + (size_t)kh * (size_t)pack_cap * (size_t)dh
                          : L.v8.data() + (size_t)kh * (size_t)ms * (size_t)dh;
  const float *ksc = packed ? pks + (size_t)kh * (size_t)pack_cap
                            : L.k_scale.data() + (size_t)kh * (size_t)ms;
  const float *vsc = packed ? pvs + (size_t)kh * (size_t)pack_cap
                            : L.v_scale.data() + (size_t)kh * (size_t)ms;
  const int n_rows = (h1 - h0) * (t1 - t0);
  const int nk = k1 - k0;
  const float ninf = -std::numeric_limits<float>::infinity();
  const float attn_scale = s->scale;
  {
    float *mp = m, *lp = lse;
    for (float *me = m + n_rows; mp < me; ++mp, ++lp) {
      *mp = ninf;
      *lp = 0.f;
    }
  }
  std::memset(acc, 0, (size_t)n_rows * (size_t)dh * sizeof(float));
  if (!keys || nk <= 0 || n_rows <= 0)
    return;

  /* Per-row Q pointer + (q_scale * attn_scale), computed once. */
  const int8_t *qptr[512];
  float qfac[512];
  int qpos[512];
  const bool use_tbl = n_rows <= 512;
  if (use_tbl) {
    const int8_t *q8_h =
        q8 + ((size_t)h0 * (size_t)n_q + (size_t)t0) * (size_t)dh;
    const float *qsc_h = qsc + (size_t)h0 * (size_t)n_q + t0;
    const size_t q_hstride = (size_t)n_q * (size_t)dh;
    const int8_t **qp = qptr;
    float *qf = qfac;
    int *pp = qpos;
    for (int h = h0; h < h1; ++h) {
      const int8_t *qt = q8_h;
      const int8_t *qt_end = qt + (size_t)(t1 - t0) * (size_t)dh;
      const float *qs = qsc_h;
      int pos = start_pos + t0;
      for (; qt < qt_end; qt += dh, ++qs, ++qp, ++qf, ++pp, ++pos) {
        *qp = qt;
        *qf = (*qs) * attn_scale;
        *pp = pos;
      }
      q8_h += q_hstride;
      qsc_h += n_q;
    }
  }

  const int8_t *kwalk = packed ? khp + (size_t)k0 * (size_t)dh : nullptr;
  const float *ksp = packed ? ksc + k0 : nullptr;
  const int *kp = keys + k0;
  const int *kp_end = kp + nk;
  float *col = scores;
  for (; kp < kp_end; ++kp, ++col) {
    const int j = *kp;
    const int8_t *k = packed ? kwalk : khp + (size_t)j * (size_t)dh;
    const float ks = packed ? *ksp : ksc[j];
    if (kp + 1 < kp_end)
      prefetch_bytes(packed ? kwalk + dh : khp + (size_t)kp[1] * (size_t)dh,
                     dh);
    float *sc = col;
    if (use_tbl) {
      const int8_t **qp = qptr;
      const float *qf = qfac;
      const int *pp = qpos;
      const int *pp_end = qpos + n_rows;
      for (; pp < pp_end; ++pp, ++qp, ++qf, sc += nk) {
        *sc = (j > *pp) ? ninf : (float)hga_dot_i8(*qp, k, dh) * (*qf) * ks;
      }
    } else {
      const int8_t *q8_h =
          q8 + ((size_t)h0 * (size_t)n_q + (size_t)t0) * (size_t)dh;
      const float *qsc_h = qsc + (size_t)h0 * (size_t)n_q + t0;
      const size_t q_hstride = (size_t)n_q * (size_t)dh;
      for (int h = h0; h < h1; ++h) {
        const int8_t *qt = q8_h;
        const int8_t *qt_end = qt + (size_t)(t1 - t0) * (size_t)dh;
        const float *qs = qsc_h;
        int pos = start_pos + t0;
        for (; qt < qt_end; qt += dh, ++qs, sc += nk, ++pos) {
          *sc = (j > pos)
                    ? ninf
                    : (float)hga_dot_i8(qt, k, dh) * (*qs * attn_scale) * ks;
        }
        q8_h += q_hstride;
        qsc_h += n_q;
      }
    }
    if (packed) {
      kwalk += dh;
      ++ksp;
    }
  }

  float *srow = scores;
  const float *srow_end = scores + (size_t)n_rows * (size_t)nk;
  float *mp = m;
  float *lp = lse;
  for (; srow < srow_end; srow += nk, ++mp, ++lp) {
    float mx = ninf;
    const float *s = srow;
    const float *s_end = srow + nk;
    for (; s < s_end; ++s)
      if (*s > mx)
        mx = *s;
    *mp = mx;
    if (!(mx > -1e20f))
      continue;
    float z = 0.f;
    float *sp = srow;
    for (; sp < s_end; ++sp) {
      const float e = (*sp == ninf) ? 0.f : expf(*sp - mx);
      *sp = e;
      z += e;
    }
    *lp = z;
  }

  /* Pass 2: dequant V once, axpy into every GQA row. */
  alignas(64) float vf[HGA_MAX_DH];
  const int8_t *vwalk = packed ? vhp + (size_t)k0 * (size_t)dh : nullptr;
  const float *vsp = packed ? vsc + k0 : nullptr;
  const int *kpv = keys + k0;
  const int *kpv_end = kpv + nk;
  const float *wr0 = scores;
  for (; kpv < kpv_end; ++kpv, ++wr0) {
    const int8_t *v = packed ? vwalk : vhp + (size_t)*kpv * (size_t)dh;
    const float vs = packed ? *vsp : vsc[*kpv];
    if (kpv + 1 < kpv_end)
      prefetch_bytes(packed ? vwalk + dh : vhp + (size_t)kpv[1] * (size_t)dh,
                     dh);
    const float *wr = wr0;
    const float *wr_end = wr0 + (size_t)n_rows * (size_t)nk;
    float *ar = acc;
    if (dh <= HGA_MAX_DH) {
      hga_i8_to_f32_scale(vf, v, vs, dh);
      for (; wr < wr_end; wr += nk, ar += dh) {
        if (*wr != 0.f)
          hga_axpy_f32(ar, vf, *wr, dh);
      }
    } else {
      for (; wr < wr_end; wr += nk, ar += dh) {
        if (*wr != 0.f)
          hga_axpy_i8(ar, v, (*wr) * vs, dh);
      }
    }
    if (packed) {
      vwalk += dh;
      ++vsp;
    }
  }
}

/* Decode (n_q=1) INT8: one KV head, packed sequential K/V, two-pass softmax.
 * Writes directly to `out`. No tile merge. */
static void decode_i8_kvhead(const int8_t *q8, const float *qsc, int h0,
                             int n_rep, int q_head_stride, int qsc_head_stride,
                             int H, int dh, float attn_scale,
                             const int8_t *kpack, const int8_t *vpack,
                             const float *ksc, const float *vsc, int cap,
                             const int *keys, int n_keys, int start_pos,
                             float *out, int out_head_stride, float *scores) {
  const float ninf = -std::numeric_limits<float>::infinity();
  if (n_keys <= 0 || n_rep <= 0)
    return;
  const int8_t *qptr[16];
  float qfac[16];
  {
    const int8_t *qp = q8 + (size_t)h0 * (size_t)q_head_stride;
    const float *qs = qsc + (size_t)h0 * (size_t)qsc_head_stride;
    const int8_t **qpp = qptr;
    const int8_t **qpp_end = qptr + n_rep;
    float *qf = qfac;
    for (; qpp < qpp_end;
         ++qpp, ++qf, qs += qsc_head_stride, qp += q_head_stride) {
      *qpp = qp;
      *qf = (*qs) * attn_scale;
    }
  }
  const int8_t *k = kpack;
  const float *ksp = ksc;
  float *sc = scores;
  const int *key = keys;
  const int *key_end = keys + n_keys;
  for (; key < key_end; ++key) {
    if (key + 1 < key_end)
      prefetch_bytes(k + dh, dh);
    float *p = sc;
    float *p_end = sc + n_rep;
    if (*key > start_pos) {
      for (; p < p_end; ++p)
        *p = ninf;
    } else {
      const float ks = *ksp;
      const int8_t **qp = qptr;
      const float *qf = qfac;
      for (; p < p_end; ++p, ++qp, ++qf)
        *p = (float)hga_dot_i8(*qp, k, dh) * (*qf) * ks;
    }
    sc += n_rep;
    k += dh;
    ++ksp;
  }
  float invz[16];
  float *iz = invz;
  float *iz_end = invz + n_rep;
  float *s0 = scores;
  for (; iz < iz_end; ++iz, ++s0) {
    float mx = ninf, z = 0.f;
    const float *s = s0;
    const float *s_end = s0 + (size_t)n_keys * (size_t)n_rep;
    for (; s < s_end; s += n_rep)
      if (*s > mx)
        mx = *s;
    if (mx > -1e20f) {
      float *sp = s0;
      for (; sp < s_end; sp += n_rep) {
        const float e = (*sp == ninf) ? 0.f : expf(*sp - mx);
        *sp = e;
        z += e;
      }
    }
    *iz = (z > 0.f) ? 1.f / z : 0.f;
  }
  alignas(64) float acc[16 * HGA_MAX_DH];
  std::memset(acc, 0, (size_t)n_rep * (size_t)dh * sizeof(float));
  alignas(64) float vf[HGA_MAX_DH];
  const int8_t *v = vpack;
  const float *vsp = vsc;
  const float *w = scores;
  const float *w_end = scores + (size_t)n_keys * (size_t)n_rep;
  for (; w < w_end; w += n_rep) {
    if (w + n_rep < w_end)
      prefetch_bytes(v + dh, dh);
    hga_i8_to_f32_scale(vf, v, *vsp, dh);
    const float *wr = w;
    const float *wr_end = w + n_rep;
    const float *izp = invz;
    float *ar = acc;
    for (; wr < wr_end; ++wr, ++izp, ar += dh) {
      const float a = (*wr) * (*izp);
      if (a != 0.f)
        hga_axpy_f32(ar, vf, a, dh);
    }
    v += dh;
    ++vsp;
  }
  float *dst = out + (size_t)h0 * (size_t)out_head_stride;
  const float *ar = acc;
  const float *ar_end = acc + n_rep * dh;
  for (; ar < ar_end; ar += dh, dst += out_head_stride)
    std::memcpy(dst, ar, (size_t)dh * sizeof(float));
  (void)H;
  (void)cap;
}

/* Small speculative-verify batch for one KV head.  Unlike decode_i8_kvhead(),
 * Q is [head, token, dh] and all n_q queries share one packed K/V list.
 * Causality is preserved by masking keys newer than start_pos + token.
 *
 * The old verify path called the one-token kernel n_q times.  That re-read and
 * dequantized the same packed K/V n_q times and created n_q OpenMP regions per
 * layer.  Here a KV head is one task, so each K and V vector is streamed once
 * while its dot products / weighted sums are accumulated for every query. */
static void decode_i8_kvhead_batch(const int8_t *q8, const float *qsc, int h0,
                                   int n_rep, int n_q, int H, int dh,
                                   float attn_scale, const int8_t *kpack,
                                   const int8_t *vpack, const float *ksc,
                                   const float *vsc, const int *keys,
                                   int n_keys, int start_pos, float *out,
                                   bool out_tok, float *scores, float *invz,
                                   float *acc) {
  const int n_rows = n_rep * n_q;
  const size_t q_head_stride = (size_t)n_q * (size_t)dh;
  const size_t qsc_head_stride = (size_t)n_q;
  const size_t score_key_stride = (size_t)n_rows;
  const size_t acc_row_stride = (size_t)dh;
  const float ninf = -std::numeric_limits<float>::infinity();
  if (n_keys <= 0 || n_rep <= 0 || n_q <= 0)
    return;

  /* Scores are key-major: adjacent rows use the same K vector while it is in
   * cache.  Rows are [token, query-head-within-KV-head]. */
  const int *key = keys;
  const int *key_end = keys + n_keys;
  const int8_t *k = kpack;
  const float *ks = ksc;
  float *sc = scores;
  /* These are per KV-head, not per key.  Keep the inner key loop to pointer
   * increments only; the query token/head strides are already precomputed. */
  const int8_t *const q_head0 = q8 + (size_t)h0 * q_head_stride;
  const float *const q_scale0 = qsc + (size_t)h0 * qsc_head_stride;
  for (; key < key_end; ++key, k += dh, ++ks, sc += score_key_stride) {
    const float kfac = *ks * attn_scale;
    const int8_t *q_head = q_head0;
    const float *q_scale = q_scale0;
    float *score_row = sc;
    int q_pos = start_pos;
    float *const score_rows_end = sc + score_key_stride;
    for (; score_row < score_rows_end; ++q_pos, score_row += n_rep) {
      if (*key > q_pos) {
        std::fill(score_row, score_row + n_rep, ninf);
      } else {
        const int8_t *qr = q_head;
        const float *qs = q_scale;
        float *score_end = score_row + n_rep;
        for (float *dst = score_row; dst < score_end;
             ++dst, qr += q_head_stride, qs += qsc_head_stride) {
          *dst = (float)hga_dot_i8(qr, k, dh) * (*qs) * kfac;
        }
      }
      q_head += dh;
      ++q_scale;
    }
  }

  const float *scores_end = scores + (size_t)n_keys * score_key_stride;
  for (int row = 0; row < n_rows; ++row) {
    float m = ninf;
    const float *sp = scores + row;
    for (; sp < scores_end; sp += score_key_stride)
      m = std::max(m, *sp);
    float z = 0.f;
    if (m > -1e20f) {
      for (float *spw = scores + row; spw < scores_end;
           spw += score_key_stride) {
        *spw = *spw == ninf ? 0.f : expf(*spw - m);
        z += *spw;
      }
    }
    invz[row] = z > 0.f ? 1.f / z : 0.f;
  }

  std::memset(acc, 0, (size_t)n_rows * (size_t)dh * sizeof(float));
  alignas(64) float vf[HGA_MAX_DH];
  const int8_t *v = vpack;
  const float *vs = vsc;
  const float *score_key = scores;
  for (; score_key < scores_end; score_key += score_key_stride, v += dh, ++vs) {
    hga_i8_to_f32_scale(vf, v, *vs, dh);
    const float *sp = score_key;
    const float *iz = invz;
    float *ar = acc;
    float *ar_end = acc + (size_t)n_rows * acc_row_stride;
    for (; ar < ar_end; ++sp, ++iz, ar += acc_row_stride) {
      const float a = *sp * *iz;
      if (a != 0.f)
        hga_axpy_f32(ar, vf, a, dh);
    }
  }

  if (out_tok) {
    float *dst_tok = out + (size_t)h0 * (size_t)dh;
    const size_t dst_tok_stride = (size_t)H * (size_t)dh;
    const float *src_tok = acc;
    for (int t = 0; t < n_q; ++t, dst_tok += dst_tok_stride,
             src_tok += (size_t)n_rep * acc_row_stride) {
      const float *src = src_tok;
      float *dst = dst_tok;
      for (int r = 0; r < n_rep; ++r, src += dh, dst += dh)
        std::memcpy(dst, src, (size_t)dh * sizeof(float));
    }
  } else {
    const size_t src_tok_stride = (size_t)n_rep * acc_row_stride;
    const size_t dst_head_stride = (size_t)n_q * (size_t)dh;
    float *dst_head = out + (size_t)h0 * dst_head_stride;
    for (int r = 0; r < n_rep; ++r, dst_head += dst_head_stride) {
      const float *src = acc + (size_t)r * acc_row_stride;
      float *dst = dst_head;
      for (int t = 0; t < n_q; ++t, src += src_tok_stride, dst += dh)
        std::memcpy(dst, src, (size_t)dh * sizeof(float));
    }
  }
}

/* One output row of the small verify kernel.  The fused kernel above is
 * cache-efficient when there are only a few CPU workers, but it exposes just
 * n_kv_heads (four for Qwen3-27B) tasks.  On the 40-core decode host that
 * leaves almost every core idle.  A row is one (proposed-token, query-head)
 * pair and owns its score, normalizer and output vector, so rows can execute
 * without locks.  Re-reading the compact INT8 V vector is much cheaper than
 * serialising 36 otherwise idle cores; it is only used for the 1..3 real-token
 * fixed verify graph. */
static void decode_i8_kvhead_batch_row(
    const int8_t *q8, const float *qsc, int h0, int n_rep, int n_q, int H,
    int dh, float attn_scale, const int8_t *kpack, const int8_t *vpack,
    const float *ksc, const float *vsc, const int *keys, int n_keys,
    int start_pos, int row, float *out, bool out_tok, float *scores,
    float *invz, float *acc) {
  if (n_keys <= 0 || n_rep <= 0 || n_q <= 0)
    return;
  const int token = row / n_rep;
  const int rep_i = row - token * n_rep;
  const int head = h0 + rep_i;
  const int q_pos = start_pos + token;
  const int8_t *qr = q8 + ((size_t)head * (size_t)n_q + (size_t)token) * dh;
  const float qfac =
      qsc[(size_t)head * (size_t)n_q + (size_t)token] * attn_scale;
  const size_t score_stride = (size_t)n_rep * (size_t)n_q;
  const float ninf = -std::numeric_limits<float>::infinity();

  float m = ninf;
  const int *key = keys;
  const int *key_end = keys + n_keys;
  const int8_t *k = kpack;
  const float *ks = ksc;
  float *sp = scores + row;
  for (; key < key_end; ++key, k += dh, ++ks, sp += score_stride) {
    const float score =
        *key > q_pos ? ninf : (float)hga_dot_i8(qr, k, dh) * qfac * *ks;
    *sp = score;
    m = std::max(m, score);
  }

  float z = 0.f;
  if (m > -1e20f) {
    for (sp = scores + row; sp < scores + (size_t)n_keys * score_stride;
         sp += score_stride) {
      const float w = *sp == ninf ? 0.f : expf(*sp - m);
      *sp = w;
      z += w;
    }
  }
  const float iz = z > 0.f ? 1.f / z : 0.f;
  invz[row] = iz;
  float *ar = acc + (size_t)row * dh;
  std::memset(ar, 0, (size_t)dh * sizeof(float));

  alignas(64) float vf[HGA_MAX_DH];
  const int8_t *v = vpack;
  const float *vs = vsc;
  for (sp = scores + row; sp < scores + (size_t)n_keys * score_stride;
       sp += score_stride, v += dh, ++vs) {
    const float a = *sp * iz;
    if (a == 0.f)
      continue;
    hga_i8_to_f32_scale(vf, v, *vs, dh);
    hga_axpy_f32(ar, vf, a, dh);
  }

  float *dst = out_tok
                   ? out + ((size_t)token * (size_t)H + (size_t)head) * dh
                   : out + ((size_t)head * (size_t)n_q + (size_t)token) * dh;
  std::memcpy(dst, ar, (size_t)dh * sizeof(float));
}

/* Decode (n_q == 1, and spec n_q <= 8 one query at a time): packed sequential
 * KV, one task per KV head. Large prefill has its own tile kernel.
 * spans_in: precomputed routing (spec verify routes once with pooled Q);
 * routing is skipped and n_sel_in/n_open_in are reported in the stats. */
static void hga_attend_decode(hga_session *s, int layer, int start_pos,
                              const float *q, int q_hs, float *out,
                              int out_layout, hga_stats *stats,
                              const std::vector<Span> *spans_in = nullptr,
                              int n_sel_in = 0, int n_open_in = 0) {
  Layer &L = s->layers[(size_t)layer];
  const int H = s->cfg.n_q_heads;
  const int dh = s->cfg.head_dim;
  const bool i8 = s->cfg.prec == HGA_PREC_I8;
  const bool out_tok = (out_layout == HGA_OUT_TOKEN_MAJOR);
  const int nthr = std::max(1, s->cfg.n_threads);
  const int n_kvh = s->cfg.n_kv_heads;
  const int rep = H / n_kvh;
  const double t0 = now_ms();

  const int q_last = start_pos;
  const int n_closed_view = std::min(L.n_closed, q_last / s->cfg.chunk_size);
  RouteSet rs;
  int n_sel = n_sel_in, n_open = n_open_in;
  std::vector<Span> spans;
  std::vector<std::vector<int>> keys_h;
  bool use_wave = false;
  if (!spans_in) {
    grow(s->scratch_q_pool, (size_t)H * (size_t)dh);
    float *q_pool = s->scratch_q_pool.data();
    {
      const float *qh = q;
      float *dst = q_pool;
      float *dst_end = q_pool + (size_t)H * (size_t)dh;
      for (; dst < dst_end; dst += dh, qh += q_hs)
        std::memcpy(dst, qh, (size_t)dh * sizeof(float));
    }
    if (s->cfg.router == HGA_ROUTER_WAVE && wave_index_ready(s, L)) {
      use_wave = true;
      route_wave(s, L, q_pool, rs, &n_sel, &n_open);
      collect_keys_wave(s, L, rs, q_last, keys_h);
    } else {
      route_layer(s, L, q_pool, n_closed_view, 1, rs, &n_sel, &n_open);
      collect_spans(s, L, rs, n_closed_view, spans);
    }
  }
  const std::vector<Span> &spans_use = spans_in ? *spans_in : spans;

  int8_t *q8p = nullptr;
  float *qscp = nullptr;
  const float *qf_hm = q;
  if (i8) {
    s->scratch_q8.resize((size_t)H * (size_t)dh);
    s->scratch_qsc.resize((size_t)H);
    q8p = s->scratch_q8.data();
    qscp = s->scratch_qsc.data();
    const float *qh = q;
    const float *qh_end = q + (size_t)H * (size_t)q_hs;
    int8_t *d8 = q8p;
    float *sc = qscp;
    for (; qh < qh_end; qh += q_hs, d8 += dh, ++sc)
      quant_vec_i8(qh, dh, d8, sc);
  } else if (!(q_hs == dh)) {
    s->scratch_qf.resize((size_t)H * (size_t)dh);
    float *dst = s->scratch_qf.data();
    const float *qh = q;
    const float *qh_end = q + (size_t)H * (size_t)q_hs;
    for (; qh < qh_end; qh += q_hs, dst += dh)
      std::memcpy(dst, qh, (size_t)dh * sizeof(float));
    qf_hm = s->scratch_qf.data();
  }
  const double t1 = now_ms();

  std::vector<const int *> keyp((size_t)n_kvh, nullptr);
  std::vector<int> nkh((size_t)n_kvh, 0);
  int n_keys_max = 0;
  s->scratch_keys.clear();
  if (use_wave) {
    const int **kp = keyp.data();
    int *nk = nkh.data();
    std::vector<int> *khp = keys_h.data();
    std::vector<int> *khp_end = khp + (size_t)n_kvh;
    for (; khp < khp_end; ++khp, ++kp, ++nk) {
      *kp = khp->empty() ? nullptr : khp->data();
      *nk = (int)khp->size();
      n_keys_max = std::max(n_keys_max, *nk);
      s->scratch_keys.insert(s->scratch_keys.end(), khp->begin(), khp->end());
    }
  } else {
    collect_keys(s, L, spans_use, q_last, s->scratch_keys);
    const int *keys0 =
        s->scratch_keys.empty() ? nullptr : s->scratch_keys.data();
    n_keys_max = (int)s->scratch_keys.size();
    const int **kp = keyp.data();
    const int **kp_end = kp + n_kvh;
    int *nk = nkh.data();
    for (; kp < kp_end; ++kp, ++nk) {
      *kp = keys0;
      *nk = n_keys_max;
    }
  }

  PackOp pack_op = PACK_REBUILD;
  const bool use_pack = !use_wave && n_keys_max > 0;
  if (use_pack) {
    pack_op =
        pack_sync(s, L, s->cfg, s->scratch_keys.data(), n_keys_max, nthr);
    if (pack_op == PACK_REBUILD)
      ++s->metrics.packed_rebuilds;
    else if (pack_op == PACK_APPEND)
      ++s->metrics.packed_appends;
    else
      ++s->metrics.packed_reuses;
  }
  s->last_keys = s->scratch_keys;
  s->last_keys_layer = layer;

  if (use_pack && i8 && q8p && n_keys_max > 0 && n_kvh <= 16 && rep <= 16 &&
      dh <= HGA_MAX_DH) {
    grow(s->scratch_scores, (size_t)n_kvh * (size_t)rep * (size_t)n_keys_max);
    const size_t pack_vec = (size_t)L.pack.cap * (size_t)dh;
    const size_t q_vec = (size_t)rep * (size_t)dh;
    const size_t sc_stride = (size_t)rep * (size_t)n_keys_max;
    const int8_t *q8s[16], *kps[16], *vps[16];
    const float *qscs[16], *kss[16], *vss[16];
    const int *keyss[16];
    float *scs[16], *outs[16];
    int nks[16];
    {
      const int8_t *q8 = q8p;
      const float *qs = qscp;
      float *o = out;
      const int8_t *kpack = L.pack.k8.data();
      const int8_t *vpack = L.pack.v8.data();
      const float *ksc = L.pack.ksc.data();
      const float *vsc = L.pack.vsc.data();
      float *scores = s->scratch_scores.data();
      const int **kp = keyp.data();
      const int *nk = nkh.data();
      const int8_t **q8d = q8s, **q8e = q8s + n_kvh;
      const float **qsd = qscs;
      const int8_t **kd = kps, **vd = vps;
      const float **ksd = kss, **vsd = vss;
      float **scd = scs, **od = outs;
      const int **keyd = keyss;
      int *nkd = nks;
      for (; q8d < q8e; ++q8d, ++qsd, ++kd, ++vd, ++ksd, ++vsd, ++scd, ++od,
                        ++keyd, ++nkd, ++kp, ++nk, q8 += q_vec, qs += rep,
                        o += q_vec, kpack += pack_vec, vpack += pack_vec,
                        ksc += L.pack.cap, vsc += L.pack.cap,
                        scores += sc_stride) {
        *q8d = q8;
        *qsd = qs;
        *kd = kpack;
        *vd = vpack;
        *ksd = ksc;
        *vsd = vsc;
        *scd = scores;
        *od = o;
        *keyd = *kp;
        *nkd = *nk;
      }
    }
#if defined(_OPENMP)
    /* spread: OMP_PROC_BIND=close would pin these 4 tasks to cores 0-3,
     * the same socket-0 cores the L2 GEMV/prefetch pool also uses. */
#pragma omp parallel for num_threads(n_kvh) proc_bind(spread) schedule(static)
#endif
    for (int kh = 0; kh < n_kvh; ++kh) {
      if (!keyss[kh] || nks[kh] <= 0)
        continue;
      decode_i8_kvhead(q8s[kh], qscs[kh], 0, rep, dh, 1, H, dh, s->scale,
                       kps[kh], vps[kh], kss[kh], vss[kh], L.pack.cap,
                       keyss[kh], nks[kh], start_pos, outs[kh], dh, scs[kh]);
    }
  } else {
    /* F16 / wave: one tile per KV head, no 2-D prefill grid and no merge. */
    const int n_rows = rep;
    grow(s->scratch_part_m, (size_t)n_kvh * (size_t)n_rows);
    grow(s->scratch_part_lse, (size_t)n_kvh * (size_t)n_rows);
    grow(s->scratch_part_acc, (size_t)n_kvh * (size_t)n_rows * (size_t)dh);
    grow(s->scratch_scores,
         (size_t)n_kvh * (size_t)n_rows * (size_t)n_keys_max);
    const size_t acc_stride = (size_t)n_rows * (size_t)dh;
    const size_t sc_stride = (size_t)n_rows * (size_t)n_keys_max;
    float *ms[16], *lss[16], *acs[16], *scs[16], *outs[16];
    const int *keyss[16];
    int nks[16], h0s[16];
    {
      float *m = s->scratch_part_m.data();
      float *ls = s->scratch_part_lse.data();
      float *ac = s->scratch_part_acc.data();
      float *sc = s->scratch_scores.data();
      float *o = out;
      int h0 = 0;
      const int **kp = keyp.data();
      const int *nk = nkh.data();
      float **md = ms, **me = ms + n_kvh;
      float **ld = lss, **ad = acs, **sd = scs, **od = outs;
      const int **keyd = keyss;
      int *nkd = nks, *hd = h0s;
      for (; md < me; ++md, ++ld, ++ad, ++sd, ++od, ++keyd, ++nkd, ++hd, ++kp,
                      ++nk, m += n_rows, ls += n_rows, ac += acc_stride,
                      sc += sc_stride, o += acc_stride, h0 += rep) {
        *md = m;
        *ld = ls;
        *ad = ac;
        *sd = sc;
        *od = o;
        *keyd = *kp;
        *nkd = *nk;
        *hd = h0;
      }
    }
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_kvh) proc_bind(spread) schedule(static)
#endif
    for (int kh = 0; kh < n_kvh; ++kh) {
      const int h0 = h0s[kh], h1 = h0 + rep;
      const int n_keys = nks[kh];
      const int *keys = keyss[kh];
      float *m = ms[kh], *ls = lss[kh], *ac = acs[kh], *sc = scs[kh];
      if (i8) {
        flash_tile_i8(
            s, L, kh, start_pos, 1, h0, h1, 0, 1, keys, 0, n_keys, q8p, qscp, m,
            ls, ac, sc, use_pack ? L.pack.k8.data() : nullptr,
            use_pack ? L.pack.v8.data() : nullptr,
            use_pack ? L.pack.ksc.data() : nullptr,
            use_pack ? L.pack.vsc.data() : nullptr, use_pack ? L.pack.cap : 0);
      } else {
        flash_tile_f16(
            s, L, kh, start_pos, 1, h0, h1, 0, 1, keys, 0, n_keys, qf_hm, m, ls,
            ac, sc, use_pack ? L.pack.k16.data() : nullptr,
            use_pack ? L.pack.v16.data() : nullptr, use_pack ? L.pack.cap : 0);
      }
      if (use_wave && n_open > 0 && kh < (int)rs.est.size() &&
          !rs.est[(size_t)kh].empty()) {
        flash_est(s, L, kh, start_pos, 1, h0, h1, 0, 1,
                  rs.est[(size_t)kh].data(), (int)rs.est[(size_t)kh].size(), q,
                  q_hs, dh, m, ls, ac);
      }
      float *o = ac;
      const float *lsp = ls;
      float *dst = outs[kh];
      float *o_end = ac + acc_stride;
      for (; o < o_end; o += dh, ++lsp, dst += dh) {
        if (*lsp > 0.f)
          hga_scale_f32(o, 1.f / *lsp, dh);
        else
          std::memset(o, 0, (size_t)dh * sizeof(float));
        std::memcpy(dst, o, (size_t)dh * sizeof(float));
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
    stats->n_attended_tokens = std::max(1, n_keys_max);
    stats->sparsity = (L.n_kv > 0) ? (float)n_keys_max / (float)L.n_kv : 1.f;
    stats->ms_route = t1 - t0;
    stats->ms_attn = t2 - t1;
    stats->pack_rebuild = (pack_op == PACK_REBUILD);
    stats->pack_append = (pack_op == PACK_APPEND);
    stats->pack_reuse = (pack_op == PACK_REUSE);
  }
}

/* Prefill (n_q > 8): unpacked KV, 2-D Q/K tiles, no PackedKV.
 * Spec verify is n_q = K+1 (typically 2..4). Route ONCE with mean-pooled Q
 * (prefill style) — adjacent spec tokens share the routed chunk/group set —
 * then attend per query. collect_keys clamps each query to start_pos+i, so
 * causality holds and pack_sync appends instead of rebuilding per query. */
static void hga_attend_f32_strided(hga_session *s, int layer, int start_pos,
                                   int n_q, const float *q, int q_hs, int q_ts,
                                   float *out, int out_layout,
                                   hga_stats *stats) {
  /* Hierarchical INT8 decode can use the same disjoint packed-KV tiles as
   * short verify.  Keep the original one-query kernel as the F16/WAVE path
   * and as the explicit HGA_VERIFY_TILES=0 A/B fallback. */
  const bool tile_single =
      n_q == 1 && hga_verify_batch_enabled() && hga_verify_tiles_enabled() &&
      s->cfg.prec == HGA_PREC_I8 && s->cfg.router != HGA_ROUTER_WAVE;
  if (n_q == 1 && !tile_single) {
    hga_attend_decode(s, layer, start_pos, q, q_hs, out, out_layout, stats);
    return;
  }
  if (n_q >= 1 && n_q <= 8) {
    Layer &Lv = s->layers[(size_t)layer];
    const int H = s->cfg.n_q_heads;
    const int dh = s->cfg.head_dim;
    const bool tok_major = (out_layout == HGA_OUT_TOKEN_MAJOR);
    std::vector<float> tmp;
    const double tr0 = now_ms();
    RouteSet rs;
    int n_sel = 0, n_open = 0;
    std::vector<Span> spans;
    bool routed = false;
    /* Wave router stays per-query (its key lists are per KV head). */
    if (!(s->cfg.router == HGA_ROUTER_WAVE && wave_index_ready(s, Lv))) {
      grow(s->scratch_q_pool, (size_t)H * (size_t)dh);
      float *pooled = s->scratch_q_pool.data();
      const size_t tok_span = (size_t)n_q * (size_t)q_ts;
      float *dst = pooled;
      const float *qh = q;
      const float *qh_end = q + (size_t)H * (size_t)q_hs;
      if (n_q == 1) {
        for (; qh < qh_end; dst += dh, qh += q_hs)
          std::memcpy(dst, qh, (size_t)dh * sizeof(float));
      } else {
        for (; qh < qh_end; dst += dh, qh += q_hs) {
          std::memset(dst, 0, (size_t)dh * sizeof(float));
          const float *qt = qh;
          const float *qt_end = qt + tok_span;
          for (; qt < qt_end; qt += q_ts)
            hga_axpy_f32(dst, qt, 1.f, dh);
          hga_scale_f32(dst, 1.f / (float)n_q, dh);
        }
      }
      const int q_last = start_pos + n_q - 1;
      const int n_closed_view =
          std::min(Lv.n_closed, q_last / s->cfg.chunk_size);
      route_layer(s, Lv, pooled, n_closed_view, n_q, rs, &n_sel, &n_open);
      collect_spans(s, Lv, rs, n_closed_view, spans);
      routed = true;
    }
    const double tr1 = now_ms();

    const int n_kvh = s->cfg.n_kv_heads;
    const int rep = n_kvh > 0 ? H / n_kvh : 0;
    /* The HGA-2 INT8 short-batch path routes and packs the full final-query KV
     * list once, then streams every packed K/V vector once per key tile for
     * all real queries. This is CPU-only and therefore leaves the fixed CUDA
     * graph untouched. Wave has per-query key lists and F16 has its own path.
     */
    const bool verify_fused_ok = hga_verify_batch_enabled() && routed &&
                                 s->cfg.prec == HGA_PREC_I8 && n_kvh > 0 &&
                                 H % n_kvh == 0 && dh <= HGA_MAX_DH && rep > 0;
    if (!verify_fused_ok && n_q <= 3) {
      static bool logged_verify_fallback = false;
      if (!logged_verify_fallback) {
        logged_verify_fallback = true;
        std::fprintf(stderr,
                     "hga: fused verify disabled batch=%d routed=%d i8=%d "
                     "kvh=%d H=%d dh=%d rep=%d\n",
                     (int)hga_verify_batch_enabled(), (int)routed,
                     (int)(s->cfg.prec == HGA_PREC_I8), n_kvh, H, dh, rep);
      }
    }
    if (verify_fused_ok) {
      s->scratch_q8.resize((size_t)H * (size_t)n_q * (size_t)dh);
      s->scratch_qsc.resize((size_t)H * (size_t)n_q);
      int8_t *q8 = s->scratch_q8.data();
      float *qsc = s->scratch_qsc.data();
      const float *qh = q;
      int8_t *q8h = q8;
      float *qsch = qsc;
      for (int h = 0; h < H;
           ++h, qh += q_hs, q8h += (size_t)n_q * (size_t)dh, qsch += n_q) {
        const float *qt = qh;
        int8_t *q8t = q8h;
        float *qsct = qsch;
        for (int t = 0; t < n_q; ++t, qt += q_ts, q8t += dh, ++qsct)
          quant_vec_i8(qt, dh, q8t, qsct);
      }

      const int q_last = start_pos + n_q - 1;
      s->scratch_keys.clear();
      collect_keys(s, Lv, spans, q_last, s->scratch_keys);
      const int n_keys = (int)s->scratch_keys.size();
      const bool use_pack = n_keys > 0;
      PackOp pack_op = PACK_REBUILD;
      const double tpack0 = now_ms();
      if (use_pack) {
        pack_op = pack_sync(s, Lv, s->cfg, s->scratch_keys.data(), n_keys,
                            std::max(1, s->cfg.n_threads));
        if (pack_op == PACK_REBUILD)
          ++s->metrics.packed_rebuilds;
        else if (pack_op == PACK_APPEND)
          ++s->metrics.packed_appends;
        else
          ++s->metrics.packed_reuses;
      }
      const double tpack1 = now_ms();
      s->last_keys = s->scratch_keys;
      s->last_keys_layer = layer;

      if (use_pack) {
        const int n_rows = rep * n_q;
        const int nthr = std::max(1, s->cfg.n_threads);
        const size_t score_stride = (size_t)n_rows * (size_t)n_keys;
        const size_t row_stride = (size_t)n_rows;
        const size_t acc_stride = row_stride * (size_t)dh;
        const size_t pack_vec = (size_t)Lv.pack.cap * (size_t)dh;
        /* Qwen3's verify graph has at most three real tokens.  Split its
         * 4 KV-head jobs into 4 * (6 * n_q) independent output rows so the
         * 40 physical decode cores participate.  For wider speculative
         * batches retain the cache-streaming fused kernel. */
        if (hga_verify_rows_enabled() && n_q <= 3 && nthr > n_kvh) {
          grow(s->scratch_scores, (size_t)n_kvh * score_stride);
          grow(s->scratch_verify_invz, (size_t)n_kvh * row_stride);
          grow(s->scratch_verify_acc, (size_t)n_kvh * acc_stride);
          const int n_work = n_kvh * n_rows;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(std::min(nthr, n_work)) proc_bind(spread) \
    schedule(static)
#endif
          for (int work = 0; work < n_work; ++work) {
            const int kh = work / n_rows;
            const int row = work - kh * n_rows;
            const int h0 = kh * rep;
            const int8_t *kpack = Lv.pack.k8.data() + (size_t)kh * pack_vec;
            const int8_t *vpack = Lv.pack.v8.data() + (size_t)kh * pack_vec;
            const float *ksc =
                Lv.pack.ksc.data() + (size_t)kh * (size_t)Lv.pack.cap;
            const float *vsc =
                Lv.pack.vsc.data() + (size_t)kh * (size_t)Lv.pack.cap;
            float *scores =
                s->scratch_scores.data() + (size_t)kh * score_stride;
            float *invz =
                s->scratch_verify_invz.data() + (size_t)kh * row_stride;
            float *acc = s->scratch_verify_acc.data() + (size_t)kh * acc_stride;
            decode_i8_kvhead_batch_row(
                q8, qsc, h0, rep, n_q, H, dh, s->scale, kpack, vpack, ksc, vsc,
                s->scratch_keys.data(), n_keys, start_pos, row, out, tok_major,
                scores, invz, acc);
          }
        } else if (hga_verify_tiles_enabled() && n_q <= 3 &&
                   nthr >= 2 * n_kvh && n_keys >= 2) {
          /* Four KV heads x eight disjoint key tiles gives 32 workers on the
           * production shape.  A worker retains all rep*n_q rows, so every K
           * and V vector is loaded/dequantized once and reused by the six GQA
           * heads and up to three real tokens. Only the small partial online-
           * softmax states are merged after the parallel key pass.
           *
           * Cap the tile count by available workers and keys.  Tile bounds,
           * scratch strides, and base addresses are initialized before the
           * hot loops; flash_tile_i8 then walks K/V/Q/score data by pointer
           * increments rather than repeatedly multiplying indexes. */
          constexpr int max_k_tiles = 8;
          const int workers_per_head = std::max(1, nthr / n_kvh);
          const int nkt =
              std::min(max_k_tiles, std::min(n_keys, workers_per_head));
          const bool use_smt =
              hga_verify_smt_enabled() && rep >= 2 && rep % 2 == 0;
          const int nht = use_smt ? 2 : 1;
          const int heads_per_part = (rep + nht - 1) / nht;
          const int part_rows = heads_per_part * n_q;
          const int n_parts = n_kvh * nkt * nht;
          /* With SMT, create the complete 2*nthr team even though Qwen has 64
           * useful jobs.  GNU OpenMP then binds worker pairs to the same
           * OMP_PLACES=cores place; using a 64-thread team would instead put
           * its first 40 workers on different cores. */
          const int nthr_attn = use_smt ? 2 * nthr : std::min(nthr, n_parts);
          const int max_nk = (n_keys + nkt - 1) / nkt;
          const size_t align_floats = 64 / sizeof(float);
          const auto pad_floats = [=](size_t n) {
            return (n + align_floats - 1) & ~(align_floats - 1);
          };
          const size_t state_part_stride = pad_floats((size_t)part_rows);
          const size_t acc_part_stride =
              pad_floats((size_t)part_rows * (size_t)dh);
          const size_t score_part_stride =
              pad_floats((size_t)part_rows * (size_t)max_nk);

          /* Leave room to align every scratch arena to a cache line.  Padded
           * per-part strides prevent adjacent workers from sharing a line. */
          grow(s->scratch_part_m,
               (size_t)n_parts * state_part_stride + align_floats - 1);
          grow(s->scratch_part_lse,
               (size_t)n_parts * state_part_stride + align_floats - 1);
          grow(s->scratch_part_acc,
               (size_t)n_parts * acc_part_stride + align_floats - 1);
          grow(s->scratch_scores,
               (size_t)n_parts * score_part_stride + align_floats - 1);
          const auto align64 = [](float *p) {
            const uintptr_t u = reinterpret_cast<uintptr_t>(p);
            return reinterpret_cast<float *>((u + 63u) & ~uintptr_t(63u));
          };
          float *const part_m = align64(s->scratch_part_m.data());
          float *const part_lse = align64(s->scratch_part_lse.data());
          float *const part_acc = align64(s->scratch_part_acc.data());
          float *const part_scores = align64(s->scratch_scores.data());

          int k_bounds[max_k_tiles + 1];
          int *kb = k_bounds;
          for (int kt = 0; kt <= nkt; ++kt, ++kb)
            *kb = (int)((int64_t)kt * (int64_t)n_keys / (int64_t)nkt);

          const auto run_part = [&](int kh, int kt, int ht) {
            const int part = (kh * nkt + kt) * nht + ht;
            float *m = part_m + (size_t)part * state_part_stride;
            float *ls = part_lse + (size_t)part * state_part_stride;
            float *ac = part_acc + (size_t)part * acc_part_stride;
            float *sc = part_scores + (size_t)part * score_part_stride;
            const int hg0 = kh * rep;
            const int h0 = hg0 + ht * heads_per_part;
            const int h1 = std::min(hg0 + rep, h0 + heads_per_part);
            flash_tile_i8(s, Lv, kh, start_pos, n_q, h0, h1, 0, n_q,
                          s->scratch_keys.data(), k_bounds[kt],
                          k_bounds[kt + 1], q8, qsc, m, ls, ac, sc,
                          Lv.pack.k8.data(), Lv.pack.v8.data(),
                          Lv.pack.ksc.data(), Lv.pack.vsc.data(), Lv.pack.cap);
          };

          const auto merge_head_tile = [&](int kh, int ht) {
            const int base = (kh * nkt) * nht + ht;
            float *m = part_m + (size_t)base * state_part_stride;
            float *ls = part_lse + (size_t)base * state_part_stride;
            float *ac = part_acc + (size_t)base * acc_part_stride;
            const int hg0 = kh * rep;
            const int h0 = hg0 + ht * heads_per_part;
            const int h1 = std::min(hg0 + rep, h0 + heads_per_part);
            const int rows = (h1 - h0) * n_q;
            for (int kt = 1; kt < nkt; ++kt) {
              const int part = (kh * nkt + kt) * nht + ht;
              merge_online_softmax(
                  m, ls, ac, part_m + (size_t)part * state_part_stride,
                  part_lse + (size_t)part * state_part_stride,
                  part_acc + (size_t)part * acc_part_stride, rows, dh);
            }

            float *ar = ac;
            const float *lsp = ls;
            float *const ar_end = ac + (size_t)rows * (size_t)dh;
            for (; ar < ar_end; ar += dh, ++lsp) {
              if (*lsp > 0.f)
                hga_scale_f32(ar, 1.f / *lsp, dh);
              else
                std::memset(ar, 0, (size_t)dh * sizeof(float));
            }

            if (!tok_major) {
              float *dst = out + (size_t)h0 * (size_t)n_q * (size_t)dh;
              std::memcpy(dst, ac, (size_t)rows * (size_t)dh * sizeof(float));
            } else {
              const size_t dst_tok_stride = (size_t)H * (size_t)dh;
              const size_t src_head_stride = (size_t)n_q * (size_t)dh;
              const float *src_head = ac;
              float *dst_head = out + (size_t)h0 * (size_t)dh;
              for (int r = h0; r < h1;
                   ++r, src_head += src_head_stride, dst_head += dh) {
                const float *src = src_head;
                float *dst = dst_head;
                const float *const src_end = src + src_head_stride;
                for (; src < src_end; src += dh, dst += dst_tok_stride)
                  std::memcpy(dst, src, (size_t)dh * sizeof(float));
              }
            }
          };

          if (use_smt) {
#if defined(_OPENMP)
#pragma omp parallel num_threads(nthr_attn) proc_bind(close)
#endif
            {
#if defined(_OPENMP)
#pragma omp for collapse(3) schedule(static, 1)
#endif
              for (int kh = 0; kh < n_kvh; ++kh) {
                for (int kt = 0; kt < nkt; ++kt) {
                  for (int ht = 0; ht < nht; ++ht)
                    run_part(kh, kt, ht);
                }
              }

#if defined(_OPENMP)
#pragma omp for collapse(2) schedule(static)
#endif
              for (int kh = 0; kh < n_kvh; ++kh)
                for (int ht = 0; ht < nht; ++ht)
                  merge_head_tile(kh, ht);
            }
          } else {
#if defined(_OPENMP)
#pragma omp parallel num_threads(nthr_attn) proc_bind(spread)
#endif
            {
#if defined(_OPENMP)
#pragma omp for collapse(2) schedule(static)
#endif
              for (int kh = 0; kh < n_kvh; ++kh) {
                for (int kt = 0; kt < nkt; ++kt)
                  run_part(kh, kt, 0);
              }

              /* Reuse the same OpenMP team after its implicit barrier. */
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
              for (int kh = 0; kh < n_kvh; ++kh)
                merge_head_tile(kh, 0);
            }
          }
        } else {
          grow(s->scratch_scores, (size_t)n_kvh * score_stride);
          grow(s->scratch_verify_invz, (size_t)n_kvh * row_stride);
          grow(s->scratch_verify_acc, (size_t)n_kvh * acc_stride);
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_kvh) proc_bind(spread) schedule(static)
#endif
          for (int kh = 0; kh < n_kvh; ++kh) {
            const int h0 = kh * rep;
            const int8_t *kpack = Lv.pack.k8.data() + (size_t)kh * pack_vec;
            const int8_t *vpack = Lv.pack.v8.data() + (size_t)kh * pack_vec;
            const float *ksc =
                Lv.pack.ksc.data() + (size_t)kh * (size_t)Lv.pack.cap;
            const float *vsc =
                Lv.pack.vsc.data() + (size_t)kh * (size_t)Lv.pack.cap;
            float *scores =
                s->scratch_scores.data() + (size_t)kh * score_stride;
            float *invz =
                s->scratch_verify_invz.data() + (size_t)kh * row_stride;
            float *acc = s->scratch_verify_acc.data() + (size_t)kh * acc_stride;
            decode_i8_kvhead_batch(q8, qsc, h0, rep, n_q, H, dh, s->scale,
                                   kpack, vpack, ksc, vsc,
                                   s->scratch_keys.data(), n_keys, start_pos,
                                   out, tok_major, scores, invz, acc);
          }
        }
        const double t2 = now_ms();
        if (stats) {
          std::memset(stats, 0, sizeof(*stats));
          stats->n_kv = Lv.n_kv;
          stats->n_closed_chunks = Lv.n_closed;
          stats->n_selected_chunks = n_sel;
          stats->n_opened_groups = n_open;
          stats->n_attended_tokens = n_keys;
          stats->sparsity = Lv.n_kv > 0 ? (float)n_keys / (float)Lv.n_kv : 1.f;
          stats->ms_route = tr1 - tr0;
          stats->ms_attn = t2 - tr1;
          stats->ms_pack = tpack1 - tpack0;
          stats->ms_kernel = t2 - tpack1;
          stats->pack_rebuild = (pack_op == PACK_REBUILD);
          stats->pack_append = (pack_op == PACK_APPEND);
          stats->pack_reuse = (pack_op == PACK_REUSE);
        }
        return;
      }
    }

    hga_stats acc{};
    if (!tok_major)
      tmp.resize((size_t)H * (size_t)dh);
    for (int i = 0; i < n_q; ++i) {
      const float *qi = q + (size_t)i * (size_t)q_ts;
      float *oi =
          tok_major ? out + (size_t)i * (size_t)H * (size_t)dh : tmp.data();
      hga_stats one{};
      hga_attend_decode(s, layer, start_pos + i, qi, q_hs, oi, out_layout, &one,
                        routed ? &spans : nullptr, n_sel, n_open);
      if (!tok_major) {
        for (int h = 0; h < H; ++h) {
          std::memcpy(out + ((size_t)h * (size_t)n_q + (size_t)i) * (size_t)dh,
                      tmp.data() + (size_t)h * (size_t)dh,
                      (size_t)dh * sizeof(float));
        }
      }
      if (i == 0) {
        acc = one;
      } else {
        acc.ms_route += one.ms_route;
        acc.ms_attn += one.ms_attn;
        acc.n_attended_tokens =
            std::max(acc.n_attended_tokens, one.n_attended_tokens);
        acc.n_selected_chunks =
            std::max(acc.n_selected_chunks, one.n_selected_chunks);
        acc.n_opened_groups =
            std::max(acc.n_opened_groups, one.n_opened_groups);
        acc.n_kv = one.n_kv;
        acc.n_closed_chunks = one.n_closed_chunks;
        acc.sparsity = one.sparsity;
        acc.pack_rebuild += one.pack_rebuild;
        acc.pack_append += one.pack_append;
        acc.pack_reuse += one.pack_reuse;
      }
    }
    if (stats) {
      *stats = acc;
      stats->ms_route += tr1 - tr0; /* shared pooled-Q routing, done once */
    }
    return;
  }
  Layer &L = s->layers[(size_t)layer];
  const int H = s->cfg.n_q_heads;
  const int dh = s->cfg.head_dim;
  const bool i8 = s->cfg.prec == HGA_PREC_I8;
  const bool out_tok = (out_layout == HGA_OUT_TOKEN_MAJOR);
  const int nthr = std::max(1, s->cfg.n_threads);
  const double t0 = now_ms();

  grow(s->scratch_q_pool, (size_t)H * (size_t)dh);
  float *q_pool = s->scratch_q_pool.data();
  {
    const size_t tok_span = (size_t)n_q * (size_t)q_ts;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(nthr) schedule(static) if (n_q >= 8)
#endif
    for (int h = 0; h < H; ++h) {
      float *dst = q_pool + (size_t)h * (size_t)dh;
      std::memset(dst, 0, (size_t)dh * sizeof(float));
      const float *qt = q + (size_t)h * (size_t)q_hs;
      const float *qt_end = qt + tok_span;
      for (; qt < qt_end; qt += q_ts)
        hga_axpy_f32(dst, qt, 1.f, dh);
      hga_scale_f32(dst, 1.f / (float)n_q, dh);
    }
  }
  const double t_pool = now_ms();

  const int q_last = start_pos + n_q - 1;
  const int n_closed_view = std::min(L.n_closed, q_last / s->cfg.chunk_size);
  RouteSet rs;
  RouteProfile route_profile;
  int n_sel = 0, n_open = 0;
  std::vector<Span> spans;
  const double t_route0 = now_ms();
  route_layer(s, L, q_pool, n_closed_view, n_q, rs, &n_sel, &n_open,
              &route_profile);
  const double t_route1 = now_ms();
  collect_spans(s, L, rs, n_closed_view, spans);
  const double t_spans = now_ms();

  const int n_kvh = s->cfg.n_kv_heads;
  int8_t *q8p = nullptr;
  float *qscp = nullptr;
  const float *qf_hm = nullptr;
  if (i8) {
    s->scratch_q8.resize((size_t)H * (size_t)n_q * (size_t)dh);
    s->scratch_qsc.resize((size_t)H * (size_t)n_q);
    q8p = s->scratch_q8.data();
    qscp = s->scratch_qsc.data();
    const size_t tok_span = (size_t)n_q * (size_t)q_ts;
    const size_t q8_hstride = (size_t)n_q * (size_t)dh;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(nthr) schedule(static) if (n_q >= 8)
#endif
    for (int h = 0; h < H; ++h) {
      const float *qt = q + (size_t)h * (size_t)q_hs;
      const float *qt_end = qt + tok_span;
      int8_t *d8 = q8p + (size_t)h * q8_hstride;
      float *sc = qscp + (size_t)h * (size_t)n_q;
      for (; qt < qt_end; qt += q_ts, d8 += dh, ++sc)
        quant_vec_i8(qt, dh, d8, sc);
    }
  } else {
    const bool already_hm = (q_ts == dh && q_hs == n_q * dh);
    if (already_hm) {
      qf_hm = q;
    } else {
      s->scratch_qf.resize((size_t)H * (size_t)n_q * (size_t)dh);
      float *dst0 = s->scratch_qf.data();
      const size_t tok_span = (size_t)n_q * (size_t)q_ts;
      const size_t dst_hstride = (size_t)n_q * (size_t)dh;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(nthr) schedule(static) if (n_q >= 8)
#endif
      for (int h = 0; h < H; ++h) {
        float *dst = dst0 + (size_t)h * dst_hstride;
        const float *qt = q + (size_t)h * (size_t)q_hs;
        const float *qt_end = qt + tok_span;
        for (; qt < qt_end; qt += q_ts, dst += dh)
          std::memcpy(dst, qt, (size_t)dh * sizeof(float));
      }
      qf_hm = s->scratch_qf.data();
    }
  }
  const double t1 = now_ms();

  const int rep = H / n_kvh;
  std::vector<const int *> keyp((size_t)n_kvh, nullptr);
  std::vector<int> nkh((size_t)n_kvh, 0);
  s->scratch_keys.clear();
  collect_keys(s, L, spans, q_last, s->scratch_keys);
  const int n_keys_max = (int)s->scratch_keys.size();
  const int n_closed_for_keys = std::min(n_closed_view, L.n_closed);
  const int f_hi_for_keys = std::min(s->cfg.keep_first, n_closed_for_keys);
  const int l_lo_for_keys =
      std::max(f_hi_for_keys, n_closed_for_keys - s->cfg.keep_last);
  const int n_fixed_spans =
      (f_hi_for_keys > 0 ? 1 : 0) + (l_lo_for_keys < n_closed_for_keys ? 1 : 0);
  int n_fixed_keys = 0;
  int n_routed_keys = 0;
  for (size_t i = 0; i < spans.size(); ++i) {
    const Span &sp = spans[i];
    const int hi = std::min(sp.start + sp.len - 1, q_last);
    if (hi >= sp.start) {
      if ((int)i < n_fixed_spans)
        n_fixed_keys += hi - sp.start + 1;
      else
        n_routed_keys += hi - sp.start + 1;
    }
  }
  {
    const int *keys0 =
        s->scratch_keys.empty() ? nullptr : s->scratch_keys.data();
    const int **kp = keyp.data();
    const int **kp_end = kp + n_kvh;
    int *nk = nkh.data();
    for (; kp < kp_end; ++kp, ++nk) {
      *kp = keys0;
      *nk = n_keys_max;
    }
  }
  s->last_keys = s->scratch_keys;
  s->last_keys_layer = layer;

  const int n_q_rows = rep * n_q;
  int nqt = 1, nkt = 1;
  pick_qk_tiles(n_kvh, std::max(1, n_q_rows), std::max(1, n_keys_max), dh,
                i8 ? 1 : 4, i8 ? 1 : 2, nthr, &nqt, &nkt);

  int n_h_tiles = std::min(nqt, std::max(1, rep));
  while (n_h_tiles > 1 && nqt % n_h_tiles != 0)
    --n_h_tiles;
  const int n_t_tiles = nqt / n_h_tiles;

  auto q_bounds = [&](int kh, int qt, int *h0, int *h1, int *t0, int *t1) {
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
  const bool use_smt_prefill =
      hga_prefill_smt_enabled() && nqt == 2 && nkt == 10;
  const int nthr_attn =
      use_smt_prefill ? 2 * nthr : std::max(1, std::min(nthr, n_parts));
  const size_t n_pm = (size_t)n_parts * (size_t)max_rows;
  const int max_nk =
      std::max(1, (n_keys_max + std::max(1, nkt) - 1) / std::max(1, nkt));
  grow(s->scratch_part_m, n_pm);
  grow(s->scratch_part_lse, n_pm);
  grow(s->scratch_part_acc, n_pm * (size_t)dh);
  grow(s->scratch_scores, (size_t)n_parts * (size_t)max_rows * (size_t)max_nk);
  float *part_m = s->scratch_part_m.data();
  float *part_lse = s->scratch_part_lse.data();
  float *part_acc = s->scratch_part_acc.data();
  float *part_sc = s->scratch_scores.data();
  const double t_setup = now_ms();

  /* Diagnostic A/B only. Touch every K/V cache line using the same
   * (KV-head, Q-tile, K-tile) worker mapping as the attention kernel. Each Q
   * tile deliberately reloads its copy because L2 is private per core on
   * turing1. The three key classes are contiguous in collect_keys(). */
  double l2_fixed_ms = 0.0, l2_routed_ms = 0.0, l2_active_ms = 0.0;
  if (env_enabled("HGA_PROFILE_L2_LOAD")) {
    const auto touch_key_range = [&](int range_lo, int range_hi) {
      const double tt0 = now_ms();
      uint64_t checksum = 0;
      const int kv_bpe = i8 ? 1 : 2;
      const size_t hstride = (size_t)s->cfg.max_seq * (size_t)dh;
#if defined(_OPENMP)
#pragma omp parallel for collapse(3) num_threads(nthr_attn) schedule(static)   \
    reduction(+ : checksum)
#endif
      for (int kh = 0; kh < n_kvh; ++kh) {
        for (int qt = 0; qt < nqt; ++qt) {
          for (int kt = 0; kt < nkt; ++kt) {
            const int k0 = kt * n_keys_max / nkt;
            const int k1 = (kt + 1) * n_keys_max / nkt;
            const int lo = std::max(k0, range_lo);
            const int hi = std::min(k1, range_hi);
            const uint8_t *kbase =
                i8 ? (const uint8_t *)(L.k8.data() + (size_t)kh * hstride)
                   : (const uint8_t *)(L.k.data() + (size_t)kh * hstride);
            const uint8_t *vbase =
                i8 ? (const uint8_t *)(L.v8.data() + (size_t)kh * hstride)
                   : (const uint8_t *)(L.v.data() + (size_t)kh * hstride);
            for (int x = lo; x < hi; ++x) {
              const int key = s->scratch_keys[(size_t)x];
              const volatile uint8_t *kr =
                  kbase + (size_t)key * (size_t)dh * (size_t)kv_bpe;
              const volatile uint8_t *vr =
                  vbase + (size_t)key * (size_t)dh * (size_t)kv_bpe;
              const int row_bytes = dh * kv_bpe;
              for (int b = 0; b < row_bytes; b += 64)
                checksum += (uint64_t)kr[b] + (uint64_t)vr[b];
              if (i8) {
                const volatile float *ks = L.k_scale.data() +
                                           (size_t)kh * (size_t)s->cfg.max_seq +
                                           key;
                const volatile float *vs = L.v_scale.data() +
                                           (size_t)kh * (size_t)s->cfg.max_seq +
                                           key;
                checksum += (uint64_t)(std::fabs(*ks) * 65536.f) +
                            (uint64_t)(std::fabs(*vs) * 65536.f);
              }
            }
          }
        }
      }
      hga_profile_touch_sink += checksum;
      return now_ms() - tt0;
    };
    l2_fixed_ms = touch_key_range(0, n_fixed_keys);
    l2_routed_ms = touch_key_range(n_fixed_keys, n_fixed_keys + n_routed_keys);
    l2_active_ms = touch_key_range(n_fixed_keys + n_routed_keys, n_keys_max);
  }
  const double t_attn_start = now_ms();

  const auto run_prefill_part = [&](int kh, int qt, int kt) {
    int h0, h1, t0, t1;
    q_bounds(kh, qt, &h0, &h1, &t0, &t1);
    const int n_keys = nkh[(size_t)kh];
    const int *keys = keyp[(size_t)kh];
    const int k0 = n_keys ? kt * n_keys / nkt : 0;
    const int k1 = n_keys ? (kt + 1) * n_keys / nkt : 0;
    if (h1 <= h0 || t1 <= t0)
      return;
    const int part = (kh * nqt + qt) * nkt + kt;
    float *m = part_m + (size_t)part * (size_t)max_rows;
    float *ls = part_lse + (size_t)part * (size_t)max_rows;
    float *ac = part_acc + (size_t)part * (size_t)max_rows * (size_t)dh;
    float *sc = part_sc + (size_t)part * (size_t)max_rows * (size_t)max_nk;
    if (i8) {
      flash_tile_i8(s, L, kh, start_pos, n_q, h0, h1, t0, t1, keys, k0, k1, q8p,
                    qscp, m, ls, ac, sc, nullptr, nullptr, nullptr, nullptr, 0);
    } else {
      flash_tile_f16(s, L, kh, start_pos, n_q, h0, h1, t0, t1, keys, k0, k1,
                     qf_hm, m, ls, ac, sc, nullptr, nullptr, 0);
    }
  };

  if (use_smt_prefill) {
#if defined(_OPENMP)
#pragma omp parallel for collapse(3) num_threads(nthr_attn) proc_bind(close)   \
    schedule(static, 1)
#endif
    for (int kh = 0; kh < n_kvh; ++kh) {
      for (int kt = 0; kt < nkt; ++kt) {
        for (int qt = 0; qt < nqt; ++qt)
          run_prefill_part(kh, qt, kt);
      }
    }
  } else {
#if defined(_OPENMP)
#pragma omp parallel for collapse(3) num_threads(nthr_attn) schedule(static)
#endif
    for (int kh = 0; kh < n_kvh; ++kh) {
      for (int qt = 0; qt < nqt; ++qt) {
        for (int kt = 0; kt < nkt; ++kt)
          run_prefill_part(kh, qt, kt);
      }
    }
  }
  const double t_kernel = now_ms();

  const int n_merge = n_kvh * nqt;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(std::min(nthr, std::max(1, n_merge)))     \
    schedule(static)
#endif
  for (int u = 0; u < n_merge; ++u) {
    const int kh = u / nqt;
    const int qt = u - kh * nqt;
    int h0, h1, t0, t1;
    q_bounds(kh, qt, &h0, &h1, &t0, &t1);
    const int n_rows = (h1 - h0) * (t1 - t0);
    if (n_rows <= 0)
      continue;
    const int base = (kh * nqt + qt) * nkt;
    float *m = part_m + (size_t)base * (size_t)max_rows;
    float *ls = part_lse + (size_t)base * (size_t)max_rows;
    float *ac = part_acc + (size_t)base * (size_t)max_rows * (size_t)dh;
    for (int kt = 1; kt < nkt; ++kt) {
      const int p = base + kt;
      merge_online_softmax(m, ls, ac, part_m + (size_t)p * (size_t)max_rows,
                           part_lse + (size_t)p * (size_t)max_rows,
                           part_acc + (size_t)p * (size_t)max_rows * (size_t)dh,
                           n_rows, dh);
    }
    float *o = ac;
    const float *lsp = ls;
    const size_t dst_tok_step = out_tok ? (size_t)(H * dh) : (size_t)dh;
    const size_t dst_h_step = out_tok ? (size_t)dh : (size_t)n_q * (size_t)dh;
    float *dst_h =
        out_tok ? out + (size_t)t0 * dst_tok_step + (size_t)h0 * (size_t)dh
                : out + ((size_t)h0 * (size_t)n_q + (size_t)t0) * (size_t)dh;
    const int ntok = t1 - t0;
    float *o_end_all = ac + (size_t)n_rows * (size_t)dh;
    for (; o < o_end_all;
         o += (size_t)ntok * (size_t)dh, lsp += ntok, dst_h += dst_h_step) {
      float *dst = dst_h;
      const float *lsp_row = lsp;
      float *o_row = o;
      float *o_end = o + (size_t)ntok * (size_t)dh;
      for (; o_row < o_end; o_row += dh, ++lsp_row, dst += dst_tok_step) {
        if (*lsp_row > 0.f)
          hga_scale_f32(o_row, 1.f / *lsp_row, dh);
        else
          std::memset(o_row, 0, (size_t)dh * sizeof(float));
        std::memcpy(dst, o_row, (size_t)dh * sizeof(float));
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
    int fixed = 0;
    int routed = 0;
    const int pos = q_last;
    for (size_t i = 0; i < spans.size(); ++i) {
      const Span &sp = spans[i];
      const int hi = std::min(sp.start + sp.len - 1, pos);
      if (hi >= sp.start) {
        if ((int)i < n_fixed_spans)
          fixed += hi - sp.start + 1;
        else
          routed += hi - sp.start + 1;
      }
    }
    const int act0 = L.n_closed * s->cfg.chunk_size;
    int active = 0;
    if (L.n_kv > act0)
      active = std::min(L.n_kv - 1, pos) - act0 + 1;
    int att = fixed + routed + active;
    if (att < 1)
      att = 1;
    stats->n_attended_tokens = att;
    stats->n_fixed_tokens = fixed;
    stats->n_routed_tokens = routed;
    stats->n_active_tokens = active;
    stats->sparsity = (L.n_kv > 0) ? (float)att / (float)L.n_kv : 1.f;
    stats->ms_route = t1 - t0;
    stats->ms_attn = t2 - t1;
    stats->ms_q_pool = t_pool - t0;
    stats->ms_route_chunk_scan = route_profile.chunk_scan;
    stats->ms_route_chunk_topk = route_profile.chunk_topk;
    stats->ms_route_group_scan = route_profile.group_scan;
    stats->ms_route_group_topk = route_profile.group_topk;
    const double measured_route =
        route_profile.chunk_scan + route_profile.chunk_topk +
        route_profile.group_scan + route_profile.group_topk;
    stats->ms_route_other =
        std::max(0.0, (t_route1 - t_route0) - measured_route);
    stats->ms_q_quant = t1 - t_spans;
    stats->ms_span_keys = (t_spans - t_route1) + (t_setup - t1);
    stats->ms_attn_kernel = t_kernel - t_attn_start;
    stats->ms_attn_merge = t2 - t_kernel;
    stats->ms_l2_load_fixed = l2_fixed_ms;
    stats->ms_l2_load_routed = l2_routed_ms;
    stats->ms_l2_load_active = l2_active_ms;
    stats->ms_kernel = stats->ms_attn_kernel;
    stats->n_route_mid_chunks = route_profile.n_mid;
    stats->n_route_group_candidates = route_profile.n_group_candidates;
    stats->route_chunk_bytes_unique = route_profile.chunk_bytes_unique;
    stats->route_chunk_bytes_logical = route_profile.chunk_bytes_logical;
    stats->route_group_bytes_unique = route_profile.group_bytes_unique;
    stats->route_group_bytes_logical = route_profile.group_bytes_logical;
  }
  (void)NEG;
}

void hga_attend(hga_session *s, int layer, int start_pos, int n_q,
                const void *q, hga_dtype q_dtype, float *out,
                hga_stats *stats) {
  const int H = s->cfg.n_q_heads;
  const int dh = s->cfg.head_dim;
  const int hs = n_q * dh;
  if (q_dtype == HGA_F32) {
    hga_attend_f32_strided(s, layer, start_pos, n_q, (const float *)q, hs, dh,
                           out, HGA_OUT_HEAD_MAJOR, stats);
    return;
  }
  s->scratch_qf.resize((size_t)H * (size_t)n_q * (size_t)dh);
  const size_t n = (size_t)H * (size_t)n_q * (size_t)dh;
  for (size_t i = 0; i < n; ++i)
    s->scratch_qf[i] = load_f(q, q_dtype, i);
  hga_attend_f32_strided(s, layer, start_pos, n_q, s->scratch_qf.data(), hs, dh,
                         out, HGA_OUT_HEAD_MAJOR, stats);
}

void hga_forward_strided(hga_session *s, int layer, int start_pos, int n_q,
                         const float *q, int q_head_stride, int q_tok_stride,
                         const float *k_rope, int k_head_stride,
                         int k_tok_stride, const float *k_raw,
                         int kr_head_stride, int kr_tok_stride, const float *v,
                         int v_head_stride, int v_tok_stride, float *out,
                         int out_layout, hga_stats *stats) {
  hga_append_f32_strided(s, layer, start_pos, n_q, k_rope, k_head_stride,
                         k_tok_stride, k_raw, kr_head_stride, kr_tok_stride, v,
                         v_head_stride, v_tok_stride);
  hga_attend_f32_strided(s, layer, start_pos, n_q, q, q_head_stride,
                         q_tok_stride, out, out_layout, stats);
  hga_close_full_chunks(s, layer);
}

void hga_route_prefill_only(hga_session *s, int layer, int start_pos, int n_q,
                            const float *q, int q_head_stride, int q_tok_stride,
                            hga_stats *stats) {
  if (!s || !q || layer < 0 || layer >= s->n_layers || start_pos < 0 ||
      n_q <= 0)
    return;
  Layer &L = s->layers[(size_t)layer];
  const int H = s->cfg.n_q_heads;
  const int C = std::max(1, s->cfg.chunk_size);
  int done = 0;
  double route_ms = 0.0;
  int max_chunks = 0;
  int n_groups = 0;
  while (done < n_q) {
    const int seg_start = start_pos + done;
    const int seg_n = std::min(C - seg_start % C, n_q - done);
    const int n_closed = L.n_closed;
    std::vector<PrefillHeadRoute> routes((size_t)H);
    const double tr0 = now_ms();
#if defined(_OPENMP)
#pragma omp parallel for num_threads(std::min(s->cfg.n_threads, H))            \
    schedule(static)
#endif
    for (int h = 0; h < H; ++h)
      route_prefill_head(s, L,
                         q + (size_t)h * (size_t)q_head_stride +
                             (size_t)done * (size_t)q_tok_stride,
                         q_tok_stride, h, seg_n, n_closed, routes[(size_t)h]);
    route_ms += now_ms() - tr0;
    for (const PrefillHeadRoute &hr : routes) {
      max_chunks = std::max(max_chunks, hr.n_chunks);
      n_groups = std::max(n_groups, (int)hr.group_ids.size());
    }
    done += seg_n;
  }
  if (stats) {
    std::memset(stats, 0, sizeof(*stats));
    stats->n_kv = L.n_kv;
    stats->n_closed_chunks = L.n_closed;
    stats->n_selected_chunks = max_chunks;
    stats->n_opened_groups = n_groups;
    stats->ms_route = route_ms;
  }
}

void hga_route_decode_only(hga_session *s, int layer, int start_pos,
                           const float *q, int q_head_stride,
                           hga_stats *stats) {
  if (!s || !q || layer < 0 || layer >= s->n_layers || start_pos < 0)
    return;
  Layer &L = s->layers[(size_t)layer];
  const int H = s->cfg.n_q_heads;
  const int dh = s->cfg.head_dim;
  const double t0 = now_ms();
  grow(s->scratch_q_pool, (size_t)H * (size_t)dh);
  float *q_pool = s->scratch_q_pool.data();
  {
    const float *qh = q;
    float *dst = q_pool;
    float *dst_end = q_pool + (size_t)H * (size_t)dh;
    for (; dst < dst_end; dst += dh, qh += q_head_stride)
      std::memcpy(dst, qh, (size_t)dh * sizeof(float));
  }
  const int n_closed_view = std::min(L.n_closed, start_pos / s->cfg.chunk_size);
  RouteSet rs;
  int n_sel = 0, n_open = 0;
  route_layer(s, L, q_pool, n_closed_view, 1, rs, &n_sel, &n_open);
  std::vector<Span> spans;
  collect_spans(s, L, rs, n_closed_view, spans);
  const double t1 = now_ms();
  if (stats) {
    std::memset(stats, 0, sizeof(*stats));
    stats->n_kv = L.n_kv;
    stats->n_closed_chunks = L.n_closed;
    stats->n_selected_chunks = n_sel;
    stats->n_opened_groups = n_open;
    stats->ms_route = t1 - t0;
  }
}

void hga_forward(hga_session *s, int layer, int start_pos, int n_q,
                 const void *q, const void *k_rope, const void *k_raw,
                 const void *v, hga_dtype dtype, float *out, hga_stats *stats) {
  hga_append(s, layer, start_pos, n_q, k_rope, k_raw, v, dtype);
  hga_attend(s, layer, start_pos, n_q, q, dtype, out, stats);
  hga_close_full_chunks(s, layer);
}

int hga_forward_fixed(hga_session *s, int layer, int start_pos,
                      const hga_fixed_batch *batch, const void *q,
                      const void *k_rope, const void *k_raw, const void *v,
                      hga_dtype dtype, float *out, hga_stats *stats) {
  if (!s || !q || !k_rope || !v || !out || layer < 0 || layer >= s->n_layers ||
      !hga_fixed_batch_validate(batch))
    return 0;

  const int H = s->cfg.n_q_heads;
  const int KVH = s->cfg.n_kv_heads;
  const int D = s->cfg.head_dim;
  const int physical = (int)batch->physical_count;
  const int real = (int)batch->real_count;
  const size_t out_count = (size_t)H * (size_t)physical * (size_t)D;
  std::memset(out, 0, out_count * sizeof(float));
  s->scratch_fixed_out.resize((size_t)H * (size_t)real * (size_t)D);

  if (dtype == HGA_F32) {
    hga_forward_strided(s, layer, start_pos, real, (const float *)q,
                        physical * D, D, (const float *)k_rope, physical * D, D,
                        (const float *)k_raw, physical * D, D, (const float *)v,
                        physical * D, D, s->scratch_fixed_out.data(),
                        HGA_OUT_HEAD_MAJOR, stats);
  } else {
    const size_t q_count = (size_t)H * (size_t)real * (size_t)D;
    const size_t kv_count = (size_t)KVH * (size_t)real * (size_t)D;
    s->scratch_fixed_input.resize(q_count + kv_count * 3);
    float *qf = s->scratch_fixed_input.data();
    float *kf = qf + q_count;
    float *krf = kf + kv_count;
    float *vf = krf + kv_count;
    /* F16 is the only non-F32 public input.  Advance pointers instead of
     * rebuilding head/token offsets in this narrow conversion loop. */
    const uint16_t *qsrc = (const uint16_t *)q;
    float *qdst = qf;
    const ptrdiff_t q_skip = (ptrdiff_t)(physical - real) * D;
    for (int h = 0; h < H; ++h) {
      for (int t = 0; t < real; ++t)
        for (int d = 0; d < D; ++d)
          *qdst++ = f16_to_f32(*qsrc++);
      if (h + 1 < H)
        qsrc += q_skip;
    }

    const uint16_t *ksrc = (const uint16_t *)k_rope;
    const uint16_t *krsrc = k_raw ? (const uint16_t *)k_raw : nullptr;
    const uint16_t *vsrc = (const uint16_t *)v;
    const ptrdiff_t kv_skip = (ptrdiff_t)(physical - real) * D;
    for (int h = 0; h < KVH; ++h) {
      for (int t = 0; t < real; ++t) {
        for (int d = 0; d < D; ++d) {
          *kf++ = f16_to_f32(*ksrc++);
          *krf++ = krsrc ? f16_to_f32(*krsrc++) : kf[-1];
          *vf++ = f16_to_f32(*vsrc++);
        }
      }
      if (h + 1 < KVH) {
        ksrc += kv_skip;
        vsrc += kv_skip;
        if (krsrc)
          krsrc += kv_skip;
      }
    }
    hga_forward_strided(s, layer, start_pos, real, qf, real * D, D, kf,
                        real * D, D, krf, real * D, D, vf, real * D, D,
                        s->scratch_fixed_out.data(), HGA_OUT_HEAD_MAJOR, stats);
  }

  const float *src = s->scratch_fixed_out.data();
  float *dst = out;
  const size_t real_bytes = (size_t)real * (size_t)D * sizeof(float);
  const size_t physical_step = (size_t)physical * (size_t)D;
  const size_t real_step = (size_t)real * (size_t)D;
  for (int h = 0; h < H; ++h, src += real_step, dst += physical_step)
    std::memcpy(dst, src, real_bytes);
  return 1;
}

void hga_last_keys(const hga_session *s, const int **keys, int *n_keys) {
  if (!s || !keys || !n_keys)
    return;
  *keys = s->last_keys.empty() ? nullptr : s->last_keys.data();
  *n_keys = (int)s->last_keys.size();
}

int hga_last_keys_layer(const hga_session *s) {
  return s ? s->last_keys_layer : -1;
}

void hga_session_set_l2(hga_session *s, void *plan) {
  if (s)
    s->l2 = plan;
}

void *hga_session_l2(const hga_session *s) { return s ? s->l2 : nullptr; }

int hga_window_keys(const hga_session *s, int layer, int q_hi, int *keys,
                    int cap) {
  if (!s || !keys || cap <= 0 || layer < 0 || layer >= s->n_layers)
    return 0;
  const Layer &L = s->layers[(size_t)layer];
  const int C = s->cfg.chunk_size;
  const int n_closed = L.n_closed;
  const int f_hi = std::min(s->cfg.keep_first, n_closed);
  const int l_lo = std::max(f_hi, n_closed - s->cfg.keep_last);
  int n = 0;
  auto push = [&](int j) {
    if (j < 0 || j > q_hi || j >= L.n_kv)
      return;
    if (n < cap)
      keys[n++] = j;
  };
  if (f_hi > 0) {
    const int hi = std::min(f_hi * C, q_hi + 1);
    for (int j = 0; j < hi; ++j)
      push(j);
  }
  if (l_lo < n_closed) {
    const int lo = l_lo * C;
    const int hi = std::min((n_closed - l_lo) * C + lo, q_hi + 1);
    for (int j = lo; j < hi; ++j)
      push(j);
  }
  const int act0 = n_closed * C;
  for (int j = act0; j < L.n_kv && j <= q_hi; ++j)
    push(j);
  return n;
}

int hga_prefetch_keys(const hga_session *s, int layer, size_t budget_bytes,
                      int *keys, int cap) {
  if (!s || !keys || cap <= 0 || budget_bytes == 0 || layer < 0 ||
      layer >= s->n_layers)
    return 0;
  const Layer &L = s->layers[(size_t)layer];
  const int n_kv = std::min(L.n_kv, s->cfg.max_seq);
  if (n_kv <= 0)
    return 0;

  const size_t elem_bytes = s->cfg.prec == HGA_PREC_I8 ? 1u : 2u;
  size_t bytes_per_token =
      2u * (size_t)s->cfg.n_kv_heads * (size_t)s->cfg.head_dim * elem_bytes;
  if (s->cfg.prec == HGA_PREC_I8)
    bytes_per_token += 2u * (size_t)s->cfg.n_kv_heads * sizeof(float);
  const int n_closed = std::min(L.n_closed, n_kv / s->cfg.chunk_size);
  const size_t summary_bytes = (size_t)n_closed * (size_t)s->cfg.n_kv_heads *
                               (size_t)s->cfg.head_dim * sizeof(float) *
                               (size_t)(1 + s->groups_per_chunk);
  const size_t kv_budget =
      budget_bytes > summary_bytes ? budget_bytes - summary_bytes : 0;
  int budget_tokens =
      (int)std::min<size_t>((size_t)n_kv, kv_budget / bytes_per_token);
  const int sink_tokens = std::min(n_kv, s->cfg.keep_first * s->cfg.chunk_size);
  /* The two sink chunks are an invariant even for an accidentally tiny
   * experimental budget. Normal L3 budgets are much larger than this. */
  budget_tokens = std::min(n_kv, std::max(budget_tokens, sink_tokens));
  budget_tokens = std::min(budget_tokens, cap);

  int n = 0;
  if (budget_tokens >= n_kv) {
    for (int j = 0; j < n_kv && n < cap; ++j)
      keys[n++] = j;
    return n;
  }
  const int n_sink = std::min(sink_tokens, budget_tokens);
  for (int j = 0; j < n_sink && n < cap; ++j)
    keys[n++] = j;
  const int n_recent = budget_tokens - n_sink;
  const int recent_begin = std::max(n_sink, n_kv - n_recent);
  for (int j = recent_begin; j < n_kv && n < cap; ++j)
    keys[n++] = j;
  return n;
}

static void hga_touch_bytes(const void *p, size_t n) {
  if (!p || n == 0)
    return;
  const uint8_t *b = (const uint8_t *)p;
  volatile uint64_t acc = 0;
  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    acc += *(const uint64_t *)(b + i);
  }
  for (; i + 8 <= n; i += 8) {
    acc += *(const uint64_t *)(b + i);
  }
  (void)acc;
}

void hga_touch_kv_tile(const hga_session *s, int layer, const int *keys,
                       int n_keys, int tid, int n_threads) {
  if (!s || !keys || n_keys <= 0 || layer < 0 || layer >= s->n_layers)
    return;
  const Layer &L = s->layers[(size_t)layer];
  const int n_kvh = s->cfg.n_kv_heads;
  const int dh = s->cfg.head_dim;
  const int ms = s->cfg.max_seq;
  const int rep = s->cfg.n_q_heads / std::max(1, n_kvh);
  const int n_q_rows = std::max(1, rep); /* decode: n_q = 1 */
  const bool i8 = s->cfg.prec == HGA_PREC_I8;
  int nqt = 1, nkt = 1;
  pick_qk_tiles(n_kvh, n_q_rows, std::max(1, n_keys), dh, i8 ? 1 : 4,
                i8 ? 1 : 2, std::max(1, n_threads), &nqt, &nkt);
  const int n_tasks = n_kvh * nqt * nkt;
  if (tid < 0 || tid >= n_threads)
    return;
  const size_t vec = (size_t)dh * (i8 ? 1u : 2u);
  /* A small L3 team may have fewer workers than kernel tiles. Each worker
   * therefore walks a strided set of tiles; the former one-tile-per-worker
   * mapping silently touched only half the KV heads with two workers. */
  for (int task = tid; task < n_tasks; task += n_threads) {
    const int kh = task / (nqt * nkt);
    const int rem = task - kh * nqt * nkt;
    const int kt = rem % nkt;
    const int k0 = kt * n_keys / nkt;
    const int k1 = (kt + 1) * n_keys / nkt;
    for (int ki = k0; ki < k1; ++ki) {
      const int j = keys[ki];
      if (j < 0 || j >= L.n_kv || j >= ms)
        continue;
      const size_t idx = kv_index(kh, j, dh, ms);
      if (i8) {
        if (!L.k8.empty())
          hga_touch_bytes(L.k8.data() + idx, vec);
        if (!L.v8.empty())
          hga_touch_bytes(L.v8.data() + idx, vec);
        if (!L.k_scale.empty())
          hga_touch_bytes(&L.k_scale[(size_t)kh * (size_t)ms + (size_t)j],
                          sizeof(float));
        if (!L.v_scale.empty())
          hga_touch_bytes(&L.v_scale[(size_t)kh * (size_t)ms + (size_t)j],
                          sizeof(float));
      } else {
        if (!L.k.empty())
          hga_touch_bytes(L.k.data() + idx, vec);
        if (!L.v.empty())
          hga_touch_bytes(L.v.data() + idx, vec);
      }
    }
  }
}

void hga_touch_summary_tile(const hga_session *s, int layer, int tid,
                            int n_threads) {
  if (!s || layer < 0 || layer >= s->n_layers || tid < 0 || n_threads <= 0)
    return;
  const Layer &L = s->layers[(size_t)layer];
  const int n_kvh = s->cfg.n_kv_heads;
  const int dh = s->cfg.head_dim;
  const int G = s->groups_per_chunk;
  const int n_closed = L.n_closed;
  const int h0 = tid * n_kvh / n_threads;
  const int h1 = (tid + 1) * n_kvh / n_threads;
  for (int kh = h0; kh < h1; ++kh) {
    const float *chunk =
        L.chunk_k.data() + (size_t)kh * (size_t)s->max_chunks * (size_t)dh;
    hga_touch_bytes(chunk, (size_t)n_closed * (size_t)dh * sizeof(float));
    const float *group = L.group_k.data() + (size_t)kh * (size_t)s->max_chunks *
                                                (size_t)G * (size_t)dh;
    hga_touch_bytes(group,
                    (size_t)n_closed * (size_t)G * (size_t)dh * sizeof(float));
  }
}
