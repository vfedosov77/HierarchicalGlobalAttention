/* Isolate production HGA routing. Fake Q/K/V, no H2D, no QKV/out matmuls.
 *
 * Prefill uses `hga_route_prefill_only` (same `route_prefill_head` loop as
 * GPU-prefill staging). Generate uses `hga_route_decode_only` (`route_layer`
 * as in decode). 65 calls ≈ one pass over a 64-layer transformer.
 *
 *   build/hga-route-bench [--threads 12] [--ctx 8192] [--ubatch 768] [--reps 65]
 */
#include "hga.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

static float urand(uint32_t &s) {
  s = s * 1664525u + 1013904223u;
  return (float)((s >> 8) & 0xffffff) / 16777216.f * 2.f - 1.f;
}

static double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch())
      .count();
}

static int env_int(const char *name, int fallback) {
  const char *e = std::getenv(name);
  if (!e || !e[0])
    return fallback;
  return std::atoi(e);
}

static void fill_cache(hga_session *s, int layer, int seq, uint32_t seed) {
  const hga_config *c = hga_session_config(s);
  const int KVH = c->n_kv_heads;
  const int D = c->head_dim;
  const int C = c->chunk_size;
  std::vector<float> k((size_t)KVH * (size_t)C * (size_t)D);
  std::vector<float> v = k;
  for (float &x : k)
    x = urand(seed);
  for (float &x : v)
    x = urand(seed);
  for (int p = 0; p < seq; p += C) {
    const int n = std::min(C, seq - p);
    hga_append(s, layer, p, n, k.data(), k.data(), v.data(), HGA_F32);
    hga_close_full_chunks(s, layer);
  }
}

int main(int argc, char **argv) {
  int n_threads = env_int("HGA_THREADS", 12);
  int ctx = 8192;
  int ubatch = 768;
  int reps = 65;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--threads" && i + 1 < argc)
      n_threads = std::atoi(argv[++i]);
    else if (a == "--ctx" && i + 1 < argc)
      ctx = std::atoi(argv[++i]);
    else if (a == "--ubatch" && i + 1 < argc)
      ubatch = std::atoi(argv[++i]);
    else if (a == "--reps" && i + 1 < argc)
      reps = std::atoi(argv[++i]);
    else if (a == "-h" || a == "--help") {
      std::printf("hga-route-bench [--threads N] [--ctx N] [--ubatch N] [--reps N]\n");
      return 0;
    }
  }
  if (n_threads < 1)
    n_threads = 1;
  if (ctx < 256)
    ctx = 256;
  if (ubatch < 64)
    ubatch = 64;
  if (ubatch > ctx)
    ubatch = ctx;
  if (reps < 1)
    reps = 1;

  hga_config cfg = hga_config_qwen38_27b(2, ctx + 64, n_threads);
  hga_session *s = hga_session_create(&cfg, 1);
  const int H = cfg.n_q_heads;
  const int D = cfg.head_dim;
  const int start = ctx - ubatch;

  std::printf("# HGA route-only  (no H2D, no QKV/out, no attention kernel)\n");
  std::printf("# shape Qwen3.8-27B  H=%d KVH=%d dh=%d  ctx=%d  prefill_ubatch=%d\n",
              H, cfg.n_kv_heads, D, ctx, ubatch);
  std::printf("# threads=%d  reps=%d  layer cache filled to %d tokens then frozen\n",
              n_threads, reps, ctx);
#if defined(_OPENMP)
  std::printf("# omp_max_threads=%d\n", omp_get_max_threads());
#endif

  fill_cache(s, 0, ctx, 7);

  /* ggml PREFILL Q layout: [dh, n_head, n_q] contiguous. */
  std::vector<float> q_prefill((size_t)ubatch * (size_t)H * (size_t)D);
  std::vector<float> q_decode((size_t)H * (size_t)D);
  uint32_t seed = 11;
  for (float &x : q_prefill)
    x = urand(seed);
  for (float &x : q_decode)
    x = urand(seed);
  const int q_head_stride = D;
  const int q_tok_stride = D * H;

  hga_stats st{};
  hga_route_prefill_only(s, 0, start, ubatch, q_prefill.data(), q_head_stride,
                         q_tok_stride, &st);
  hga_route_decode_only(s, 0, ctx - 1, q_decode.data(), D, &st);

  volatile int sink = 0;
  const double t0 = now_ms();
  for (int i = 0; i < reps; ++i) {
    hga_route_prefill_only(s, 0, start, ubatch, q_prefill.data(), q_head_stride,
                           q_tok_stride, &st);
    sink += st.n_selected_chunks + st.n_opened_groups;
  }
  const double t1 = now_ms();
  const double prefill_total = t1 - t0;
  const double prefill_call = prefill_total / (double)reps;

  const double t2 = now_ms();
  for (int i = 0; i < reps; ++i) {
    hga_route_decode_only(s, 0, ctx - 1, q_decode.data(), D, &st);
    sink += st.n_selected_chunks + st.n_opened_groups;
  }
  const double t3 = now_ms();
  const double decode_total = t3 - t2;
  const double decode_call = decode_total / (double)reps;

  std::printf("\n");
  std::printf("prefill  n_q=%-4d  one layer  %7.2f ms/call   %d calls %7.1f ms\n",
              ubatch, prefill_call, reps, prefill_total);
  std::printf("         x16 HGA attn layers     %7.1f ms   (one transformer prefill step)\n",
              prefill_call * 16.0);
  std::printf("         x%d loop (this run)     %7.1f ms\n", reps, prefill_total);
  std::printf("generate n_q=1     one layer  %7.3f ms/call   %d calls %7.2f ms\n",
              decode_call, reps, decode_total);
  std::printf("         x16 HGA attn layers     %7.2f ms   (one transformer decode step)\n",
              decode_call * 16.0);
  std::printf("         x%d loop (this run)     %7.2f ms\n", reps, decode_total);
  std::printf("last stats  kv=%d closed=%d chunks=%d groups=%d  sink=%d\n",
              st.n_kv, st.n_closed_chunks, st.n_selected_chunks,
              st.n_opened_groups, (int)sink);
  std::printf("HGA_ROUTE_MEASURE prefill_ms_per_layer=%.3f prefill_ms_x16=%.1f "
              "prefill_ms_%d=%.1f generate_ms_per_layer=%.4f generate_ms_x16=%.2f "
              "generate_ms_%d=%.2f ctx=%d ubatch=%d threads=%d\n",
              prefill_call, prefill_call * 16.0, reps, prefill_total, decode_call,
              decode_call * 16.0, reps, decode_total, ctx, ubatch, n_threads);

  hga_session_free(s);
  return 0;
}
