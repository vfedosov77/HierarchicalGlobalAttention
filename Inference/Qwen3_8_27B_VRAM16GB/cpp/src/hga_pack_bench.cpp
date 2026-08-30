/* Isolate production PREFILL KV append and INT8 quantization.
 *
 * Production appends a 768-token ubatch as twelve 64-token regions. Repeating
 * that exact call pattern measures the persistent packing pool without route
 * scoring, attention, graph construction, or PCIe traffic.
 *
 *   build/hga-pack-bench --route-threads 24 --pack-threads 12 --reps 20
 */
#include "hga.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static float urand(uint32_t &s) {
  s = s * 1664525u + 1013904223u;
  return (float)((s >> 8) & 0xffffff) / 16777216.f * 2.f - 1.f;
}

static double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch())
      .count();
}

static int env_int(const char *name, int fallback) {
  const char *value = std::getenv(name);
  return value && value[0] ? std::atoi(value) : fallback;
}

int main(int argc, char **argv) {
  int route_threads = env_int("HGA_THREADS", 12);
  int pack_threads = env_int("HGA_PACK_THREADS", route_threads);
  int ubatch = 768;
  int chunk = 64;
  int reps = 20;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--route-threads" && i + 1 < argc)
      route_threads = std::atoi(argv[++i]);
    else if (arg == "--pack-threads" && i + 1 < argc)
      pack_threads = std::atoi(argv[++i]);
    else if (arg == "--ubatch" && i + 1 < argc)
      ubatch = std::atoi(argv[++i]);
    else if (arg == "--chunk" && i + 1 < argc)
      chunk = std::atoi(argv[++i]);
    else if (arg == "--reps" && i + 1 < argc)
      reps = std::atoi(argv[++i]);
    else if (arg == "-h" || arg == "--help") {
      std::printf("hga-pack-bench [--route-threads N] [--pack-threads N] "
                  "[--ubatch N] [--chunk N] [--reps N]\n");
      return 0;
    }
  }
  route_threads = std::max(1, route_threads);
  pack_threads = std::max(1, pack_threads);
  chunk = std::max(1, chunk);
  ubatch = std::max(chunk, ubatch);
  reps = std::max(1, reps);

  hga_config cfg =
      hga_config_qwen38_27b(2, std::max(8192, ubatch + chunk), route_threads);
  cfg.n_pack_threads = pack_threads;
  cfg.prec = HGA_PREC_I8;
  hga_session *session = hga_session_create(&cfg, 1);

  const int KVH = cfg.n_kv_heads;
  const int D = cfg.head_dim;
  std::vector<float> k((size_t)KVH * (size_t)chunk * (size_t)D);
  std::vector<float> raw(k.size());
  std::vector<float> v(k.size());
  uint32_t seed = 17;
  for (float &x : k)
    x = urand(seed);
  for (float &x : raw)
    x = urand(seed);
  for (float &x : v)
    x = urand(seed);

  auto append_ubatch = [&] {
    for (int pos = 0; pos < ubatch; pos += chunk) {
      const int n = std::min(chunk, ubatch - pos);
      hga_append(session, 0, pos, n, k.data(), raw.data(), v.data(), HGA_F32);
    }
  };
  for (int i = 0; i < 3; ++i)
    append_ubatch();

  const double t0 = now_ms();
  for (int i = 0; i < reps; ++i)
    append_ubatch();
  const double elapsed = now_ms() - t0;
  const double per_ubatch = elapsed / (double)reps;
  const double per_layer = per_ubatch;
  std::printf("# HGA packing-only: %d x %d-token append regions, Qwen KVH=%d dh=%d\n",
              (ubatch + chunk - 1) / chunk, chunk, KVH, D);
  std::printf("# route_threads=%d pack_threads=%d reps=%d\n", route_threads,
              pack_threads, reps);
  std::printf("HGA_PACK_MEASURE append_ms_per_ubatch=%.3f "
              "append_ms_x16=%.1f elapsed_ms=%.1f ubatch=%d chunk=%d "
              "route_threads=%d pack_threads=%d\n",
              per_ubatch, per_layer * 16.0, elapsed, ubatch, chunk,
              route_threads, pack_threads);

  hga_session_free(session);
  return 0;
}
