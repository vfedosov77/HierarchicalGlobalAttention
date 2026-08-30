#include "llama-hga.h"

#include "hga.h"
#include "hga_l2.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-hparams.h"
#include "llama-kv-cache.h"
#include "llama.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

/* llama_cparams is extended by apply_hga.py with:
 *   bool hga_enabled; int hga_levels; int hga_chunk_size; int hga_group_size;
 *   int hga_keep_first; int hga_keep_last; float hga_frac_l1; float
 * hga_frac_l2; bool hga_i8; void * hga_runtime;
 */

static int hga_layer_index(const llama_cparams &cparams,
                           const llama_hparams &hparams, int il) {
  /* The MTP draft context owns one full-attention block (blk.64). Its HGA
   * session is deliberately separate from the target session, so index it 0
   * instead of after the target's 16 full-attention layers. */
  if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
    return 0;
  }
  int k = 0;
  for (int i = 0; i < il; ++i) {
    if (!hparams.is_recr(i))
      ++k;
  }
  return k;
}

static int hga_n_full_layers(const llama_hparams &hparams) {
  int n = 0;
  for (int i = 0; i < (int)hparams.n_layer(); ++i) {
    if (!hparams.is_recr(i))
      ++n;
  }
  return n;
}

/* The production server/spec runner owns one target and one MTP context.
 * Their HGA calls are serialized, so the target's persistent worker pool can
 * warm either session without allocating another CPU team. */
static hga_session *g_hga_target_session = nullptr;
static hga_session *g_hga_mtp_session = nullptr;

struct hga_prefill_cpu_tot {
  double chunk_ms = 0.0;
  double route_ms = 0.0;
  double pack_ms = 0.0;
  double append_ms = 0.0;
  double close_ms = 0.0;
  double union_ms = 0.0;
  double scale_clear_ms = 0.0;
  double kv_copy_ms = 0.0;
  double pack_other_ms = 0.0;
  int calls = 0;
  bool dumped = false;
};

static hga_prefill_cpu_tot g_prefill_cpu_tot;

static void hga_prefill_cpu_dump_total() {
  if (g_prefill_cpu_tot.dumped || g_prefill_cpu_tot.calls <= 0) {
    return;
  }
  g_prefill_cpu_tot.dumped = true;
  const double inv =
      g_prefill_cpu_tot.chunk_ms > 0.0 ? 100.0 / g_prefill_cpu_tot.chunk_ms : 0.0;
  std::fprintf(stderr,
               "hga-prof prefill TOTAL cpu_hga=%.1f ms layers=%d  "
               "route=%.1f (%.0f%%) pack=%.1f (%.0f%%)  "
               "append=%.1f close=%.1f union=%.1f scale-clear=%.1f "
               "kv-copy=%.1f other=%.1f  "
               "(host compute; D2H QKV / H2D KV are separate xfer lines)\n",
               g_prefill_cpu_tot.chunk_ms, g_prefill_cpu_tot.calls,
               g_prefill_cpu_tot.route_ms, g_prefill_cpu_tot.route_ms * inv,
               g_prefill_cpu_tot.pack_ms, g_prefill_cpu_tot.pack_ms * inv,
               g_prefill_cpu_tot.append_ms, g_prefill_cpu_tot.close_ms,
               g_prefill_cpu_tot.union_ms, g_prefill_cpu_tot.scale_clear_ms,
               g_prefill_cpu_tot.kv_copy_ms, g_prefill_cpu_tot.pack_other_ms);
  const char *bottleneck =
      g_prefill_cpu_tot.kv_copy_ms >= g_prefill_cpu_tot.route_ms &&
              g_prefill_cpu_tot.kv_copy_ms >= g_prefill_cpu_tot.union_ms
          ? "cpu_kv_copy"
      : g_prefill_cpu_tot.route_ms >= g_prefill_cpu_tot.union_ms ? "cpu_route"
                                                                : "cpu_union";
  std::fprintf(stderr, "hga-prof prefill BOTTLENECK %s (of CPU HGA staging)\n",
               bottleneck);
}

static bool hga_l3_prefetch_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HGA_L3_PREFETCH");
    return v && v[0] && std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

static bool hga_gpu_prefill_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HGA_GPU_PREFILL");
    return v && v[0] && std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

static int hga_gpu_prefill_max_keys() {
  static const int value = [] {
    const char *v = std::getenv("HGA_GPU_PREFILL_MAX_KEYS");
    if (!v || !v[0])
      return 12288;
    const long n = std::strtol(v, nullptr, 10);
    return (int)std::max(512L, std::min(65536L, n));
  }();
  return value;
}

static int hga_gpu_prefill_min_keys() {
  static const int value = [] {
    const char *v = std::getenv("HGA_GPU_PREFILL_MIN_KEYS");
    if (!v || !v[0])
      return 0;
    const long n = std::strtol(v, nullptr, 10);
    return (int)std::max(0L, std::min(65536L, n));
  }();
  return value;
}

static bool hga_gpu_verify_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HGA_GPU_VERIFY");
    return !v || !v[0] || std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

static int hga_gpu_verify_max_keys() {
  static const int value = [] {
    const char *v = std::getenv("HGA_GPU_VERIFY_MAX_KEYS");
    if (!v || !v[0])
      return hga_gpu_prefill_max_keys();
    const long n = std::strtol(v, nullptr, 10);
    return (int)std::max(256L, std::min(65536L, n));
  }();
  return value;
}

/* Experimental activation-wire format. The model graph deliberately remains
 * F32 where ggml CUDA kernels require it (quantized MUL_MAT destinations,
 * norms, recurrent ops, flash-attention Q/output and softmax). Enabling this
 * switch casts only tensors that cross the GPU/CPU HGA boundary to F16, then
 * restores F32 on the receiving device. HGA route IDs and the persistent I8
 * KV cache are unchanged. */
static bool hga_f16_transport_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HGA_F16_TRANSPORT");
    return v && v[0] && std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

/* Experimental PREFILL K/V wire: CUDA quantizes each 32-value block to
 * native Q8_0 before D2H. Q and partial-RoPE Kraw remain F16 because routing
 * consumes them at higher precision. */
static bool hga_gpu_kv_i8_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HGA_GPU_KV_I8");
    return v && v[0] && std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

ggml_backend_t hga_sched_gpu_backend(ggml_backend_sched_t sched) {
  if (!sched) {
    return nullptr;
  }
  const int n = ggml_backend_sched_get_n_backends(sched);
  for (int i = 0; i < n; ++i) {
    ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
    ggml_backend_dev_t dev = b ? ggml_backend_get_device(b) : nullptr;
    if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
      return b;
    }
  }
  return nullptr;
}

void hga_pin_gpu(ggml_backend_sched_t sched, ggml_tensor *t) {
  if (!sched || !t) {
    return;
  }
  if (ggml_backend_t gpu = hga_sched_gpu_backend(sched)) {
    ggml_backend_sched_set_tensor_backend(sched, t, gpu);
  }
}

void hga_pin_gpu_pack(ggml_backend_sched_t sched, ggml_tensor *t,
                      int32_t phase) {
  if (hga_gpu_pack(phase)) {
    hga_pin_gpu(sched, t);
  }
}

void hga_pin_gpu_prefill(ggml_backend_sched_t sched, ggml_tensor *t,
                         int32_t phase) {
  if (phase == HGA_SWAP_PREFILL) {
    hga_pin_gpu(sched, t);
  }
}

void hga_pin_gpu_decode(ggml_backend_sched_t sched, ggml_tensor *t,
                        int32_t phase) {
  if (hga_decode_pack(phase)) {
    hga_pin_gpu(sched, t);
  }
}

void hga_pin_gpu_prefill_probe(ggml_backend_sched_t sched, ggml_tensor *t,
                               int32_t phase) {
  /* HGA's CPU routing/staging op is deliberately custom and has no CUDA
   * implementation. Its descendants are named and pinned separately. */
  if (phase != HGA_SWAP_PREFILL || !t || t->op == GGML_OP_CUSTOM) {
    return;
  }
  const char *enabled = std::getenv("HGA_PREFILL_PIN_ALL");
  if (enabled && enabled[0] && enabled[0] != '0') {
    hga_pin_gpu(sched, t);
  }
}

ggml_tensor *hga_copy_to_gpu(llm_graph_context *gctx, ggml_tensor *src,
                             const char *name) {
  if (!gctx || !src) {
    return src;
  }
  ggml_backend_t gpu = hga_sched_gpu_backend(gctx->sched);
  if (!gpu) {
    return src;
  }
  if (ggml_backend_t assigned =
          ggml_backend_sched_get_tensor_backend(gctx->sched, src)) {
    ggml_backend_dev_t dev = ggml_backend_get_device(assigned);
    if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
      return src;
    }
  }
  const bool wire_f16 = hga_f16_transport_enabled() &&
                        src->type == GGML_TYPE_F16;
  static bool logged_f32 = false;
  static bool logged_f16 = false;
  bool &logged = wire_f16 ? logged_f16 : logged_f32;
  if (!logged) {
    logged = true;
    fprintf(
        stderr,
        "hga: H2D attn → CUDA  wire=%s ne=[%lld,%lld]  %.1f KiB  "
        "(o_proj input restored to F32 on GPU)\n",
        wire_f16 ? "f16" : "f32",
        (long long)src->ne[0], (long long)src->ne[1],
        ggml_nbytes(src) / 1024.0);
  }
  /* 2D so wo MUL_MAT sees a normal [n_embd, n_tok] activation, not the
   * 4D custom-op layout [n_embd, n_tok, 1, 1]. */
  ggml_tensor *dst =
      ggml_new_tensor_2d(gctx->ctx0, src->type, src->ne[0], src->ne[1]);
  if (name && name[0]) {
    ggml_set_name(dst, name);
  }
  ggml_backend_sched_set_tensor_backend(gctx->sched, dst, gpu);
  ggml_tensor *cpy = ggml_cpy(gctx->ctx0, src, dst);
  ggml_backend_sched_set_tensor_backend(gctx->sched, cpy, gpu);
  ggml_build_forward_expand(gctx->gf, cpy);
  if (!wire_f16) {
    return cpy;
  }
  /* Keep the optimized quantized CUDA MUL_MAT path, whose activation and
   * destination contract is F32. Only the PCIe payload is F16. */
  ggml_tensor *restored = ggml_cast(gctx->ctx0, cpy, GGML_TYPE_F32);
  if (name && name[0]) {
    ggml_format_name(restored, "%s_f16_to_f32", name);
  }
  ggml_backend_sched_set_tensor_backend(gctx->sched, restored, gpu);
  ggml_build_forward_expand(gctx->gf, restored);
  return restored;
}

ggml_tensor *hga_copy_to_cpu(llm_graph_context *gctx, ggml_tensor *src,
                             const char *name) {
  if (!gctx || !src || !gctx->backend_cpu) {
    return src;
  }
  ggml_tensor *wire = src;
  if (hga_f16_transport_enabled() &&
      gctx->cparams.hga_phase == HGA_SWAP_PREFILL &&
      src->type == GGML_TYPE_F32 &&
      hga_sched_gpu_backend(gctx->sched)) {
    wire = ggml_cast(gctx->ctx0, src, GGML_TYPE_F16);
    if (name && name[0]) {
      ggml_format_name(wire, "%s_f32_to_f16", name);
    }
    hga_pin_gpu(gctx->sched, wire);
  }
  /* Dense 2D D2H. 3D strided views of Q+gate segfaulted on CUDA sm_70. */
  const int64_t n0 = wire->ne[0] * (wire->ne[1] > 0 ? wire->ne[1] : 1);
  const int64_t n1 = wire->ne[2] > 0 ? wire->ne[2] : 1;
  static int nlog = 0;
  if (nlog < 8) {
    nlog++;
    fprintf(stderr,
            "hga: D2H %s → CPU  wire=%s ne=[%lld,%lld,%lld]  "
            "2d=[%lld,%lld]  %.1f KiB\n",
            name && name[0] ? name : "act", ggml_type_name(wire->type),
            (long long)src->ne[0],
            (long long)src->ne[1], (long long)src->ne[2],
            (long long)n0, (long long)n1, ggml_nbytes(wire) / 1024.0);
  }
  ggml_tensor *src2d = ggml_reshape_2d(gctx->ctx0, wire, n0, n1);
  ggml_tensor *dst2d =
      ggml_new_tensor_2d(gctx->ctx0, wire->type, n0, n1);
  if (name && name[0]) {
    ggml_set_name(dst2d, name);
  }
  ggml_backend_sched_set_tensor_backend(gctx->sched, dst2d, gctx->backend_cpu);
  ggml_tensor *cpy = ggml_cpy(gctx->ctx0, src2d, dst2d);
  if (name && name[0]) {
    ggml_set_name(cpy, name);
  }
  ggml_backend_sched_set_tensor_backend(gctx->sched, cpy, gctx->backend_cpu);
  ggml_build_forward_expand(gctx->gf, cpy);
  if (src->ne[1] > 1 || src->ne[2] > 1) {
    ggml_tensor *dst3 = ggml_reshape_3d(gctx->ctx0, dst2d, src->ne[0], src->ne[1], src->ne[2]);
    ggml_backend_sched_set_tensor_backend(gctx->sched, dst3, gctx->backend_cpu);
    ggml_build_forward_expand(gctx->gf, dst3);
    return dst3;
  }
  return cpy;
}

static ggml_tensor *hga_q8_0_to_cpu(llm_graph_context *gctx,
                                    ggml_tensor *src, const char *name) {
  if (!gctx || !src || src->type != GGML_TYPE_F32 ||
      !hga_sched_gpu_backend(gctx->sched) || src->ne[0] % 32 != 0) {
    return nullptr;
  }
  ggml_backend_t gpu = hga_sched_gpu_backend(gctx->sched);
  ggml_tensor *q8 = ggml_new_tensor_3d(gctx->ctx0, GGML_TYPE_Q8_0,
                                       src->ne[0], src->ne[1], src->ne[2]);
  if (name && name[0])
    ggml_format_name(q8, "%s_gpu_q8_0", name);
  ggml_backend_sched_set_tensor_backend(gctx->sched, q8, gpu);
  ggml_tensor *quant = ggml_cpy(gctx->ctx0, src, q8);
  ggml_backend_sched_set_tensor_backend(gctx->sched, quant, gpu);
  ggml_build_forward_expand(gctx->gf, quant);
  return hga_copy_to_cpu(gctx, quant, name);
}

/* Named host copies so prefill D2H is a graph node, not an anonymous
 * scheduler side-effect mixed into the CPU staging timer. */
static void hga_prefill_stage_d2h_qkv(llm_graph_context *gctx, ggml_tensor *&Q,
                                      ggml_tensor *&K_rope, ggml_tensor *&V,
                                      ggml_tensor *&K_raw) {
  const bool kraw_alias = (K_raw == K_rope);
  ggml_tensor *const K_rope_gpu = K_rope;
  Q = hga_copy_to_cpu(gctx, Q, "hga_prefill_Q_d2h");
  if (hga_gpu_kv_i8_enabled()) {
    K_rope = hga_q8_0_to_cpu(gctx, K_rope_gpu, "hga_prefill_K_q8_d2h");
    V = hga_q8_0_to_cpu(gctx, V, "hga_prefill_V_q8_d2h");
    K_raw = hga_copy_to_cpu(gctx, kraw_alias ? K_rope_gpu : K_raw,
                            "hga_prefill_Kraw_d2h");
  } else {
    K_rope = hga_copy_to_cpu(gctx, K_rope_gpu, "hga_prefill_K_d2h");
    V = hga_copy_to_cpu(gctx, V, "hga_prefill_V_d2h");
    K_raw = kraw_alias
        ? K_rope
        : hga_copy_to_cpu(gctx, K_raw, "hga_prefill_Kraw_d2h");
  }
}

void hga_cparams_from_ctx_params(llama_cparams &cparams,
                                 const llama_context_params &params) {
  cparams.hga_enabled = params.hga_enabled;
  cparams.hga_levels = params.hga_levels;
  cparams.hga_chunk_size = params.hga_chunk_size;
  cparams.hga_group_size = params.hga_group_size;
  cparams.hga_keep_first = params.hga_keep_first;
  cparams.hga_keep_last = params.hga_keep_last;
  cparams.hga_frac_l1 = params.hga_frac_l1;
  cparams.hga_frac_l2 = params.hga_frac_l2;
  cparams.hga_i8 = params.hga_i8;
  cparams.hga_wave = params.hga_wave;
  cparams.hga_frac_retr = params.hga_frac_retr;
  cparams.hga_frac_est = params.hga_frac_est;
  cparams.hga_runtime = nullptr;
}

void hga_runtime_init(llama_cparams &cparams, const llama_hparams &hparams) {
  if (!cparams.hga_enabled) {
    cparams.hga_runtime = nullptr;
    return;
  }
  const bool is_mtp = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP;
  /* MTP is a separate one-block graph. It needs its own HGA session: its KV
   * history must be routed too, and draft rollback is handled by hga_append's
   * start_pos truncation. */
  const int n_full = is_mtp ? 1 : hga_n_full_layers(hparams);
  if (n_full <= 0) {
    fprintf(stderr, "hga: no full-attention layers; disabling\n");
    cparams.hga_enabled = false;
    cparams.hga_runtime = nullptr;
    return;
  }
  /* HGA OpenMP team. The 16 GB launcher sets HGA_THREADS to the host's
   * physical core count; otherwise follow llama `-t`/`-tb`. */
  int nthr_cap = std::max((int)cparams.n_threads, (int)cparams.n_threads_batch);
  if (const char *e = std::getenv("HGA_THREADS")) {
    const int v = std::atoi(e);
    if (v > 0)
      nthr_cap = v;
  }
  const int nthr_hga = std::max(1, nthr_cap);
  int nthr_pack = nthr_hga;
  if (const char *e = std::getenv("HGA_PACK_THREADS")) {
    const int v = std::atoi(e);
    if (v > 0)
      nthr_pack = v;
  }
  hga_config cfg = hga_config_qwen38_27b(cparams.hga_levels, (int)cparams.n_ctx,
                                         nthr_hga);
  cfg.n_q_heads = (int)hparams.n_head();
  cfg.n_kv_heads = (int)hparams.n_head_kv();
  cfg.head_dim = (int)hparams.n_embd_head_k();
  cfg.rotary_dim = std::min((int)hparams.n_rot(), cfg.head_dim);
  cfg.chunk_size = cparams.hga_chunk_size > 0 ? cparams.hga_chunk_size : 64;
  cfg.group_size = cparams.hga_group_size > 0 ? cparams.hga_group_size : 16;
  cfg.keep_first = cparams.hga_keep_first;
  cfg.keep_last = cparams.hga_keep_last;
  cfg.frac_l1 = cparams.hga_frac_l1;
  cfg.frac_l2 = cparams.hga_frac_l2;
  cfg.levels = (cparams.hga_levels == 1) ? 1 : 2; /* default HGA-2 */
  cfg.max_seq = (int)cparams.n_ctx;
  cfg.n_threads = nthr_hga;
  cfg.n_pack_threads = std::max(1, nthr_pack);
  cfg.prec = cparams.hga_i8 ? HGA_PREC_I8 : HGA_PREC_F16;
  cfg.router = cparams.hga_wave ? HGA_ROUTER_WAVE : HGA_ROUTER_HIER;
  cfg.frac_retr = cparams.hga_frac_retr > 0.f ? cparams.hga_frac_retr : 0.018f;
  cfg.frac_est = cparams.hga_frac_est >= 0.f ? cparams.hga_frac_est : 0.232f;
  hga_session *sess = hga_session_create(&cfg, n_full);
  cparams.hga_runtime = sess;
  /* MTP has no target K/V GEMV reuse and no L2 GEMV plan. graph_mtp sets
   * hga_phase per ubatch: PREFILL for n_tokens>8 (prompt catch-up), DECODE
   * for draft/verify-sized batches. Do not freeze DECODE here — that made
   * catch-up contiguous-D2H the whole prompt (the 4 s "verify" spike). */
  if (is_mtp) {
    g_hga_mtp_session = sess;
    cparams.hga_phase = HGA_SWAP_NONE;
  } else if (!getenv("HGA_L2_OFF")) {
    g_hga_target_session = sess;
    const int n_embd = (int)hparams.n_embd;
    const int n_out = (int)hparams.n_head_kv() * (int)hparams.n_embd_head_k();
    hga_l2_plan *plan =
        hga_l2_plan_create(cfg.n_threads, n_full, n_embd, n_out);
    if (plan) {
      hga_l2_plan_start(plan);
      hga_session_set_l2(sess, plan);
    }
  }
  fprintf(stderr,
          "hga: enabled  levels=%d  router=%s  full-attn layers=%d  chunk=%d "
          "group=%d  "
          "keep_first=%d keep_last=%d  frac_l1=%.3f frac_l2=%.3f  retr=%.3f "
          "est=%.3f  prec=%s  heads=%d/%d dh=%d rd=%d  ctx=%d "
          "route-threads=%d pack-threads=%d\n",
          cfg.levels, cfg.router == HGA_ROUTER_WAVE ? "wave" : "hier", n_full,
          cfg.chunk_size, cfg.group_size, cfg.keep_first, cfg.keep_last,
          cfg.frac_l1, cfg.frac_l2, cfg.frac_retr, cfg.frac_est,
          cfg.prec == HGA_PREC_I8 ? "i8" : "f16", cfg.n_q_heads, cfg.n_kv_heads,
          cfg.head_dim, cfg.rotary_dim, cfg.max_seq, cfg.n_threads,
          cfg.n_pack_threads);
  fprintf(stderr,
          "hga: activation transport=%s  routing/cache=%s  CUDA graph tensors remain F32 where required\n",
          hga_f16_transport_enabled() ? "f16" : "f32",
          cfg.prec == HGA_PREC_I8 ? "integer" : "f16");
}

void hga_runtime_free(llama_cparams &cparams) {
  if (cparams.hga_runtime) {
    hga_session *sess = (hga_session *)cparams.hga_runtime;
    if (g_hga_target_session == sess)
      g_hga_target_session = nullptr;
    if (g_hga_mtp_session == sess)
      g_hga_mtp_session = nullptr;
    if (void *l2 = hga_session_l2(sess)) {
      hga_l2_plan_free((hga_l2_plan *)l2);
      hga_session_set_l2(sess, nullptr);
    }
    hga_session_free(sess);
    cparams.hga_runtime = nullptr;
  }
}

class llm_graph_input_hga : public llm_graph_input_i {
public:
  hga_session *sess = nullptr;
  int layer = -1;
  int capacity = 0;       /* positive: GPU prefill, negative: CPU fallback */
  int total_capacity = 0; /* fixed historical + direct segment bound */
  int history_capacity = 0; /* one united INT8 image width */
  int history_valid_build = 0; /* valid prefix represented by this graph */
  ggml_tensor *history_valid = nullptr; /* dynamic scalar, CUDA resident */
  int chunk_offset = -1;
  void set_input(const llama_ubatch *ubatch) override {
    if (!sess || !ubatch)
      return;
    const int start =
        (ubatch->pos && ubatch->n_tokens > 0) ? (int)ubatch->pos[0] : 0;
    hga_set_ubatch(sess, start, (int)ubatch->n_tokens);
    if (history_valid) {
      const float valid = (float)std::min(start, history_capacity);
      ggml_backend_tensor_set(history_valid, &valid, 0, sizeof(valid));
    }
  }
  bool can_reuse(const llm_graph_params &params) override {
    if (capacity != 0 && sess && layer >= 0) {
      const llama_ubatch &ub = params.ubatch;
      const int start =
          (ub.pos && ub.n_tokens > 0) ? (int)ub.pos[0] : 0;
      if (capacity > 0 && chunk_offset != start % 64)
        return false;
      const int need = hga_gpu_prefill_current_capacity(
          sess, layer, start, (int)ub.n_tokens);
      const int rounded =
          std::max(hga_gpu_prefill_min_keys(), (need + 15) & ~15);
      if (capacity < 0)
        return need > 0 && rounded > hga_gpu_prefill_max_keys();
      /* K/V shapes are fixed, but reusing the graph while its visible-history
       * boundary grows changes model output. Rebuild at the ubatch prefix
       * boundaries until the historical image is saturated; after that the
       * boundary is constant and the graph is safely reusable. */
      const int history_valid_now = std::min(start, history_capacity);
      return need > 0 && rounded <= hga_gpu_prefill_max_keys() &&
             history_valid_now == history_valid_build;
    }
    return true;
  }
};

static bool hga_f32_dim0(const ggml_tensor *t) {
  return t && t->data && t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float);
}

static inline void gather_strided_f32(float *dst, const char *src, int n,
                                      size_t nb) {
  if (nb == sizeof(float)) {
    std::memcpy(dst, src, (size_t)n * sizeof(float));
    return;
  }
  const char *p = src;
  const char *pend = src + (size_t)n * nb;
  for (; p < pend; p += nb, ++dst)
    *dst = *(const float *)p;
}

/* Fallback only: gather a non-F32 / non-dim-contiguous tensor into HGA layout.
 * Walk byte strides (nb[0]/nb[1]/nb[2]) by addition, no d*nb[0] in the dim loop. */
static void pack_heads_f32(const ggml_tensor *t, int n_heads, int n_tok, int dh,
                           std::vector<float> &out) {
  out.resize((size_t)n_heads * (size_t)n_tok * (size_t)dh);
  const char *data = (const char *)t->data;
  const size_t nb0 = t->nb[0];
  const size_t nb1 = t->nb[1];
  const size_t nb2 = t->nb[2];
  float *dst = out.data();
  const bool is_f32 = t->type == GGML_TYPE_F32;
  const bool is_f16 = t->type == GGML_TYPE_F16;
  const char *hbase = data;
  const char *h_end = data + (size_t)n_heads * nb1;
  for (; hbase < h_end; hbase += nb1) {
    const char *kbase = hbase;
    const char *k_end = hbase + (size_t)n_tok * nb2;
    for (; kbase < k_end; kbase += nb2, dst += dh) {
      if (is_f32) {
        gather_strided_f32(dst, kbase, dh, nb0);
      } else if (is_f16) {
        if (nb0 == sizeof(ggml_fp16_t)) {
          /* The generic ggml row helper is deliberately scalar. The CPU
           * backend implementation is compiled for this machine's F16C/AVX2
           * ISA and converts eight elements per instruction. Q/K/V/Kraw are
           * contiguous in dim 0 on the HGA transport path. */
          ggml_cpu_fp16_to_fp32((const ggml_fp16_t *)kbase, dst, dh);
          continue;
        }
        const char *p = kbase;
        const char *p_end = kbase + (size_t)dh * nb0;
        float *d = dst;
        for (; p < p_end; p += nb0, ++d)
          *d = ggml_fp16_to_fp32(*(const ggml_fp16_t *)p);
      } else {
        const char *p = kbase;
        const char *p_end = kbase + (size_t)dh * nb0;
        float *d = dst;
        for (; p < p_end; p += nb0, ++d)
          memcpy(d, p, sizeof(float));
      }
    }
  }
}

struct hga_f32_stage_view {
  const float *data = nullptr;
  int head_stride = 0;
  int tok_stride = 0;
};

struct hga_f32_stage_scratch {
  std::vector<float> q;
  std::vector<float> k;
  std::vector<float> v;
  std::vector<float> kraw;
};

/* HGA routing remains F32 internally. F16 transport is expanded once on the
 * CPU into the existing input contract; contiguous F32 keeps its zero-copy
 * fast path. */
static bool hga_stage_as_f32(const ggml_tensor *t, int n_heads, int n_tok,
                             int dh, std::vector<float> &scratch,
                             hga_f32_stage_view &view) {
  if (!t || !t->data || n_heads <= 0 || n_tok <= 0 || dh <= 0) {
    return false;
  }
  if (hga_f32_dim0(t)) {
    view.data = (const float *)t->data;
    view.head_stride = (int)(t->nb[1] / sizeof(float));
    view.tok_stride = (int)(t->nb[2] / sizeof(float));
    return true;
  }
  if (t->type != GGML_TYPE_F16 || t->nb[0] != sizeof(ggml_fp16_t)) {
    return false;
  }
  pack_heads_f32(t, n_heads, n_tok, dh, scratch);
  view.data = scratch.data();
  view.head_stride = n_tok * dh;
  view.tok_stride = dh;
  return true;
}

struct hga_op_ud {
  hga_session *sess;
  int hga_il;
  int32_t phase;
  bool is_mtp;
};

struct hga_gpu_prefill_ud {
  hga_session *sess;
  int hga_il;
  int capacity;
  int n_q_graph;
  bool stage_i8;
  bool united;
};

struct hga_gpu_verify_ud {
  hga_session *sess;
  int hga_il;
  int history_capacity;
  int graph_n_q;
  int32_t phase;
  bool is_mtp;
};

/* ggml views inherit their storage tensor's scalar type. The compact staging
 * image is byte-addressed and intentionally contains I8 and F16 sections, so
 * retag the metadata-only view after ggml records its backing buffer/offset. */
static ggml_tensor *hga_view_3d_as(ggml_context *ctx, ggml_tensor *storage,
                                   ggml_type type, int64_t ne0, int64_t ne1,
                                   int64_t ne2, size_t nb1, size_t nb2,
                                   size_t offset) {
  ggml_tensor *view = ggml_view_3d(ctx, storage, ne0, ne1, ne2, nb1, nb2,
                                   offset);
  view->type = type;
  view->nb[0] = ggml_type_size(type);
  return view;
}

static ggml_tensor *hga_view_2d_as(ggml_context *ctx, ggml_tensor *storage,
                                   ggml_type type, int64_t ne0, int64_t ne1,
                                   size_t nb1, size_t offset) {
  ggml_tensor *view = ggml_view_2d(ctx, storage, ne0, ne1, nb1, offset);
  view->type = type;
  view->nb[0] = ggml_type_size(type);
  return view;
}

/* One graph node is shared by all full-attention layers. A four-byte dynamic
 * input hides historical slots beyond the current prefix; current-ubatch keys
 * use the ordinary causal triangle. Recreate on the first HGA layer of each
 * graph so a recycled ggml_context can never return a stale tensor pointer. */
static ggml_tensor *hga_gpu_prefill_causal_mask(
    llm_graph_context *gctx, int hga_il, int64_t history_capacity,
    ggml_tensor *history_valid, int64_t n_q, int64_t direct_capacity) {
  struct mask_cache {
    ggml_context *ctx = nullptr;
    int64_t history_capacity = 0;
    int64_t n_q = 0;
    int64_t direct_capacity = 0;
    ggml_tensor *history_valid = nullptr;
    ggml_tensor *mask = nullptr;
  };
  static thread_local mask_cache cache;
  if (hga_il != 0 && cache.ctx == gctx->ctx0 &&
      cache.history_capacity == history_capacity &&
      cache.n_q == n_q && cache.direct_capacity == direct_capacity &&
      cache.mask != nullptr)
    return cache.mask;
  GGML_ASSERT(history_valid != nullptr && direct_capacity >= n_q);

  ggml_tensor *history_index = ggml_arange(
      gctx->ctx0, 0.0f, (float)history_capacity, 1.0f);
  ggml_set_name(history_index, "hga_mask_history_index");
  ggml_tensor *valid = ggml_repeat_4d(
      gctx->ctx0,
      ggml_reshape_3d(gctx->ctx0, history_valid, 1, 1, 1),
      history_capacity, 1, 1, 1);
  ggml_set_name(valid, "hga_mask_valid_repeat");
  ggml_tensor *history_is_valid = ggml_step(
      gctx->ctx0,
      ggml_scale_bias(
          gctx->ctx0, ggml_sub(gctx->ctx0, valid, history_index),
          1.0f, -0.5f));
  ggml_set_name(history_is_valid, "hga_mask_history_valid");
  ggml_tensor *history_threshold = ggml_scale_bias(
      gctx->ctx0, history_is_valid, -(float)n_q, (float)n_q);
  ggml_set_name(history_threshold, "hga_mask_history_threshold");
  ggml_tensor *direct_threshold =
      ggml_arange(gctx->ctx0, 0.0f, (float)direct_capacity, 1.0f);
  ggml_set_name(direct_threshold, "hga_mask_direct_threshold");
  ggml_tensor *threshold = ggml_concat(
      gctx->ctx0, history_threshold, direct_threshold, 0);
  ggml_set_name(threshold, "hga_mask_threshold_concat");
  /* Direct columns [n_q, direct_capacity) are padding. Their threshold is
   * beyond every real query index, so the same causal comparison masks them
   * without another visibility tensor or CPU-authored mask. */
  const int64_t n_keys = history_capacity + direct_capacity;
  threshold = ggml_repeat_4d(
      gctx->ctx0, ggml_reshape_3d(gctx->ctx0, threshold, n_keys, 1, 1),
      n_keys, n_q, 1, 1);
  ggml_set_name(threshold, "hga_mask_threshold_repeat");
  ggml_tensor *query_pos = ggml_repeat_4d(
      gctx->ctx0,
      ggml_reshape_3d(
          gctx->ctx0,
          ggml_arange(gctx->ctx0, 0.0f, (float)n_q, 1.0f), 1, n_q, 1),
      n_keys, n_q, 1, 1);
  ggml_set_name(query_pos, "hga_mask_query_repeat");
  ggml_tensor *visible = ggml_step(
      gctx->ctx0,
      ggml_scale_bias(gctx->ctx0,
                      ggml_sub(gctx->ctx0, query_pos, threshold),
                      1.0f, 0.5f));
  ggml_set_name(visible, "hga_mask_visible");
  ggml_tensor *mask_f32 =
      ggml_scale_bias(gctx->ctx0, visible, 10000.0f, -10000.0f);
  ggml_set_name(mask_f32, "hga_mask_scale_f32");
  ggml_tensor *mask = ggml_cast(gctx->ctx0, mask_f32, GGML_TYPE_F16);
  ggml_set_name(mask, "hga_prefill_causal_mask");
  hga_pin_gpu(gctx->sched, history_index);
  hga_pin_gpu(gctx->sched, valid);
  hga_pin_gpu(gctx->sched, history_is_valid);
  hga_pin_gpu(gctx->sched, history_threshold);
  hga_pin_gpu(gctx->sched, direct_threshold);
  hga_pin_gpu(gctx->sched, threshold);
  hga_pin_gpu(gctx->sched, query_pos);
  hga_pin_gpu(gctx->sched, visible);
  hga_pin_gpu(gctx->sched, mask_f32);
  hga_pin_gpu(gctx->sched, mask);
  cache = {gctx->ctx0, history_capacity, n_q, direct_capacity,
           history_valid, mask};
  return mask;
}

static void hga_gpu_prefill_stage_op(ggml_tensor *dst, int ith, int nth,
                                     void *userdata) {
  if (ith != 0)
    return;
  GGML_UNUSED(nth);
  auto *ud = (hga_gpu_prefill_ud *)userdata;
  hga_session *sess = ud->sess;
  if (!sess)
    return;

  ggml_tensor *Q = dst->src[0];
  ggml_tensor *Krope = dst->src[1];
  ggml_tensor *V = dst->src[2];
  ggml_tensor *Kraw = dst->src[3];
  int n_q = (int)Q->ne[2];
  const uint32_t n_real = hga_ubatch_padded_n_real();
  if (n_real > 0 && n_real < (uint32_t)n_q)
    n_q = (int)n_real;
  const int start = hga_ubatch_start(sess);
  const int hga_il = ud->hga_il;
  const size_t image_bytes = ggml_nbytes(dst);

  const int dh = (int)Q->ne[0];
  const int n_heads = (int)Q->ne[1];
  const int n_kv_heads = (int)Krope->ne[1];
  static thread_local hga_f32_stage_scratch scratch;
  hga_f32_stage_view qv, kv, vv, krv;
  const bool q8_kv = Krope->type == GGML_TYPE_Q8_0 &&
                     V->type == GGML_TYPE_Q8_0;
  const bool inputs_ok =
      hga_stage_as_f32(Q, n_heads, n_q, dh, scratch.q, qv) &&
      (q8_kv
           ? true
           : hga_stage_as_f32(Krope, n_kv_heads, n_q, dh, scratch.k, kv)) &&
      (q8_kv
           ? true
           : hga_stage_as_f32(V, n_kv_heads, n_q, dh, scratch.v, vv)) &&
      (Kraw == Krope
           ? (krv = kv, true)
           : hga_stage_as_f32(Kraw, n_kv_heads, n_q, dh, scratch.kraw, krv));
  if (!inputs_ok) {
    std::fprintf(stderr,
                 "hga-gpu: stage requires contiguous-dim F32/F16 Q/K/V or Q8_0 K/V at layer %d\n",
                 hga_il);
    std::abort();
  }

  const auto t0 = std::chrono::steady_clock::now();
  hga_stats st{};
  int n_keys = 0;
  if (ud->united && ud->stage_i8 && q8_kv) {
    n_keys = hga_prepare_gpu_prefill_i8_q8_0_strided(
        sess, hga_il, start, n_q, qv.data, qv.head_stride, qv.tok_stride,
        Krope->data, (int)Krope->nb[0], (int)Krope->nb[1],
        (int)Krope->nb[2], krv.data, krv.head_stride, krv.tok_stride, V->data,
        (int)V->nb[0], (int)V->nb[1], (int)V->nb[2], dst->data, image_bytes,
        ud->capacity, &st);
  } else if (ud->united && ud->stage_i8) {
    n_keys = hga_prepare_gpu_prefill_i8_strided(
        sess, hga_il, start, n_q, qv.data, qv.head_stride, qv.tok_stride,
        kv.data, kv.head_stride, kv.tok_stride, krv.data, krv.head_stride,
        krv.tok_stride, vv.data, vv.head_stride, vv.tok_stride, dst->data,
        image_bytes, ud->capacity, &st);
  } else if (ud->united) {
    n_keys = hga_prepare_gpu_prefill_f16_ubatch_strided(
        sess, hga_il, start, n_q, qv.data, qv.head_stride, qv.tok_stride,
        kv.data, kv.head_stride, kv.tok_stride, krv.data, krv.head_stride,
        krv.tok_stride, vv.data, vv.head_stride, vv.tok_stride,
        (uint16_t *)dst->data, image_bytes / sizeof(uint16_t), ud->capacity,
        &st);
  } else {
    n_keys = hga_prepare_gpu_prefill_f16_strided(
        sess, hga_il, start, n_q, qv.data, qv.head_stride, qv.tok_stride,
        kv.data, kv.head_stride, kv.tok_stride, krv.data, krv.head_stride,
        krv.tok_stride, vv.data, vv.head_stride, vv.tok_stride,
        (uint16_t *)dst->data, image_bytes / sizeof(uint16_t), ud->capacity,
        &st);
  }
  if (n_keys <= 0) {
    std::fprintf(stderr,
                 "hga-gpu: staging failed layer=%d hga=%d start=%d n_q=%d capacity=%d bytes=%zu\n",
                 hga_il, hga_il, start, n_q, ud->capacity, image_bytes);
    std::abort();
  }

  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  static double chunk_ms = 0.0;
  static double route_ms = 0.0;
  static double pack_ms = 0.0;
  static double append_ms = 0.0;
  static double close_ms = 0.0;
  static double union_ms = 0.0;
  static double scale_clear_ms = 0.0;
  static double kv_copy_ms = 0.0;
  static double pack_other_ms = 0.0;
  static int route_requests = 0;
  static int route_union = 0;
  static int route_retained = 0;
  static int route_head_uses = 0;
  static int route_chunk_uses = 0;
  static int route_max_requests = 0;
  static int route_max_heads = 0;
  static int route_max_chunks = 0;
  static int route_history_selected = 0;
  static int route_history_max = 0;
  static int route_history_samples = 0;
  static int route_limit = 0;
  static int context_max = 0;
  static int calls = 0;
  static bool tot_atexit = false;
  chunk_ms += ms;
  route_ms += st.ms_route;
  pack_ms += st.ms_pack;
  append_ms += st.ms_prefill_append;
  close_ms += st.ms_prefill_close;
  union_ms += st.ms_prefill_union;
  scale_clear_ms += st.ms_prefill_scale_clear;
  kv_copy_ms += st.ms_prefill_kv_copy;
  pack_other_ms += st.ms_prefill_pack_other;
  route_requests += st.n_route_group_requests;
  route_union += st.n_route_group_union;
  route_retained += st.n_route_group_retained;
  route_head_uses += st.n_route_group_head_uses;
  route_chunk_uses += st.n_route_group_chunk_uses;
  route_max_requests =
      std::max(route_max_requests, st.n_route_group_max_requests);
  route_max_heads = std::max(route_max_heads, st.n_route_group_max_heads);
  route_max_chunks = std::max(route_max_chunks, st.n_route_group_max_chunks);
  route_history_selected += st.n_route_history_selected;
  route_history_max =
      std::max(route_history_max, st.n_route_history_max);
  route_history_samples += hga_session_config(sess)->n_kv_heads;
  route_limit = st.n_route_topk_limit;
  context_max = std::max(context_max, st.n_kv);
  ++calls;
  if (calls % 16 == 0) {
    std::fprintf(stderr,
                 "hga-gpu: CPU stage 16 layers %.2f ms route=%.2f pack=%.2f context=%d keys=%d capacity=%d n_q=%d groups=requested:%d union:%d retained:%d overlap=%.1f%% uses/group=%.2f heads/group=%.2f chunks/group=%.2f max=%d/%d/%d selected-history=%.1f/%d fair-topk=%d\n",
                 chunk_ms, route_ms, pack_ms, context_max, n_keys,
                 ud->capacity, n_q,
                 route_requests, route_union, route_retained,
                 route_requests > 0
                     ? 100.0 * (1.0 - (double)route_union / route_requests)
                     : 0.0,
                 route_union > 0 ? (double)route_requests / route_union : 0.0,
                 route_union > 0 ? (double)route_head_uses / route_union : 0.0,
                 route_union > 0 ? (double)route_chunk_uses / route_union : 0.0,
                 route_max_requests, route_max_heads, route_max_chunks,
                 route_history_samples > 0
                     ? (double)route_history_selected / route_history_samples
                     : 0.0,
                 route_history_max,
                 route_limit);
    std::fprintf(stderr,
                 "hga-gpu: CPU detail 16 layers append=%.2f close=%.2f route=%.2f union=%.2f scale-clear=%.2f kv-copy=%.2f other=%.2f ms\n",
                 append_ms, close_ms, route_ms, union_ms, scale_clear_ms,
                 kv_copy_ms, pack_other_ms);
    g_prefill_cpu_tot.chunk_ms += chunk_ms;
    g_prefill_cpu_tot.route_ms += route_ms;
    g_prefill_cpu_tot.pack_ms += pack_ms;
    g_prefill_cpu_tot.append_ms += append_ms;
    g_prefill_cpu_tot.close_ms += close_ms;
    g_prefill_cpu_tot.union_ms += union_ms;
    g_prefill_cpu_tot.scale_clear_ms += scale_clear_ms;
    g_prefill_cpu_tot.kv_copy_ms += kv_copy_ms;
    g_prefill_cpu_tot.pack_other_ms += pack_other_ms;
    g_prefill_cpu_tot.calls += 16;
    if (!tot_atexit) {
      tot_atexit = true;
      std::atexit(hga_prefill_cpu_dump_total);
    }
    chunk_ms = route_ms = pack_ms = 0.0;
    append_ms = close_ms = union_ms = scale_clear_ms = 0.0;
    kv_copy_ms = pack_other_ms = 0.0;
    route_requests = route_union = route_retained = route_limit = 0;
    route_head_uses = route_chunk_uses = 0;
    route_max_requests = route_max_heads = route_max_chunks = 0;
    route_history_selected = route_history_max = 0;
    route_history_samples = 0;
    context_max = 0;
  }
}

/* Decode / spec-verify / prefill HGA wall time. */
static void hga_account(int n_q, const hga_stats &st,
                        std::chrono::steady_clock::time_point t0, int32_t phase,
                        bool is_mtp) {
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  if (is_mtp) {
    static int n = 0;
    static double gen_ms = 0;
    static int gen_n = 0;
    n++;
    if (n_q <= 8) {
      gen_ms += ms;
      gen_n++;
    }
    if (n_q > 8 || n <= 8 || n % 8 == 0) {
      fprintf(stderr,
              "hga-prof mtp #%d n_q=%d kv=%d = %.2f ms  route=%.2f attn=%.2f  "
              "att=%d  phase=%d  gen_hga_sum=%.1f ms / %d calls\n",
              n, n_q, st.n_kv, ms, st.ms_route, st.ms_attn,
              st.n_attended_tokens, (int)phase, gen_ms, gen_n);
    }
    return;
  }
  if (n_q != 1) {
    /* Spec verify is VERIFY (or DECODE fallback) with n_q = K+1. Prefill is n_q>>1. */
    if (hga_decode_pack(phase)) {
      static double tok_ms = 0, sum_ms = 0, route_ms = 0, attn_ms = 0,
                    pack_ms = 0, kernel_ms = 0;
      static int layer_i = 0, n_batch = 0, att_sum = 0, kv_last = 0, nq_last = 0;
      tok_ms += ms;
      route_ms += st.ms_route;
      attn_ms += st.ms_attn;
      pack_ms += st.ms_pack;
      kernel_ms += st.ms_kernel;
      att_sum += st.n_attended_tokens;
      kv_last = st.n_kv;
      nq_last = n_q;
      layer_i++;
      if (layer_i == 16) {
        n_batch++;
        sum_ms += tok_ms;
        fprintf(stderr,
                "hga-prof verify batch %d: HGA 16 layers n_q=%d = %.2f ms  "
                "(mean %.2f)  route=%.2f attn=%.2f (pack=%.2f kernel=%.2f)  att/layer=%.0f kv=%d  "
                "hga_sum=%.1f ms\n",
                n_batch, nq_last, tok_ms, sum_ms / n_batch, route_ms,
                attn_ms, pack_ms, kernel_ms, att_sum / 16.0, kv_last, sum_ms);
        tok_ms = 0;
        route_ms = 0;
      attn_ms = 0;
      pack_ms = 0;
      kernel_ms = 0;
        att_sum = 0;
        layer_i = 0;
      }
      return;
    }
    static double prefill_ms = 0, pool_ms = 0, chunk_scan_ms = 0,
                  chunk_topk_ms = 0, group_scan_ms = 0, group_topk_ms = 0,
                  route_other_ms = 0, qquant_ms = 0, span_keys_ms = 0,
                  kernel_ms = 0, merge_ms = 0, l2_fixed_ms = 0,
                  l2_routed_ms = 0, l2_active_ms = 0;
    static int prefill_tok = 0, prefill_calls = 0;
    prefill_ms += ms;
    pool_ms += st.ms_q_pool;
    chunk_scan_ms += st.ms_route_chunk_scan;
    chunk_topk_ms += st.ms_route_chunk_topk;
    group_scan_ms += st.ms_route_group_scan;
    group_topk_ms += st.ms_route_group_topk;
    route_other_ms += st.ms_route_other;
    qquant_ms += st.ms_q_quant;
    span_keys_ms += st.ms_span_keys;
    kernel_ms += st.ms_attn_kernel;
    merge_ms += st.ms_attn_merge;
    l2_fixed_ms += st.ms_l2_load_fixed;
    l2_routed_ms += st.ms_l2_load_routed;
    l2_active_ms += st.ms_l2_load_active;
    prefill_tok += n_q;
    prefill_calls++;
    if (prefill_calls % 16 == 0) {
      const int chunk = prefill_calls / 16;
      fprintf(stderr,
              "hga-prof prefill chunk %d: %.1f ms for %d tok  (%.2f ms/tok "
              "all-16-layers)  att=%d kv=%d n_q=%d\n",
              chunk, prefill_ms, prefill_tok,
              prefill_ms / std::max(1, prefill_tok / 16), st.n_attended_tokens,
              st.n_kv, n_q);
      fprintf(stderr,
              "hga-prof prefill route %d: pool=%.2f  chunk-scan=%.2f "
              "topk=%.2f  group-scan=%.2f topk=%.2f  other=%.2f  "
              "qquant=%.2f  spans/setup=%.2f ms  mid=%d groups=%d  "
              "summary-unique=%.1f+%.1f KiB logical=%.1f+%.1f KiB\n",
              chunk, pool_ms, chunk_scan_ms, chunk_topk_ms, group_scan_ms,
              group_topk_ms, route_other_ms, qquant_ms, span_keys_ms,
              st.n_route_mid_chunks, st.n_route_group_candidates,
              st.route_chunk_bytes_unique / 1024.0,
              st.route_group_bytes_unique / 1024.0,
              st.route_chunk_bytes_logical / 1024.0,
              st.route_group_bytes_logical / 1024.0);
      fprintf(stderr,
              "hga-prof prefill attention %d: kernel=%.2f merge=%.2f ms  "
              "tokens fixed=%d routed=%d active=%d total=%d\n",
              chunk, kernel_ms, merge_ms, st.n_fixed_tokens,
              st.n_routed_tokens, st.n_active_tokens, st.n_attended_tokens);
      if (l2_fixed_ms + l2_routed_ms + l2_active_ms > 0.0) {
        fprintf(stderr,
                "hga-prof prefill L2-load %d: fixed=%.2f routed=%.2f "
                "active=%.2f total=%.2f ms (diagnostic preload)\n",
                chunk, l2_fixed_ms, l2_routed_ms, l2_active_ms,
                l2_fixed_ms + l2_routed_ms + l2_active_ms);
      }
      prefill_ms = 0;
      prefill_tok = 0;
      pool_ms = chunk_scan_ms = chunk_topk_ms = group_scan_ms = 0;
      group_topk_ms = route_other_ms = qquant_ms = span_keys_ms = 0;
      kernel_ms = merge_ms = 0;
      l2_fixed_ms = l2_routed_ms = l2_active_ms = 0;
    }
    return;
  }
  static double tok_ms = 0, sum_ms = 0;
  static int layer_i = 0, n_tok = 0, att_sum = 0, kv_last = 0;
  tok_ms += ms;
  att_sum += st.n_attended_tokens;
  kv_last = st.n_kv;
  layer_i++;
  if (layer_i == 16) {
    n_tok++;
    sum_ms += tok_ms;
    if (n_tok <= 3 || n_tok % 16 == 0) {
      fprintf(stderr,
              "hga-prof decode tok %d: HGA 16 layers = %.2f ms  (mean %.2f)  "
              "att/layer=%.0f kv=%d  ~%.1f tok/s if HGA were the only cost\n",
              n_tok, tok_ms, sum_ms / n_tok, att_sum / 16.0, kv_last,
              1000.0 / std::max(0.01, tok_ms));
    }
    tok_ms = 0;
    att_sum = 0;
    layer_i = 0;
  }
}

static void hga_gpu_verify_stage_op(ggml_tensor *dst, int ith, int nth,
                                    void *userdata) {
  if (ith != 0)
    return;
  GGML_UNUSED(nth);
  auto *ud = (hga_gpu_verify_ud *)userdata;
  hga_session *sess = ud->sess;
  if (!sess)
    return;

  ggml_tensor *Q = dst->src[0];
  ggml_tensor *Krope = dst->src[1];
  ggml_tensor *V = dst->src[2];
  ggml_tensor *Kraw = dst->src[3];
  int n_q = ud->graph_n_q;
  const uint32_t n_real = hga_ubatch_padded_n_real();
  if (n_real > 0 && n_real < (uint32_t)n_q)
    n_q = (int)n_real;
  const int start = hga_ubatch_start(sess);

  const int dh = (int)Q->ne[0];
  const int n_heads = (int)Q->ne[1];
  const int n_kv_heads = (int)Krope->ne[1];
  static thread_local hga_f32_stage_scratch scratch;
  hga_f32_stage_view qv, kv, vv, krv;
  const bool inputs_ok =
      hga_stage_as_f32(Q, n_heads, n_q, dh, scratch.q, qv) &&
      hga_stage_as_f32(Krope, n_kv_heads, n_q, dh, scratch.k, kv) &&
      hga_stage_as_f32(V, n_kv_heads, n_q, dh, scratch.v, vv) &&
      (Kraw == Krope
           ? (krv = kv, true)
           : hga_stage_as_f32(Kraw, n_kv_heads, n_q, dh, scratch.kraw, krv));
  if (!inputs_ok) {
    std::fprintf(stderr,
                 "hga-gpu: VERIFY stage requires contiguous-dim F32/F16 Q/K/V at layer %d\n",
                 ud->hga_il);
    std::abort();
  }

  const auto t0 = std::chrono::steady_clock::now();
  hga_stats st{};
  const int n_history = hga_prepare_gpu_verify_i8_strided(
      sess, ud->hga_il, start, n_q, ud->graph_n_q,
      qv.data, qv.head_stride, qv.tok_stride, kv.data, kv.head_stride,
      kv.tok_stride, krv.data, krv.head_stride, krv.tok_stride, vv.data,
      vv.head_stride, vv.tok_stride,
      dst->data, ggml_nbytes(dst), ud->history_capacity, &st);
  if (n_history < 0) {
    std::fprintf(stderr,
                 "hga-gpu: VERIFY staging failed layer=%d start=%d n_q=%d graph=%d capacity=%d\n",
                 ud->hga_il, start, n_q, ud->graph_n_q,
                 ud->history_capacity);
    std::abort();
  }

  hga_account(n_q, st, t0, ud->phase, ud->is_mtp);
  hga_l2_after_hga(sess, ud->hga_il);
}

static void hga_custom_op(ggml_tensor *dst, int ith, int nth, void *userdata) {
  /* Single-threaded op: ith==0 does the work. ggml may still split; ignore
   * extra tasks. OpenMP inside hga_forward_strided uses the session thread
   * count. */
  if (ith != 0)
    return;
  GGML_UNUSED(nth);
  auto *ud = (hga_op_ud *)userdata;
  hga_session *sess = ud->sess;
  if (!sess)
    return;
  const auto t0 = std::chrono::steady_clock::now();

  ggml_tensor *Q = dst->src[0];
  ggml_tensor *Krope = dst->src[1];
  ggml_tensor *V = dst->src[2];
  ggml_tensor *Kraw = dst->src[3];

  int n_q = (int)Q->ne[2];
  /* Padded generate graphs are n_ubatch=3 for reuse. HGA only needs the real
   * tokens: n_real=1 uses the decode kernel; n_real=2 uses the small-batch
   * verify path. Pad keys must not enter the HGA KV session. */
  {
    const uint32_t n_real = hga_ubatch_padded_n_real();
    if (n_real > 0 && n_real < (uint32_t) n_q) {
      n_q = (int) n_real;
    }
  }
  const int start = hga_ubatch_start(sess);
  const int hga_il = ud->hga_il;

  static bool logged_src = false;
  if (!logged_src) {
    logged_src = true;
    const bool host =
        Q->buffer ? ggml_backend_buffer_is_host(Q->buffer) : false;
    auto is_host = [](const ggml_tensor *t) {
      return t && t->buffer && ggml_backend_buffer_is_host(t->buffer);
    };
    fprintf(stderr,
            "hga: src host Q=%d K=%d V=%d Kraw=%d  Q ne=[%lld,%lld,%lld] type=%s\n",
            (int)host, (int)is_host(Krope), (int)is_host(V), (int)is_host(Kraw),
            (long long)Q->ne[0], (long long)Q->ne[1], (long long)Q->ne[2],
            ggml_type_name(Q->type));
  }

  hga_stats st{};
  const bool output_f16 = dst->type == GGML_TYPE_F16;
  static thread_local std::vector<float> output_scratch;
  float *dout = (float *)dst->data;
  if (output_f16) {
    output_scratch.assign((size_t)ggml_nelements(dst), 0.0f);
    dout = output_scratch.data();
  }
  const auto finish_output = [&] {
    if (output_f16) {
      /* This output is the other large half of the activation-wire
       * conversion. Use the CPU backend's vectorized F16C implementation,
       * not ggml's scalar reference row helper. */
      ggml_cpu_fp32_to_fp16(output_scratch.data(),
                            (ggml_fp16_t *)dst->data,
                            ggml_nelements(dst));
    }
  };

  if (hga_f32_dim0(Q) && hga_f32_dim0(Krope) && hga_f32_dim0(V) &&
      hga_f32_dim0(Kraw)) {
    /* ggml [dh, heads, tok] → strides in floats. Quantize K/V/Q to INT8 in one
     * pass.  This is deliberately independent of swap phase: MTP's prompt
     * catch-up arrives as an ordinary multi-token ubatch and must append its
     * routed K/V at those absolute positions before the first draft token. */
    hga_forward_strided(
        sess, hga_il, start, n_q, (const float *)Q->data,
        (int)(Q->nb[1] / sizeof(float)), (int)(Q->nb[2] / sizeof(float)),
        (const float *)Krope->data, (int)(Krope->nb[1] / sizeof(float)),
        (int)(Krope->nb[2] / sizeof(float)), (const float *)Kraw->data,
        (int)(Kraw->nb[1] / sizeof(float)), (int)(Kraw->nb[2] / sizeof(float)),
        (const float *)V->data, (int)(V->nb[1] / sizeof(float)),
        (int)(V->nb[2] / sizeof(float)), dout, HGA_OUT_TOKEN_MAJOR, &st);
    hga_account(n_q, st, t0, ud->phase, ud->is_mtp);
    if (n_q > 8 && hga_l3_prefetch_enabled()) {
      const int n_layers = hga_session_n_layers(sess);
      if (ud->is_mtp) {
        /* MTP is the last CPU HGA before the next target ubatch. */
        hga_l3_prefetch_layer(g_hga_target_session, g_hga_target_session, 0);
      } else if (hga_il + 1 < n_layers) {
        hga_l3_prefetch_layer(sess, sess, hga_il + 1);
      } else if (g_hga_mtp_session) {
        hga_l3_prefetch_layer(sess, g_hga_mtp_session, 0);
      }
    } else if (n_q >= 1 && n_q <= 8) {
      hga_l2_after_hga(sess, hga_il);
    }
    finish_output();
    return;
  }

  const int dh = (int)Q->ne[0];
  const int H = (int)Q->ne[1];
  const int KVH = (int)Krope->ne[1];
  std::vector<float> q, k, v, kraw;
  pack_heads_f32(Q, H, n_q, dh, q);
  pack_heads_f32(Krope, KVH, n_q, dh, k);
  pack_heads_f32(V, KVH, n_q, dh, v);
  pack_heads_f32(Kraw, KVH, n_q, dh, kraw);
  const int q_hs = n_q * dh;
  const int k_hs = n_q * dh;
  hga_forward_strided(sess, hga_il, start, n_q, q.data(), q_hs, dh, k.data(),
                      k_hs, dh, kraw.data(), k_hs, dh, v.data(), k_hs, dh, dout,
                      HGA_OUT_TOKEN_MAJOR, &st);
  hga_account(n_q, st, t0, ud->phase, ud->is_mtp);
  if (n_q > 8 && hga_l3_prefetch_enabled()) {
    const int n_layers = hga_session_n_layers(sess);
    if (ud->is_mtp)
      hga_l3_prefetch_layer(g_hga_target_session, g_hga_target_session, 0);
    else if (hga_il + 1 < n_layers)
      hga_l3_prefetch_layer(sess, sess, hga_il + 1);
    else if (g_hga_mtp_session)
      hga_l3_prefetch_layer(sess, g_hga_mtp_session, 0);
  } else if (n_q >= 1 && n_q <= 8) {
    hga_l2_after_hga(sess, hga_il);
  }
  finish_output();
}

ggml_tensor *hga_build_full_attn(llm_graph_context *gctx,
                                 llm_graph_input_attn_kv *inp, ggml_tensor *Q,
                                 ggml_tensor *K_rope, ggml_tensor *V,
                                 ggml_tensor *K_raw, float kq_scale, int il) {
  GGML_UNUSED(kq_scale);
  GGML_UNUSED(inp);
  hga_session *sess = (hga_session *)gctx->cparams.hga_runtime;
  if (!sess) {
    fprintf(stderr, "hga: missing runtime at layer %d\n", il);
    return Q;
  }

  {
    static bool logged_phase = false;
    if (!logged_phase) {
      logged_phase = true;
      fprintf(stderr,
              "hga: graph phase=%d swap=%p (0=none 1=prefill 2=decode)\n",
              (int)gctx->cparams.hga_phase, gctx->cparams.hga_swap);
    }
  }
  /* The target profiler and weight streamer callbacks are installed (and
   * chained) by hga_swap_ensure after scheduler creation.  Do not install a
   * scheduler callback here: doing so replaces the streamer, silently leaves
   * the wrong exchange weights resident, and corrupts speculative output. */

  /* Register a graph input once so start_pos is refreshed every eval. */
  llm_graph_input_hga *hga_input = nullptr;
  {
    auto hin = std::make_unique<llm_graph_input_hga>();
    hin->sess = sess;
    hga_input = hin.get();
    gctx->res->add_input(std::move(hin));
  }

  if (K_raw == nullptr)
    K_raw = K_rope;

  /* Hybrid PREFILL: CPU routing remains independent for every 64-token
   * chunk/head, then one capacity-bounded historical KV union serves the
   * complete physical ubatch. CUDA adds direct K/V, applies only the shared
   * causal triangle, and evaluates all queries with flash attention. */
  const int32_t phase = gctx->cparams.hga_phase;
  if (phase == HGA_SWAP_PREFILL && Q->ne[2] > 8 &&
      hga_gpu_prefill_enabled() && hga_sched_gpu_backend(gctx->sched)) {
    const int hga_il = hga_layer_index(gctx->cparams, gctx->hparams, il);
    const int graph_start =
        (gctx->ubatch.pos && gctx->ubatch.n_tokens > 0)
            ? (int)gctx->ubatch.pos[0]
            : hga_ubatch_start(sess);
    const int need = hga_gpu_prefill_current_capacity(
        sess, hga_il, graph_start, (int)Q->ne[2]);
    const int total_capacity =
        std::max(hga_gpu_prefill_min_keys(), (need + 15) & ~15);
    if (need > 0 && total_capacity <= hga_gpu_prefill_max_keys()) {
      hga_input->layer = hga_il;
      hga_input->capacity = total_capacity;
      hga_input->total_capacity = total_capacity;
      hga_input->chunk_offset = graph_start % 64;
      const int64_t dh = Q->ne[0];
      const int64_t n_q = Q->ne[2];
      const int64_t direct_capacity = std::max<int64_t>(
          n_q, (int64_t)gctx->cparams.n_ubatch);
      const int64_t n_heads = Q->ne[1];
      const int64_t kvh = K_rope->ne[1];
      const bool stage_i8 =
          hga_session_config(sess)->prec == HGA_PREC_I8;
      const int history_capacity = hga_gpu_prefill_ubatch_history_capacity(
          sess, hga_il, graph_start, (int)n_q, hga_gpu_prefill_max_keys());
      hga_input->history_capacity = history_capacity;
      hga_input->history_valid_build =
          std::min(graph_start, history_capacity);
      if (history_capacity > 0) {
        if (hga_il == 0) {
          hga_input->history_valid =
              ggml_new_tensor_1d(gctx->ctx0, GGML_TYPE_F32, 1);
          ggml_set_name(hga_input->history_valid, "hga_history_valid");
          ggml_set_input(hga_input->history_valid);
          hga_pin_gpu(gctx->sched, hga_input->history_valid);
        }
        hga_gpu_prefill_i8_layout layout{};
        if (stage_i8) {
          GGML_ASSERT(hga_gpu_prefill_i8_image_layout(
              sess, history_capacity, (int)n_q, &layout));
        }
        const int64_t history_kv_elems =
            kvh * (int64_t)history_capacity * dh;
        const int64_t image_bytes = stage_i8
            ? (int64_t)layout.n_bytes
            : 2 * history_kv_elems * (int64_t)sizeof(ggml_fp16_t);
        auto *gud = static_cast<hga_gpu_prefill_ud *>(
            gctx->res->alloc_custom_userdata(sizeof(hga_gpu_prefill_ud)));
        *gud = {
            sess,
            hga_layer_index(gctx->cparams, gctx->hparams, il),
            history_capacity,
            (int)n_q,
            stage_i8,
            true,
        };

        ggml_tensor *Q_cpu = Q, *K_cpu = K_rope, *V_cpu = V, *Kraw_cpu = K_raw;
        hga_prefill_stage_d2h_qkv(gctx, Q_cpu, K_cpu, V_cpu, Kraw_cpu);
        ggml_tensor *args[4] = {Q_cpu, K_cpu, V_cpu, Kraw_cpu};
        ggml_tensor *stage_cpu = ggml_custom_4d(
            gctx->ctx0, stage_i8 ? GGML_TYPE_I8 : GGML_TYPE_F16,
            stage_i8 ? image_bytes
                     : image_bytes / (int64_t)sizeof(ggml_fp16_t),
            1, 1, 1, args, 4,
            hga_gpu_prefill_stage_op, 1, gud);
        ggml_set_name(stage_cpu, "hga_gpu_stage_cpu_united");
        ggml_backend_sched_set_tensor_backend(gctx->sched, stage_cpu,
                                              gctx->backend_cpu);
        ggml_tensor *stage_gpu = ggml_new_tensor_1d(
            gctx->ctx0, stage_i8 ? GGML_TYPE_I8 : GGML_TYPE_F16,
            stage_i8 ? image_bytes
                     : image_bytes / (int64_t)sizeof(ggml_fp16_t));
        ggml_set_name(stage_gpu, "hga_gpu_stage_united");
        hga_pin_gpu(gctx->sched, stage_gpu);
        stage_gpu = ggml_cpy(gctx->ctx0, stage_cpu, stage_gpu);
        ggml_set_name(stage_gpu, "hga_gpu_stage_h2d_united");
        hga_pin_gpu(gctx->sched, stage_gpu);
        ggml_build_forward_expand(gctx->gf, stage_gpu);

        ggml_tensor *K_i8 = nullptr;
        ggml_tensor *V_i8 = nullptr;
        ggml_tensor *K_scale = nullptr;
        ggml_tensor *V_scale = nullptr;
        ggml_tensor *K_hist_f16 = nullptr;
        ggml_tensor *V_hist_f16 = nullptr;
        ggml_tensor *K = nullptr;
        ggml_tensor *Vf = nullptr;
        if (stage_i8) {
          K_i8 = hga_view_3d_as(
              gctx->ctx0, stage_gpu, GGML_TYPE_I8, dh, history_capacity, kvh,
              (size_t)dh, (size_t)history_capacity * (size_t)dh,
              layout.k_offset);
          V_i8 = hga_view_3d_as(
              gctx->ctx0, stage_gpu, GGML_TYPE_I8, dh, history_capacity, kvh,
              (size_t)dh, (size_t)history_capacity * (size_t)dh,
              layout.v_offset);
          K_scale = hga_view_3d_as(
              gctx->ctx0, stage_gpu, GGML_TYPE_F32, 1, history_capacity, kvh,
              sizeof(float), (size_t)history_capacity * sizeof(float),
              layout.k_scale_offset);
          V_scale = hga_view_3d_as(
              gctx->ctx0, stage_gpu, GGML_TYPE_F32, 1, history_capacity, kvh,
              sizeof(float), (size_t)history_capacity * sizeof(float),
              layout.v_scale_offset);
          K_hist_f16 = ggml_cast(gctx->ctx0, K_i8, GGML_TYPE_F16);
          ggml_set_name(K_hist_f16, "hga_prefill_K_i8_to_f16");
          V_hist_f16 = ggml_cast(gctx->ctx0, V_i8, GGML_TYPE_F16);
          ggml_set_name(V_hist_f16, "hga_prefill_V_i8_to_f16");
          K = ggml_mul(gctx->ctx0, K_hist_f16, K_scale);
          ggml_set_name(K, "hga_prefill_K_dequant_scale");
          Vf = ggml_mul(gctx->ctx0, V_hist_f16, V_scale);
          ggml_set_name(Vf, "hga_prefill_V_dequant_scale");
        } else {
          const size_t elem = sizeof(ggml_fp16_t);
          K = ggml_view_3d(
              gctx->ctx0, stage_gpu, dh, history_capacity, kvh,
              (size_t)dh * elem,
              (size_t)history_capacity * (size_t)dh * elem, 0);
          ggml_set_name(K, "hga_prefill_K_f16_history");
          Vf = ggml_view_3d(
              gctx->ctx0, stage_gpu, dh, history_capacity, kvh,
              (size_t)dh * elem,
              (size_t)history_capacity * (size_t)dh * elem,
              (size_t)history_kv_elems * elem);
          ggml_set_name(Vf, "hga_prefill_V_f16_history");
        }

        ggml_tensor *K_direct =
            ggml_permute(gctx->ctx0, K_rope, 0, 2, 1, 3);
        ggml_tensor *V_direct = ggml_permute(gctx->ctx0, V, 0, 2, 1, 3);
        /* K/V shape is invariant for every prefill ubatch. In particular the
         * final short suffix must not shrink flash-attention's K dimension and
         * create another compute allocation. CUDA's pad kernel accepts F32,
         * so pad before the existing F16 cast. The GPU causal mask below makes
         * all padded columns invisible. */
        if (n_q < direct_capacity) {
          K_direct = ggml_pad(gctx->ctx0, K_direct, 0,
                              direct_capacity - n_q, 0, 0);
          ggml_set_name(K_direct, "hga_prefill_K_direct_pad512_f32");
          V_direct = ggml_pad(gctx->ctx0, V_direct, 0,
                              direct_capacity - n_q, 0, 0);
          ggml_set_name(V_direct, "hga_prefill_V_direct_pad512_f32");
        }
        K_direct = ggml_cast(gctx->ctx0, K_direct, GGML_TYPE_F16);
        ggml_set_name(K_direct, "hga_prefill_K_direct_cast_transpose");
        V_direct = ggml_cast(gctx->ctx0, V_direct, GGML_TYPE_F16);
        ggml_set_name(V_direct, "hga_prefill_V_direct_cast_transpose");
        K = ggml_concat(gctx->ctx0, K, K_direct, 1);
        ggml_set_name(K, "hga_prefill_K_concat");
        Vf = ggml_concat(gctx->ctx0, Vf, V_direct, 1);
        ggml_set_name(Vf, "hga_prefill_V_concat");
        ggml_tensor *Qf = ggml_permute(gctx->ctx0, Q, 0, 2, 1, 3);
        ggml_tensor *mask = hga_gpu_prefill_causal_mask(
            gctx, hga_il, history_capacity, hga_input->history_valid, n_q,
            direct_capacity);

        if (stage_i8) {
          hga_pin_gpu(gctx->sched, K_i8);
          hga_pin_gpu(gctx->sched, V_i8);
          hga_pin_gpu(gctx->sched, K_scale);
          hga_pin_gpu(gctx->sched, V_scale);
          hga_pin_gpu(gctx->sched, K_hist_f16);
          hga_pin_gpu(gctx->sched, V_hist_f16);
        }
        hga_pin_gpu(gctx->sched, K);
        hga_pin_gpu(gctx->sched, Vf);
        hga_pin_gpu(gctx->sched, K_direct);
        hga_pin_gpu(gctx->sched, V_direct);
        hga_pin_gpu(gctx->sched, Qf);

        ggml_tensor *cur = ggml_flash_attn_ext(
            gctx->ctx0, Qf, K, Vf, mask, kq_scale, 0.0f, 0.0f);
        ggml_set_name(cur, "hga_prefill_flash_attn");
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        hga_pin_gpu(gctx->sched, cur);
        cur = ggml_reshape_2d(gctx->ctx0, cur, dh * n_heads, n_q);
        ggml_set_name(cur, "hga_gpu_attn_united");
        hga_pin_gpu(gctx->sched, cur);
        ggml_build_forward_expand(gctx->gf, cur);
        gctx->cb(cur, "hga_gpu_attn", il);

        static bool logged_gpu_prefill_united = false;
        if (!logged_gpu_prefill_united) {
          logged_gpu_prefill_united = true;
          std::fprintf(stderr,
                       "hga-gpu: PREFILL united %s KV + shared causal flash-attn hist-cap=%d valid=%lld total-cap=%d need=%d direct=%lld/%lld image=%.2f MiB heads=%lld/%lld\n",
                       stage_i8 ? "INT8" : "F16",
                       history_capacity,
                       (long long)std::min(graph_start, history_capacity),
                       total_capacity, need, (long long)n_q,
                       (long long)direct_capacity,
                       image_bytes / 1024.0 / 1024.0,
                       (long long)n_heads, (long long)kvh);
        }
        return cur;
      }
      const int64_t chunk = 64;
      const int64_t n_segments =
          (graph_start % chunk + n_q + chunk - 1) / chunk;
      const int64_t direct_visibility_elems = n_heads * n_q;
      int64_t image_bytes = 0;
      int64_t shape_off = 0;
      int64_t shape_pos = graph_start;
      int min_history_capacity = total_capacity;
      int max_history_capacity = 0;
      for (int64_t seg = 0; seg < n_segments; ++seg) {
        const int64_t seg_n =
            std::min(chunk - shape_pos % chunk, n_q - shape_off);
        const int capacity = hga_gpu_prefill_segment_history_capacity(
            sess, hga_il, graph_start, (int)shape_pos, (int)seg_n,
            total_capacity);
        if (stage_i8) {
          hga_gpu_prefill_i8_layout layout{};
          GGML_ASSERT(hga_gpu_prefill_i8_image_layout(
              sess, capacity, (int)n_q, &layout));
          image_bytes += (int64_t)layout.n_bytes;
        } else {
          const int64_t kv_elems = kvh * (int64_t)capacity * dh;
          const int64_t visibility_elems = n_heads * (int64_t)capacity;
          image_bytes += (2 * kv_elems + visibility_elems +
                          direct_visibility_elems) *
                         (int64_t)sizeof(ggml_fp16_t);
        }
        min_history_capacity = std::min(min_history_capacity, capacity);
        max_history_capacity = std::max(max_history_capacity, capacity);
        shape_off += seg_n;
        shape_pos += seg_n;
      }
      auto *gud = static_cast<hga_gpu_prefill_ud *>(
          gctx->res->alloc_custom_userdata(sizeof(hga_gpu_prefill_ud)));
      *gud = {
          sess,
          hga_layer_index(gctx->cparams, gctx->hparams, il),
          total_capacity,
          (int)n_q,
          stage_i8,
          false,
      };

      ggml_tensor *Q_cpu = Q, *K_cpu = K_rope, *V_cpu = V, *Kraw_cpu = K_raw;
      hga_prefill_stage_d2h_qkv(gctx, Q_cpu, K_cpu, V_cpu, Kraw_cpu);
      ggml_tensor *args[4] = {Q_cpu, K_cpu, V_cpu, Kraw_cpu};
      ggml_tensor *stage_cpu = ggml_custom_4d(
          gctx->ctx0, stage_i8 ? GGML_TYPE_I8 : GGML_TYPE_F16,
          stage_i8 ? image_bytes
                   : image_bytes / (int64_t)sizeof(ggml_fp16_t),
          1, 1, 1, args, 4,
          hga_gpu_prefill_stage_op, 1, gud);
      ggml_set_name(stage_cpu, "hga_gpu_stage_cpu");
      ggml_backend_sched_set_tensor_backend(gctx->sched, stage_cpu,
                                            gctx->backend_cpu);

      ggml_tensor *stage_gpu =
          ggml_new_tensor_1d(gctx->ctx0,
                             stage_i8 ? GGML_TYPE_I8 : GGML_TYPE_F16,
                             stage_i8
                                 ? image_bytes
                                 : image_bytes /
                                       (int64_t)sizeof(ggml_fp16_t));
      ggml_set_name(stage_gpu, "hga_gpu_stage");
      hga_pin_gpu(gctx->sched, stage_gpu);
      stage_gpu = ggml_cpy(gctx->ctx0, stage_cpu, stage_gpu);
      ggml_set_name(stage_gpu, "hga_gpu_stage_h2d");
      hga_pin_gpu(gctx->sched, stage_gpu);
      ggml_build_forward_expand(gctx->gf, stage_gpu);

      const size_t elem = sizeof(ggml_fp16_t);
      ggml_tensor *cur = nullptr;
      int64_t q_off = 0;
      int64_t pos = graph_start;
      size_t image_offset_bytes = 0;
      for (int64_t seg = 0; seg < n_segments; ++seg) {
        const int64_t seg_n = std::min(chunk - pos % chunk, n_q - q_off);
        const int capacity = hga_gpu_prefill_segment_history_capacity(
            sess, hga_il, graph_start, (int)pos, (int)seg_n,
            total_capacity);
        const int64_t kv_elems = kvh * (int64_t)capacity * dh;
        const int64_t visibility_elems = n_heads * (int64_t)capacity;
        hga_gpu_prefill_i8_layout i8_layout{};
        if (stage_i8) {
          GGML_ASSERT(hga_gpu_prefill_i8_image_layout(
              sess, capacity, (int)n_q, &i8_layout));
        }
        const size_t segment_bytes = stage_i8
            ? i8_layout.n_bytes
            : (size_t)(2 * kv_elems + visibility_elems +
                       direct_visibility_elems) * elem;
        const size_t base = image_offset_bytes;
        ggml_tensor *K = nullptr;
        ggml_tensor *Vf = nullptr;
        ggml_tensor *first_visible = nullptr;
        if (stage_i8) {
          ggml_tensor *K_i8 = hga_view_3d_as(
              gctx->ctx0, stage_gpu, GGML_TYPE_I8, dh, capacity, kvh,
              (size_t)dh, (size_t)capacity * (size_t)dh,
              base + i8_layout.k_offset);
          ggml_tensor *V_i8 = hga_view_3d_as(
              gctx->ctx0, stage_gpu, GGML_TYPE_I8, dh, capacity, kvh,
              (size_t)dh, (size_t)capacity * (size_t)dh,
              base + i8_layout.v_offset);
          ggml_tensor *K_scale = hga_view_3d_as(
              gctx->ctx0, stage_gpu, GGML_TYPE_F32, 1, capacity, kvh,
              sizeof(float), (size_t)capacity * sizeof(float),
              base + i8_layout.k_scale_offset);
          ggml_tensor *V_scale = hga_view_3d_as(
              gctx->ctx0, stage_gpu, GGML_TYPE_F32, 1, capacity, kvh,
              sizeof(float), (size_t)capacity * sizeof(float),
              base + i8_layout.v_scale_offset);
          K = ggml_mul(gctx->ctx0,
                       ggml_cast(gctx->ctx0, K_i8, GGML_TYPE_F16), K_scale);
          ggml_tensor *V_dequant = ggml_mul(
              gctx->ctx0, ggml_cast(gctx->ctx0, V_i8, GGML_TYPE_F16),
              V_scale);
          Vf = ggml_permute(gctx->ctx0, V_dequant, 1, 0, 2, 3);
          first_visible = hga_view_3d_as(
              gctx->ctx0, stage_gpu, GGML_TYPE_F16, capacity, 1, n_heads,
              (size_t)capacity * elem, (size_t)capacity * elem,
              base + i8_layout.visibility_offset);
          hga_pin_gpu(gctx->sched, K_i8);
          hga_pin_gpu(gctx->sched, V_i8);
          hga_pin_gpu(gctx->sched, K_scale);
          hga_pin_gpu(gctx->sched, V_scale);
          hga_pin_gpu(gctx->sched, V_dequant);
        } else {
          K = ggml_view_3d(
              gctx->ctx0, stage_gpu, dh, capacity, kvh, (size_t)dh * elem,
              (size_t)capacity * (size_t)dh * elem, base);
          Vf = ggml_view_3d(
              gctx->ctx0, stage_gpu, capacity, dh, kvh,
              (size_t)capacity * elem,
              (size_t)capacity * (size_t)dh * elem,
              base + (size_t)kv_elems * elem);
          first_visible = ggml_view_3d(
              gctx->ctx0, stage_gpu, capacity, 1, n_heads,
              (size_t)capacity * elem, (size_t)capacity * elem,
              base + (size_t)(2 * kv_elems) * elem);
        }
        const int64_t direct_len = q_off + seg_n;
        ggml_tensor *direct_first_visible = stage_i8
            ? hga_view_3d_as(
                  gctx->ctx0, stage_gpu, GGML_TYPE_F16, direct_len, 1,
                  n_heads, (size_t)n_q * elem, (size_t)n_q * elem,
                  base + i8_layout.direct_visibility_offset)
            : ggml_view_3d(
                  gctx->ctx0, stage_gpu, direct_len, 1, n_heads,
                  (size_t)n_q * elem, (size_t)n_q * elem,
                  base +
                      (size_t)(2 * kv_elems + visibility_elems) * elem);
        ggml_tensor *K_direct_src = ggml_view_3d(
            gctx->ctx0, K_rope, dh, kvh, direct_len,
            K_rope->nb[1], K_rope->nb[2], 0);
        ggml_tensor *V_direct_src = ggml_view_3d(
            gctx->ctx0, V, dh, kvh, direct_len,
            V->nb[1], V->nb[2], 0);
        ggml_tensor *K_direct =
            ggml_permute(gctx->ctx0, K_direct_src, 0, 2, 1, 3);
        ggml_tensor *V_direct =
            ggml_permute(gctx->ctx0, V_direct_src, 1, 2, 0, 3);
        K_direct = ggml_cast(gctx->ctx0, K_direct, GGML_TYPE_F16);
        V_direct = ggml_cast(gctx->ctx0, V_direct, GGML_TYPE_F16);
        K = ggml_concat(gctx->ctx0, K, K_direct, 1);
        Vf = ggml_concat(gctx->ctx0, Vf, V_direct, 0);
        first_visible = ggml_concat(
            gctx->ctx0, first_visible, direct_first_visible, 0);
        const int64_t n_keys = capacity + direct_len;
        ggml_tensor *Qs = ggml_view_3d(
            gctx->ctx0, Q, dh, n_heads, seg_n, Q->nb[1], Q->nb[2],
            (size_t)q_off * Q->nb[2]);
        ggml_tensor *Qf = ggml_permute(gctx->ctx0, Qs, 0, 2, 1, 3);
        hga_pin_gpu(gctx->sched, K);
        hga_pin_gpu(gctx->sched, Vf);
        hga_pin_gpu(gctx->sched, first_visible);
        hga_pin_gpu(gctx->sched, direct_first_visible);
        hga_pin_gpu(gctx->sched, K_direct);
        hga_pin_gpu(gctx->sched, V_direct);
        hga_pin_gpu(gctx->sched, Qf);

        /* Visibility inside one query chunk is cumulative, so CPU staging
         * carries one first-visible offset per (head,key), rather than a
         * 64-row F16 mask. Expand the compact thresholds on CUDA. */
        ggml_tensor *query_pos = ggml_repeat_4d(gctx->ctx0,
                                   ggml_reshape_3d(
                                       gctx->ctx0,
                                       ggml_arange(gctx->ctx0, 0.0f,
                                                   (float)seg_n, 1.0f),
                                       1, seg_n, 1),
                                   n_keys, seg_n, n_heads, 1);
        ggml_tensor *threshold =
            ggml_cast(gctx->ctx0, first_visible, GGML_TYPE_F32);
        ggml_tensor *visible = ggml_step(
            gctx->ctx0,
            ggml_scale_bias(gctx->ctx0,
                            ggml_sub(gctx->ctx0, query_pos, threshold),
                            1.0f, 0.5f));
        ggml_tensor *mask =
            ggml_scale_bias(gctx->ctx0, visible, 10000.0f, -10000.0f);
        hga_pin_gpu(gctx->sched, query_pos);
        hga_pin_gpu(gctx->sched, threshold);
        hga_pin_gpu(gctx->sched, visible);
        hga_pin_gpu(gctx->sched, mask);

        ggml_tensor *kq = ggml_mul_mat(gctx->ctx0, K, Qf);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        hga_pin_gpu(gctx->sched, kq);
        kq = ggml_soft_max_ext(gctx->ctx0, kq, mask, kq_scale, 0.0f);
        hga_pin_gpu(gctx->sched, kq);
        ggml_tensor *part = ggml_mul_mat(gctx->ctx0, Vf, kq);
        hga_pin_gpu(gctx->sched, part);
        part = ggml_permute(gctx->ctx0, part, 0, 2, 1, 3);
        part = ggml_cont_2d(gctx->ctx0, part, dh * n_heads, seg_n);
        hga_pin_gpu(gctx->sched, part);
        cur = cur ? ggml_concat(gctx->ctx0, cur, part, 1) : part;
        hga_pin_gpu(gctx->sched, cur);
        image_offset_bytes += segment_bytes;
        q_off += seg_n;
        pos += seg_n;
      }
      GGML_ASSERT(cur && q_off == n_q);
      ggml_set_name(cur, "hga_gpu_attn_2d");
      hga_pin_gpu(gctx->sched, cur);
      ggml_build_forward_expand(gctx->gf, cur);
      gctx->cb(cur, "hga_gpu_attn", il);

      static bool logged_gpu_prefill = false;
      if (!logged_gpu_prefill) {
        logged_gpu_prefill = true;
        std::fprintf(stderr,
                     "hga-gpu: PREFILL chunk/head routing enabled stage=%s hist-cap=%d..%d total-cap=%d need=%d direct=%lld segments=%lld image=%.2f MiB Q=%lld K/V-heads=%lld\n",
                     stage_i8 ? "i8+f32-scale" : "f16",
                     min_history_capacity, max_history_capacity,
                     total_capacity, need, (long long)n_q,
                     (long long)n_segments,
                     image_bytes / 1024.0 / 1024.0,
                     (long long)n_q, (long long)kvh);
      }
      return cur;
    }

    static bool logged_gpu_fallback = false;
    hga_input->layer = hga_il;
    hga_input->capacity = -1;
    hga_input->total_capacity = -1;
    hga_input->history_capacity = -1;
    hga_input->chunk_offset = -1;
    if (!logged_gpu_fallback) {
      logged_gpu_fallback = true;
      std::fprintf(stderr,
                   "hga-gpu: PREFILL fallback to CPU need=%d rounded=%d max=%d router/shape unsupported\n",
                   need, total_capacity, hga_gpu_prefill_max_keys());
    }
  }

  /* VERIFY keeps HGA routing/cache ownership on the CPU, but stages its one
   * shared selected historical key list to CUDA. The current short batch's
   * K/V is concatenated directly on device, then flash attention evaluates
   * softmax(Q*K^T)*V. A fixed worst-case history width preserves graph reuse
   * while the staged F16 mask hides unused slots. */
  if (phase == HGA_SWAP_VERIFY && Q->ne[2] > 1 && Q->ne[2] <= 8 &&
      hga_gpu_verify_enabled() && hga_sched_gpu_backend(gctx->sched) &&
      hga_session_config(sess)->prec == HGA_PREC_I8) {
    const int hga_il = hga_layer_index(gctx->cparams, gctx->hparams, il);
    const int64_t dh = Q->ne[0];
    const int64_t graph_n_q = Q->ne[2];
    const int64_t n_heads = Q->ne[1];
    const int64_t kvh = K_rope->ne[1];
    const int need = hga_gpu_verify_capacity(sess, hga_il, (int)graph_n_q);
    const int history_capacity = (need + 15) & ~15;
    if (need > 0 && history_capacity <= hga_gpu_verify_max_keys()) {
      hga_gpu_verify_i8_layout layout{};
      GGML_ASSERT(hga_gpu_verify_i8_image_layout(
          sess, history_capacity, (int)graph_n_q, &layout));

      /* V has no CPU normalization predecessor in the packed verify graph.
       * Materialize the small safe D2H copy before the CPU routing op. */
      V = hga_copy_to_cpu(gctx, V, "hga_verify_V_cpu");

      auto *vud = static_cast<hga_gpu_verify_ud *>(
          gctx->res->alloc_custom_userdata(sizeof(hga_gpu_verify_ud)));
      *vud = {
          sess,
          hga_il,
          history_capacity,
          (int)graph_n_q,
          phase,
          gctx->cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP,
      };
      ggml_tensor *args[4] = {Q, K_rope, V, K_raw};
      ggml_tensor *stage_cpu = ggml_custom_4d(
          gctx->ctx0, GGML_TYPE_I8, (int64_t)layout.n_bytes, 1, 1, 1, args, 4,
          hga_gpu_verify_stage_op, 1, vud);
      ggml_set_name(stage_cpu, "hga_verify_stage_cpu");
      ggml_backend_sched_set_tensor_backend(gctx->sched, stage_cpu,
                                            gctx->backend_cpu);

      ggml_tensor *stage_gpu = ggml_new_tensor_1d(
          gctx->ctx0, GGML_TYPE_I8, (int64_t)layout.n_bytes);
      ggml_set_name(stage_gpu, "hga_verify_stage_gpu");
      hga_pin_gpu(gctx->sched, stage_gpu);
      stage_gpu = ggml_cpy(gctx->ctx0, stage_cpu, stage_gpu);
      ggml_set_name(stage_gpu, "hga_verify_stage_h2d");
      hga_pin_gpu(gctx->sched, stage_gpu);
      ggml_build_forward_expand(gctx->gf, stage_gpu);

      ggml_tensor *K_i8 = hga_view_3d_as(
          gctx->ctx0, stage_gpu, GGML_TYPE_I8, dh, history_capacity, kvh,
          (size_t)dh, (size_t)history_capacity * (size_t)dh,
          layout.k_offset);
      ggml_tensor *V_i8 = hga_view_3d_as(
          gctx->ctx0, stage_gpu, GGML_TYPE_I8, dh, history_capacity, kvh,
          (size_t)dh, (size_t)history_capacity * (size_t)dh,
          layout.v_offset);
      ggml_tensor *K_scale = hga_view_3d_as(
          gctx->ctx0, stage_gpu, GGML_TYPE_F32, 1, history_capacity, kvh,
          sizeof(float), (size_t)history_capacity * sizeof(float),
          layout.k_scale_offset);
      ggml_tensor *V_scale = hga_view_3d_as(
          gctx->ctx0, stage_gpu, GGML_TYPE_F32, 1, history_capacity, kvh,
          sizeof(float), (size_t)history_capacity * sizeof(float),
          layout.v_scale_offset);
      ggml_tensor *mask = hga_view_2d_as(
          gctx->ctx0, stage_gpu, GGML_TYPE_F16,
          history_capacity + graph_n_q, graph_n_q,
          (size_t)(history_capacity + graph_n_q) * sizeof(ggml_fp16_t),
          layout.mask_offset);
      ggml_set_name(mask, "hga_verify_causal_mask");

      ggml_tensor *K_hist = ggml_mul(
          gctx->ctx0, ggml_cast(gctx->ctx0, K_i8, GGML_TYPE_F16), K_scale);
      ggml_set_name(K_hist, "hga_verify_K_dequant");
      ggml_tensor *V_hist = ggml_mul(
          gctx->ctx0, ggml_cast(gctx->ctx0, V_i8, GGML_TYPE_F16), V_scale);
      ggml_set_name(V_hist, "hga_verify_V_dequant");
      ggml_tensor *K_direct = ggml_cast(
          gctx->ctx0, ggml_permute(gctx->ctx0, K_rope, 0, 2, 1, 3),
          GGML_TYPE_F16);
      ggml_set_name(K_direct, "hga_verify_K_direct");
      ggml_tensor *V_direct = ggml_cast(
          gctx->ctx0, ggml_permute(gctx->ctx0, V, 0, 2, 1, 3),
          GGML_TYPE_F16);
      ggml_set_name(V_direct, "hga_verify_V_direct");
      ggml_tensor *K_all = ggml_concat(gctx->ctx0, K_hist, K_direct, 1);
      ggml_set_name(K_all, "hga_verify_K_concat");
      ggml_tensor *V_all = ggml_concat(gctx->ctx0, V_hist, V_direct, 1);
      ggml_set_name(V_all, "hga_verify_V_concat");
      ggml_tensor *Qf = ggml_permute(gctx->ctx0, Q, 0, 2, 1, 3);
      ggml_set_name(Qf, "hga_verify_Q_gpu");

      hga_pin_gpu(gctx->sched, K_i8);
      hga_pin_gpu(gctx->sched, V_i8);
      hga_pin_gpu(gctx->sched, K_scale);
      hga_pin_gpu(gctx->sched, V_scale);
      hga_pin_gpu(gctx->sched, mask);
      hga_pin_gpu(gctx->sched, K_hist);
      hga_pin_gpu(gctx->sched, V_hist);
      hga_pin_gpu(gctx->sched, K_direct);
      hga_pin_gpu(gctx->sched, V_direct);
      hga_pin_gpu(gctx->sched, K_all);
      hga_pin_gpu(gctx->sched, V_all);
      hga_pin_gpu(gctx->sched, Qf);

      ggml_tensor *cur = ggml_flash_attn_ext(
          gctx->ctx0, Qf, K_all, V_all, mask, kq_scale, 0.0f, 0.0f);
      ggml_set_name(cur, "hga_verify_flash_attn");
      ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
      hga_pin_gpu(gctx->sched, cur);
      cur = ggml_reshape_2d(gctx->ctx0, cur, dh * n_heads, graph_n_q);
      ggml_set_name(cur, "hga_verify_gpu_attn_2d");
      hga_pin_gpu(gctx->sched, cur);
      ggml_build_forward_expand(gctx->gf, cur);
      gctx->cb(cur, "hga_gpu_verify_attn", il);

      static bool logged_gpu_verify = false;
      if (!logged_gpu_verify) {
        logged_gpu_verify = true;
        std::fprintf(stderr,
                     "hga-gpu: VERIFY CPU routing + CUDA flash-attn history-cap=%d graph-nq=%lld image=%.2f MiB heads=%lld/%lld\n",
                     history_capacity, (long long)graph_n_q,
                     layout.n_bytes / 1024.0 / 1024.0,
                     (long long)n_heads, (long long)kvh);
      }
      return cur;
    }

    static bool logged_gpu_verify_fallback = false;
    if (!logged_gpu_verify_fallback) {
      logged_gpu_verify_fallback = true;
      std::fprintf(stderr,
                   "hga-gpu: VERIFY fallback to CPU need=%d rounded=%d max=%d\n",
                   need, history_capacity, hga_gpu_verify_max_keys());
    }
  }

  /* HGA is CPU-only. In decode/verify q/k-norm and RoPE already produce Q,
   * Krope and Kraw on CPU. V is the only GPU result with no CPU operation in
   * front of HGA, so it needs one explicit safe D2H. Wrapping the three CPU
   * tensors in another cpy created three extra scheduler splits per dense
   * layer. Prefill keeps its large activations on CUDA until HGA consumes
   * them. Do not pin GPU views directly to CPU: that aliases device memory. */
  /* MTP prompt catch-up is a multi-hundred-token ubatch. Decode D2H of that
   * Q/K/V is the contiguous-copy prefill bug (tens of MiB per eval). Treat
   * n_tok>8 like trunk PREFILL: leave activations on CUDA, ggml copies in. */
  const bool mtp_pf = gctx->cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
                      Q->ne[2] > 8;
  if (hga_decode_pack(phase) && !mtp_pf) {
    V = hga_copy_to_cpu(gctx, V, "hga_V_cpu");
  } else if (phase != HGA_SWAP_PREFILL && !mtp_pf) {
    ggml_backend_sched_set_tensor_backend(gctx->sched, Q, gctx->backend_cpu);
    ggml_backend_sched_set_tensor_backend(gctx->sched, K_rope,
                                          gctx->backend_cpu);
    ggml_backend_sched_set_tensor_backend(gctx->sched, V, gctx->backend_cpu);
    ggml_backend_sched_set_tensor_backend(gctx->sched, K_raw,
                                          gctx->backend_cpu);
  }

  ggml_build_forward_expand(gctx->gf, Q);
  ggml_build_forward_expand(gctx->gf, K_rope);
  ggml_build_forward_expand(gctx->gf, V);
  ggml_build_forward_expand(gctx->gf, K_raw);

  /* HGA owns routed KV on the CPU. llama cpy_k/cpy_v/kq_mask are not
   * expanded: they are unused, and kq_mask.n_kv forced VERIFY rebuilds. */

  const int64_t n_embd = Q->ne[0] * Q->ne[1];
  const int64_t n_tok = Q->ne[2];

  auto *ud = static_cast<hga_op_ud *>(
      gctx->res->alloc_custom_userdata(sizeof(hga_op_ud)));
  *ud = {
      (hga_session *)gctx->cparams.hga_runtime,
      hga_layer_index(gctx->cparams, gctx->hparams, il),
      gctx->cparams.hga_phase,
      gctx->cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP,
  };

  ggml_tensor *args[4] = {Q, K_rope, V, K_raw};
  const ggml_type output_type = hga_f16_transport_enabled() &&
                                        phase == HGA_SWAP_PREFILL
                                    ? GGML_TYPE_F16
                                    : GGML_TYPE_F32;
  ggml_tensor *cur = ggml_custom_4d(gctx->ctx0, output_type, n_embd, n_tok, 1,
                                    1, args, 4, hga_custom_op, 1, ud);

  ggml_set_name(cur, "hga_attn");
  ggml_backend_sched_set_tensor_backend(gctx->sched, cur, gctx->backend_cpu);
  ggml_build_forward_expand(gctx->gf, cur);
  gctx->cb(cur, "hga_attn", il);
  return cur;
}

static uint32_t g_hga_pad_n_real = 0;

uint32_t hga_ubatch_padded_n_real() {
    return g_hga_pad_n_real;
}

void hga_ubatch_pad_reset() {
    g_hga_pad_n_real = 0;
}

bool hga_ubatch_pad_to(llama_ubatch & ub, uint32_t n_pad_to) {
    if (ub.n_tokens == 0 || ub.n_tokens >= n_pad_to || !ub.data || ub.n_seqs != 1) {
        return false;
    }
    auto & d = *ub.data;
    const uint32_t n0 = ub.n_tokens;
    const uint32_t n_pos = ub.n_pos ? ub.n_pos : 1u;

    const llama_token tok = (ub.token && n0) ? ub.token[n0 - 1] : 0;
    const int32_t n_seq = (ub.n_seq_id && n0) ? ub.n_seq_id[n0 - 1] : 1;
    std::vector<llama_seq_id> last_seqs;
    if (ub.seq_id && n_seq > 0) {
        last_seqs.assign(ub.seq_id[n0 - 1], ub.seq_id[n0 - 1] + n_seq);
    } else {
        last_seqs.push_back(0);
    }

    std::vector<llama_pos> pos_old = d.pos;
    d.token.resize(n_pad_to);
    d.n_seq_id.resize(n_pad_to);
    d.output.resize(n_pad_to);
    d.seq_id.resize(n_pad_to);
    d.pos.assign((size_t) n_pad_to * n_pos, 0);

    for (uint32_t j = 0; j < n_pos; ++j) {
        llama_pos last = 0;
        for (uint32_t i = 0; i < n0; ++i) {
            last = pos_old[j * n0 + i];
            d.pos[j * n_pad_to + i] = last;
        }
        for (uint32_t i = n0; i < n_pad_to; ++i) {
            d.pos[j * n_pad_to + i] = last + llama_pos(i - n0 + 1);
        }
    }
    if (!d.embd.empty() && n0 > 0) {
        const size_t n_embd = d.embd.size() / n0;
        d.embd.resize((size_t) n_pad_to * n_embd);
        const float * last = d.embd.data() + (size_t) (n0 - 1) * n_embd;
        for (uint32_t i = n0; i < n_pad_to; ++i) {
            std::memcpy(d.embd.data() + (size_t) i * n_embd, last, n_embd * sizeof(float));
        }
        ub.embd = d.embd.data();
    }
    for (uint32_t i = n0; i < n_pad_to; ++i) {
        d.token[i] = tok;
        d.n_seq_id[i] = (int32_t) last_seqs.size();
        /* Prefill dummy n=2→n_ubatch: pads must not be outputs or lm_head
         * runs on 512 tokens and OOMs. Generate pads go through
         * hga_maybe_pad_decode_batch and copy logits flags. */
        d.output[i] = 0;
        for (llama_seq_id s : last_seqs) {
            d.seq_id_data.push_back(s);
        }
    }
    llama_seq_id * ptr = d.seq_id_data.data();
    for (uint32_t i = 0; i < n_pad_to; ++i) {
        d.seq_id[i] = ptr;
        ptr += d.n_seq_id[i] > 0 ? (uint32_t) d.n_seq_id[i] : 0;
    }

    ub.token    = d.token.data();
    ub.pos      = d.pos.data();
    ub.n_seq_id = d.n_seq_id.data();
    ub.seq_id   = d.seq_id.data();
    ub.output   = d.output.data();
    ub.n_tokens = n_pad_to;
    ub.n_seq_tokens = n_pad_to;
    g_hga_pad_n_real = n0;

    fprintf(stderr, "hga-swap: pad ubatch %u -> %u tokens (graph = n_ubatch)\n",
            n0, n_pad_to);
    return true;
}

void hga_ubatch_prefill_one_output(llama_ubatch & ub) {
    if (!ub.output || ub.n_tokens == 0) {
        return;
    }
    uint32_t n_real = g_hga_pad_n_real;
    if (n_real == 0 || n_real > ub.n_tokens) {
        n_real = ub.n_tokens;
    }
    for (uint32_t i = 0; i < ub.n_tokens; ++i) {
        ub.output[i] = 0;
    }
    ub.output[n_real - 1] = 1;
}

bool hga_maybe_pad_decode_batch(const llama_cparams & cparams,
                                const llama_batch & in,
                                uint32_t n_embd,
                                hga_llama_batch_pad & mem) {
    if (!cparams.hga_seen_large_prefill) {
        return false;
    }
    if (cparams.n_ubatch == 0 || cparams.n_ubatch > 16) {
        return false;
    }
    if (in.n_tokens <= 0 || (uint32_t) in.n_tokens >= cparams.n_ubatch) {
        return false;
    }
    const uint32_t n0 = (uint32_t) in.n_tokens;
    const uint32_t n_pad_to = cparams.n_ubatch;

    mem.token.resize(n_pad_to);
    mem.pos.resize(n_pad_to);
    mem.n_seq_id.resize(n_pad_to);
    mem.seq_id.resize(n_pad_to);
    mem.logits.resize(n_pad_to, 0);
    mem.seq_id_data.resize(n_pad_to, 0);

    llama_token last_tok = 0;
    llama_pos last_pos = 0;
    int8_t last_logits = 1;
    llama_seq_id last_seq = 0;
    for (uint32_t i = 0; i < n0; ++i) {
        if (in.token) {
            mem.token[i] = in.token[i];
            last_tok = in.token[i];
        }
        if (in.pos) {
            mem.pos[i] = in.pos[i];
            last_pos = in.pos[i];
        } else {
            mem.pos[i] = (llama_pos) i;
            last_pos = (llama_pos) i;
        }
        if (in.logits) {
            mem.logits[i] = in.logits[i];
            last_logits = in.logits[i];
        } else {
            mem.logits[i] = 1;
        }
        if (in.n_seq_id && in.seq_id && in.n_seq_id[i] > 0) {
            last_seq = in.seq_id[i][0];
        }
        mem.n_seq_id[i] = 1;
        mem.seq_id_data[i] = last_seq;
        mem.seq_id[i] = &mem.seq_id_data[i];
    }
    for (uint32_t i = n0; i < n_pad_to; ++i) {
        mem.token[i] = last_tok;
        mem.pos[i] = last_pos + llama_pos(i - n0 + 1);
        mem.logits[i] = last_logits;
        mem.n_seq_id[i] = 1;
        mem.seq_id_data[i] = last_seq;
        mem.seq_id[i] = &mem.seq_id_data[i];
    }
    if (in.embd && n_embd > 0) {
        mem.embd.resize((size_t) n_pad_to * n_embd);
        std::memcpy(mem.embd.data(), in.embd, (size_t) n0 * n_embd * sizeof(float));
        const float * last = mem.embd.data() + (size_t) (n0 - 1) * n_embd;
        for (uint32_t i = n0; i < n_pad_to; ++i) {
            std::memcpy(mem.embd.data() + (size_t) i * n_embd, last, n_embd * sizeof(float));
        }
    }

    mem.batch = in;
    mem.batch.n_tokens = (int32_t) n_pad_to;
    mem.batch.token    = in.token ? mem.token.data() : nullptr;
    mem.batch.embd     = in.embd  ? mem.embd.data()  : nullptr;
    mem.batch.pos      = mem.pos.data();
    mem.batch.n_seq_id = mem.n_seq_id.data();
    mem.batch.seq_id   = mem.seq_id.data();
    mem.batch.logits   = mem.logits.data();
    g_hga_pad_n_real   = n0;

    fprintf(stderr, "hga-swap: pad llama_batch %u -> %u tokens (generate graph)\n",
            n0, n_pad_to);
    return true;
}
