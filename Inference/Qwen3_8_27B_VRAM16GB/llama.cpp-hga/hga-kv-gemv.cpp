#include "hga-kv-gemv.h"
#include "llama-hga.h"

#include "hga.h"
#include "hga_l2.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-hparams.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int hga_dense_index(const llama_hparams & hparams, int il) {
    int k = 0;
    for (int i = 0; i < il; ++i) {
        if (!hparams.is_recr(i)) ++k;
    }
    return k;
}

struct kv_gemv_ud {
    hga_session * sess;
    int hga_il;
};

/* Graph-build recording so the first decode GEMV can repack every dense layer. */
static ggml_tensor * g_wk[64];
static ggml_tensor * g_wv[64];
static const void * g_wk_data[64];
static const void * g_wv_data[64];
static int g_n_rec = 0;
static bool g_bound = false;

static hga_l2_dequant_fn hga_type_dequant(ggml_type type) {
    const ggml_type_traits * tt = ggml_get_type_traits(type);
    if (!tt || !tt->to_float) {
        return nullptr;
    }
    return (hga_l2_dequant_fn) tt->to_float;
}

static void hga_l2_bind_tensor_pair(hga_l2_plan * plan, int hga_il, ggml_tensor * wk, ggml_tensor * wv) {
    if (!plan || !wk || !wv || !wk->data || !wv->data) return;
    hga_l2_dequant_fn dk = hga_type_dequant(wk->type);
    hga_l2_dequant_fn dv = hga_type_dequant(wv->type);
    if (!dk || !dv) {
        fprintf(stderr, "hga-l2: no to_float for %s/%s — skip bind layer %d\n",
                ggml_type_name(wk->type), ggml_type_name(wv->type), hga_il);
        return;
    }
    const int wk_bn = (int) ggml_blck_size(wk->type);
    const int wv_bn = (int) ggml_blck_size(wv->type);
    hga_l2_bind_weights(
        plan, hga_il,
        wk->data, wk->nb[1], ggml_type_size(wk->type), wk_bn, dk,
        wv->data, wv->nb[1], ggml_type_size(wv->type), wv_bn, dv);
}

static void hga_l2_bind_all(hga_l2_plan * plan) {
    if (g_bound || !plan) return;
    for (int i = 0; i < g_n_rec; ++i) {
        hga_l2_bind_tensor_pair(plan, i, g_wk[i], g_wv[i]);
    }
    g_bound = true;
    fprintf(stderr, "hga-l2: bound %d dense K/V layers into per-core slices\n", g_n_rec);
}

static void kv_gemv_account(double ms) {
    static double tok_ms = 0, sum_ms = 0;
    static int layer_i = 0, n_tok = 0;
    tok_ms += ms;
    layer_i++;
    if (layer_i == 16) {
        n_tok++;
        sum_ms += tok_ms;
        if (n_tok <= 3 || n_tok % 16 == 0) {
            fprintf(stderr, "hga-prof decode tok %d: KV GEMV 16 layers = %.2f ms  (mean %.2f)\n",
                    n_tok, tok_ms, sum_ms / n_tok);
        }
        tok_ms = 0;
        layer_i = 0;
    }
}

static void hga_kv_gemv_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    if (ith != 0) return;
    GGML_UNUSED(nth);
    auto * ud = (kv_gemv_ud *) userdata;
    hga_session * sess = ud->sess;
    hga_l2_plan * plan = sess ? (hga_l2_plan *) hga_session_l2(sess) : nullptr;
    if (!plan) return;

    ggml_tensor * x  = dst->src[0];
    ggml_tensor * wk = dst->src[1];
    ggml_tensor * wv = dst->src[2];
    if (!x || !x->data || !wk || !wv) return;

    const int hga_il = ud->hga_il;
    const int n_k = (int) wk->ne[1];
    const int n_v = (int) wv->ne[1];
    const int n_tok = (int) x->ne[1];
    const int n_embd = (int) x->ne[0];
    if (n_k + n_v != (int) dst->ne[0]) return;

    hga_l2_bind_all(plan);

    const auto t0 = std::chrono::steady_clock::now();
    float * out = (float *) dst->data;
    const float * xptr = (const float *) x->data;
    const size_t x_tok = x->nb[1] / sizeof(float);
    const size_t o_tok = dst->nb[1] / sizeof(float);
    const bool x_contig = x->nb[0] == sizeof(float) && x_tok == (size_t) n_embd;
    const char * xrow = x_contig ? nullptr : (const char *) x->data;
    float * krow = out;

    float * krow_end = out + (size_t) n_tok * o_tok;
    for (; krow < krow_end; krow += o_tok) {
        /* Hidden may not be 5120-contiguous if a view; copy one token. */
        const float * xt = xptr;
        static thread_local std::vector<float> xcontig;
        if (!x_contig) {
            xcontig.resize((size_t) n_embd);
            if (x->nb[0] == sizeof(float)) {
                std::memcpy(xcontig.data(), xrow, (size_t) n_embd * sizeof(float));
            } else {
                const char * p = xrow;
                const char * pend = xrow + (size_t) n_embd * x->nb[0];
                float * d = xcontig.data();
                for (; p < pend; p += x->nb[0], ++d)
                    *d = *(const float *) p;
            }
            xt = xcontig.data();
            xrow += x->nb[1];
        } else {
            xptr += x_tok;
        }
        float * kt = krow;
        float * vt = kt + n_k;
        (void) n_v;
        hga_l2_gemv(plan, hga_il, xt, kt, vt);
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (n_tok == 1) kv_gemv_account(ms);
}

ggml_tensor * hga_build_kv_proj(
        llm_graph_context * gctx,
        ggml_tensor * x,
        ggml_tensor * wk,
        ggml_tensor * wv,
        int il) {
    if (!gctx || !x || !wk || !wv) return nullptr;
    /* VERIFY reuses the fixed decode graph and is the latency-critical
     * 1..3-token path.  It must use the same CPU K/V projection as DECODE;
     * otherwise every dense layer produces V on CUDA and inserts a separate
     * GPU→CPU dependency immediately before HGA. */
    if (!hga_decode_pack((int32_t) gctx->cparams.hga_phase)) return nullptr;
    if (!gctx->cparams.hga_enabled || !gctx->cparams.hga_runtime) return nullptr;
    if (getenv("HGA_L2_OFF")) return nullptr;

    hga_session * sess = (hga_session *) gctx->cparams.hga_runtime;
    if (!hga_session_l2(sess)) return nullptr;

    const int64_t n_tok = x->ne[1] > 0 ? x->ne[1] : 1;

    const int hga_il = hga_dense_index(gctx->hparams, il);
    if (hga_il < 0 || hga_il >= 64) return nullptr;
    if (g_wk_data[hga_il] != wk->data || g_wv_data[hga_il] != wv->data) {
        g_bound = false;
    }
    g_wk[hga_il] = wk;
    g_wv[hga_il] = wv;
    g_wk_data[hga_il] = wk->data;
    g_wv_data[hga_il] = wv->data;
    if (hga_il + 1 > g_n_rec) g_n_rec = hga_il + 1;

    auto * ud = static_cast<kv_gemv_ud *>(
        gctx->res->alloc_custom_userdata(sizeof(kv_gemv_ud)));
    *ud = { sess, hga_il };

    const int64_t n_k = wk->ne[1];
    const int64_t n_v = wv->ne[1];
    ggml_tensor * args[3] = { x, wk, wv };
    ggml_tensor * KV = ggml_custom_4d(
        gctx->ctx0, GGML_TYPE_F32, n_k + n_v, n_tok, 1, 1,
        args, 3, hga_kv_gemv_op, 1, ud);
    ggml_set_name(KV, "hga_kv_gemv");
    ggml_backend_sched_set_tensor_backend(gctx->sched, KV, gctx->backend_cpu);
    ggml_build_forward_expand(gctx->gf, KV);
    gctx->cb(KV, "hga_kv_gemv", il);
    return KV;
}

bool hga_build_kv_gemv_pair(
        llm_graph_context * gctx,
        ggml_tensor * wk,
        ggml_tensor * wv,
        ggml_tensor * x,
        ggml_tensor ** Kcur,
        ggml_tensor ** Vcur,
        int il) {
    if (!Kcur || !Vcur) return false;
    *Kcur = nullptr;
    *Vcur = nullptr;
    ggml_tensor * KV = hga_build_kv_proj(gctx, x, wk, wv, il);
    if (!KV) return false;
    const int64_t n_k = wk->ne[1];
    const int64_t n_v = wv->ne[1];
    const int64_t n_tok = KV->ne[1];
    *Kcur = ggml_view_2d(gctx->ctx0, KV, n_k, n_tok, KV->nb[1], 0);
    *Vcur = ggml_view_2d(gctx->ctx0, KV, n_v, n_tok, KV->nb[1],
                         n_k * ggml_element_size(KV));
    return *Kcur && *Vcur;
}

void hga_l2_after_hga(hga_session * sess, int hga_il) {
    if (!sess) return;
    hga_l2_plan * plan = (hga_l2_plan *) hga_session_l2(sess);
    if (!plan) return;
    const int nL = hga_session_n_layers(sess);
    int next = hga_il + 1;
    if (next >= nL) next = 0;
    const int * keys = nullptr;
    int n_keys = 0;
    hga_last_keys(sess, &keys, &n_keys);
    hga_l2_kick_prefetch(plan, next, sess, keys, n_keys);
}

void hga_l3_prefetch_layer(hga_session * worker_sess,
                           const hga_session * data_sess, int hga_il) {
    if (!worker_sess || !data_sess || hga_il < 0 ||
        hga_il >= hga_session_n_layers(data_sess)) return;
    hga_l2_plan * plan =
        (hga_l2_plan *) hga_session_l2(worker_sess);
    if (!plan) return;

    static const int workers = [] {
        const char * v = std::getenv("HGA_L3_PREFETCH_THREADS");
        return std::max(1, v ? std::atoi(v) : 2);
    }();
    static const size_t budget_bytes = [] {
        const char * v = std::getenv("HGA_L3_PREFETCH_MIB");
        const int mib = std::max(1, v ? std::atoi(v) : 24);
        return (size_t) mib * 1024u * 1024u;
    }();
    static thread_local std::vector<int> keys;
    const int n_kv = hga_session_n_kv(data_sess, hga_il);
    keys.resize((size_t)std::max(0, n_kv));
    const int n_keys = hga_prefetch_keys(data_sess, hga_il, budget_bytes,
                                         keys.data(), (int)keys.size());
    if (n_keys <= 0) return;
    hga_l3_kick_kv_prefetch(plan, hga_il, data_sess, keys.data(), n_keys,
                            workers);
    static int kicks = 0;
    ++kicks;
    if (kicks <= 4 || kicks % 64 == 0) {
        fprintf(stderr,
                "hga-l3: kick=%d layer=%d keys=%d/%d budget=%zuMiB workers=%d\n",
                kicks, hga_il, n_keys, n_kv, budget_bytes >> 20, workers);
    }
}
