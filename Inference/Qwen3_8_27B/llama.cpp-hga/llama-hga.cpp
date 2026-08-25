#include "llama-hga.h"

#include "hga.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-hparams.h"
#include "llama-kv-cache.h"
#include "llama.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

/* llama_cparams is extended by apply_hga.py with:
 *   bool hga_enabled; int hga_levels; int hga_chunk_size; int hga_group_size;
 *   int hga_keep_first; int hga_keep_last; float hga_frac_l1; float hga_frac_l2;
 *   bool hga_i8; void * hga_runtime;
 */

static int hga_layer_index(const llama_hparams & hparams, int il) {
    int k = 0;
    for (int i = 0; i < il; ++i) {
        if (!hparams.is_recr(i)) ++k;
    }
    return k;
}

static int hga_n_full_layers(const llama_hparams & hparams) {
    int n = 0;
    for (int i = 0; i < (int) hparams.n_layer(); ++i) {
        if (!hparams.is_recr(i)) ++n;
    }
    return n;
}

void hga_cparams_from_ctx_params(llama_cparams & cparams, const llama_context_params & params) {
    cparams.hga_enabled     = params.hga_enabled;
    cparams.hga_levels      = params.hga_levels;
    cparams.hga_chunk_size  = params.hga_chunk_size;
    cparams.hga_group_size  = params.hga_group_size;
    cparams.hga_keep_first  = params.hga_keep_first;
    cparams.hga_keep_last   = params.hga_keep_last;
    cparams.hga_frac_l1     = params.hga_frac_l1;
    cparams.hga_frac_l2     = params.hga_frac_l2;
    cparams.hga_i8          = params.hga_i8;
    cparams.hga_runtime     = nullptr;
}

void hga_runtime_init(llama_cparams & cparams, const llama_hparams & hparams) {
    if (!cparams.hga_enabled) {
        cparams.hga_runtime = nullptr;
        return;
    }
    const int n_full = hga_n_full_layers(hparams);
    if (n_full <= 0) {
        fprintf(stderr, "hga: no full-attention layers; disabling\n");
        cparams.hga_enabled = false;
        cparams.hga_runtime = nullptr;
        return;
    }
    hga_config cfg = hga_config_qwen38_27b(cparams.hga_levels, (int) cparams.n_ctx, (int) cparams.n_threads);
    cfg.n_q_heads   = (int) hparams.n_head();
    cfg.n_kv_heads  = (int) hparams.n_head_kv();
    cfg.head_dim    = (int) hparams.n_embd_head_k();
    cfg.rotary_dim  = std::min((int) hparams.n_rot(), cfg.head_dim);
    cfg.chunk_size  = cparams.hga_chunk_size  > 0 ? cparams.hga_chunk_size  : 64;
    cfg.group_size  = cparams.hga_group_size  > 0 ? cparams.hga_group_size  : 16;
    cfg.keep_first  = cparams.hga_keep_first;
    cfg.keep_last   = cparams.hga_keep_last;
    cfg.frac_l1     = cparams.hga_frac_l1;
    cfg.frac_l2     = cparams.hga_frac_l2;
    cfg.levels      = cparams.hga_levels;
    cfg.max_seq     = (int) cparams.n_ctx;
    cfg.n_threads   = std::max(1, (int) cparams.n_threads);
    cfg.prec        = cparams.hga_i8 ? HGA_PREC_I8 : HGA_PREC_F16;
    hga_session * sess = hga_session_create(&cfg, n_full);
    cparams.hga_runtime = sess;
    fprintf(stderr, "hga: enabled  levels=%d  full-attn layers=%d  chunk=%d group=%d  "
            "keep_first=%d keep_last=%d  frac_l1=%.3f frac_l2=%.3f  prec=%s  heads=%d/%d dh=%d rd=%d  ctx=%d threads=%d\n",
            cfg.levels, n_full, cfg.chunk_size, cfg.group_size, cfg.keep_first, cfg.keep_last,
            cfg.frac_l1, cfg.frac_l2, cfg.prec == HGA_PREC_I8 ? "i8" : "f16",
            cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rotary_dim,
            cfg.max_seq, cfg.n_threads);
}

void hga_runtime_free(llama_cparams & cparams) {
    if (cparams.hga_runtime) {
        hga_session_free((hga_session *) cparams.hga_runtime);
        cparams.hga_runtime = nullptr;
    }
}

class llm_graph_input_hga : public llm_graph_input_i {
public:
    hga_session * sess = nullptr;
    void set_input(const llama_ubatch * ubatch) override {
        if (!sess || !ubatch) return;
        const int start = (ubatch->pos && ubatch->n_tokens > 0) ? (int) ubatch->pos[0] : 0;
        hga_set_ubatch(sess, start, (int) ubatch->n_tokens);
    }
    bool can_reuse(const llm_graph_params & params) override {
        GGML_UNUSED(params);
        return true;
    }
};

static hga_dtype hga_from_ggml(ggml_type t) {
    if (t == GGML_TYPE_F16) return HGA_F16;
    return HGA_F32;
}

static void pack_heads(const ggml_tensor * t, int n_heads, int n_tok, int dh, std::vector<float> & out) {
    /* ggml layout [dh, n_heads, n_tok] -> HGA [n_heads, n_tok, dh] */
    out.resize((size_t) n_heads * (size_t) n_tok * (size_t) dh);
    const char * data = (const char *) t->data;
    const size_t ts = ggml_type_size(t->type);
    for (int h = 0; h < n_heads; ++h) {
        for (int k = 0; k < n_tok; ++k) {
            for (int d = 0; d < dh; ++d) {
                const char * p = data + (size_t) d * t->nb[0] + (size_t) h * t->nb[1] + (size_t) k * t->nb[2];
                float v;
                if (t->type == GGML_TYPE_F16) {
                    v = ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
                } else {
                    memcpy(&v, p, sizeof(float));
                }
                out[((size_t) h * (size_t) n_tok + (size_t) k) * (size_t) dh + (size_t) d] = v;
            }
        }
    }
    (void) ts;
}

struct hga_op_ud {
    llama_cparams * cparams;
    const llama_hparams * hparams;
    int il;
};

static void hga_custom_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    /* Single-threaded op: ith==0 does the work. ggml may still split; ignore extra tasks. */
    if (ith != 0) return;
    GGML_UNUSED(nth);
    auto * ud = (hga_op_ud *) userdata;
    hga_session * sess = (hga_session *) ud->cparams->hga_runtime;
    if (!sess) return;

    ggml_tensor * Q     = dst->src[0];
    ggml_tensor * Krope = dst->src[1];
    ggml_tensor * V     = dst->src[2];
    ggml_tensor * Kraw  = dst->src[3];

    const int dh  = (int) Q->ne[0];
    const int H   = (int) Q->ne[1];
    const int n_q = (int) Q->ne[2];
    const int KVH = (int) Krope->ne[1];
    const int start = hga_ubatch_start(sess);
    const int hga_il = hga_layer_index(*ud->hparams, ud->il);

    std::vector<float> q, k, v, kraw, out((size_t) H * (size_t) n_q * (size_t) dh);
    pack_heads(Q, H, n_q, dh, q);
    pack_heads(Krope, KVH, n_q, dh, k);
    pack_heads(V, KVH, n_q, dh, v);
    if (Kraw) pack_heads(Kraw, KVH, n_q, dh, kraw);
    else kraw = k;

    hga_stats st{};
    hga_forward(sess, hga_il, start, n_q,
                q.data(), k.data(), kraw.data(), v.data(), HGA_F32, out.data(), &st);

    /* dst is F32 [H*dh, n_q] contiguous */
    float * d = (float *) dst->data;
    for (int t = 0; t < n_q; ++t) {
        for (int h = 0; h < H; ++h) {
            memcpy(d + (size_t) t * (size_t) (H * dh) + (size_t) h * (size_t) dh,
                   out.data() + ((size_t) h * (size_t) n_q + (size_t) t) * (size_t) dh,
                   (size_t) dh * sizeof(float));
        }
    }
    (void) hga_from_ggml;
}

ggml_tensor * hga_build_full_attn(
        llm_graph_context * gctx,
        llm_graph_input_attn_kv * inp,
        ggml_tensor * Q,
        ggml_tensor * K_rope,
        ggml_tensor * V,
        ggml_tensor * K_raw,
        float kq_scale,
        int il) {
    GGML_UNUSED(kq_scale);
    hga_session * sess = (hga_session *) gctx->cparams.hga_runtime;
    if (!sess) {
        fprintf(stderr, "hga: missing runtime at layer %d\n", il);
        return Q;
    }

    /* Register a graph input once so start_pos is refreshed every eval. */
    {
        auto inp = std::make_unique<llm_graph_input_hga>();
        inp->sess = sess;
        gctx->res->add_input(std::move(inp));
    }

    Q      = ggml_cont(gctx->ctx0, Q);
    K_rope = ggml_cont(gctx->ctx0, K_rope);
    V      = ggml_cont(gctx->ctx0, V);
    if (K_raw) K_raw = ggml_cont(gctx->ctx0, K_raw);
    else K_raw = K_rope;

    ggml_build_forward_expand(gctx->gf, Q);
    ggml_build_forward_expand(gctx->gf, K_rope);
    ggml_build_forward_expand(gctx->gf, V);
    ggml_build_forward_expand(gctx->gf, K_raw);

    /* Keep llama.cpp's hybrid KV graph inputs live (k_idxs/v_idxs). Skipping
     * this leaves those tensors unallocated and crashes in set_input. */
    ggml_tensor * kq_mask = nullptr;
    if (inp && inp->mctx) {
        ggml_build_forward_expand(gctx->gf, inp->mctx->cpy_k(gctx->ctx0, K_rope, inp->get_k_idxs(), il));
        ggml_build_forward_expand(gctx->gf, inp->mctx->cpy_v(gctx->ctx0, V,      inp->get_v_idxs(), il));
        kq_mask = inp->get_kq_mask();
        if (kq_mask) {
            /* Keep the hybrid KQ mask input allocated even though HGA does not
             * consume it — otherwise set_input_kq_mask hits a null buffer. */
            ggml_build_forward_expand(gctx->gf, kq_mask);
        }
    }

    const int64_t n_embd = Q->ne[0] * Q->ne[1];
    const int64_t n_tok  = Q->ne[2];

    static hga_op_ud uds[256];
    GGML_ASSERT(il >= 0 && il < 256);
    hga_op_ud & ud = uds[il];
    ud.cparams = const_cast<llama_cparams *>(&gctx->cparams);
    ud.hparams = &gctx->hparams;
    ud.il = il;

    ggml_tensor * args[4] = { Q, K_rope, V, K_raw };
    ggml_tensor * cur = ggml_custom_4d(
        gctx->ctx0, GGML_TYPE_F32, n_embd, n_tok, 1, 1,
        args, 4, hga_custom_op, 1, &ud);

    ggml_set_name(cur, "hga_attn");
    ggml_backend_sched_set_tensor_backend(gctx->sched, cur, gctx->backend_cpu);
    ggml_build_forward_expand(gctx->gf, cur);
    gctx->cb(cur, "hga_attn", il);
    return cur;
}
