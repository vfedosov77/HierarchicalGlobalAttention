#!/usr/bin/env python3
"""Copy HGA sources into a llama.cpp checkout and apply the graph/CLI hooks."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

# ggml-org/llama.cpp v0.3.0 is an annotated tag on this commit, which is also
# nightly tag b10621. `git describe --tags --exact-match` therefore reports
# b10621 after a shallow clone of v0.3.0 (the annotated tag object is not a
# commit). Pin by SHA so a correct checkout is not rejected.
HGA_LLAMA_TAG = "v0.3.0"
HGA_LLAMA_SHA = "c1d0e7a004015f23bc0233470b747b596f29b264"


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _git(root: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def assert_llama_release_pin(root: Path) -> None:
    """Accept HEAD if it is llama.cpp v0.3.0, even when describe prints b10621.

    Copies with no .git (offline host) are not checked here.
    """
    if not (root / ".git").exists():
        return
    head = _git(root, "rev-parse", "HEAD")
    if not head:
        return
    tag_sha = _git(root, "rev-parse", "-q", "--verify", f"refs/tags/{HGA_LLAMA_TAG}^{{commit}}")
    if head == HGA_LLAMA_SHA or (tag_sha and head == tag_sha):
        extra = _git(root, "describe", "--tags", "--exact-match")
        note = f", git describe={extra}" if extra and extra != HGA_LLAMA_TAG else ""
        print(f"  pinned: {HGA_LLAMA_TAG} ({head[:12]}{note})")
        return
    observed = _git(root, "describe", "--tags", "--exact-match")
    if not observed:
        print(f"  NOTE: {root} is a git repo but has no exact-release tag.", file=sys.stderr)
        return
    die(
        f"{root} is at '{observed}' ({head[:12]}) but HGA is pinned to llama.cpp "
        f"{HGA_LLAMA_TAG} ({HGA_LLAMA_SHA[:12]}).  run scripts/setup.sh "
        f"(it asks to switch the checkout to {HGA_LLAMA_TAG})."
    )


def once(path: Path, needle: str, insert: str, *, after: bool = True, marker: str | None = None) -> None:
    text = path.read_text(encoding="utf-8")
    if (marker or insert.strip()) in text:
        print(f"  already patched: {path.name}")
        return
    if needle not in text:
        die(f"anchor not found in {path}: {needle[:80]!r}")
    if after:
        text = text.replace(needle, needle + insert, 1)
    else:
        text = text.replace(needle, insert + needle, 1)
    path.write_text(text, encoding="utf-8")
    print(f"  patched {path}")


def maybe_replace(path: Path, old: str, new: str) -> None:
    """Swap `old` → `new` when present; no-op if the tree is already on `new`."""
    text = path.read_text(encoding="utf-8")
    if new in text or old not in text:
        return
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"  updated {path.name}")


def replace(path: Path, old: str, new: str, *, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    # `new` often contains `old` as a prefix (insert-after). Check `new` first
    # or a second run would apply the insertion twice.
    if new in text:
        print(f"  already patched: {path.name}")
        return
    if old not in text:
        die(f"replace target not found in {path}: {old[:80]!r}")
    n = text.count(old)
    if count == -1:
        text = text.replace(old, new)
    else:
        if n < count:
            die(f"replace expected {count} occurrence(s) in {path}, found {n}: {old[:80]!r}")
        text = text.replace(old, new, count)
    path.write_text(text, encoding="utf-8")
    print(f"  patched {path}")


def patch_muse_glimmer_tool_termination(root: Path) -> None:
    """Make single Muse Glimmer tool calls terminate the generation.

    Qwen3.8's embedded template is handled by llama.cpp's specialized Muse
    Glimmer PEG parser. Its top-level expression is missing ``p.end()``, so
    with ``parallel_tool_calls=false`` the parser accepts one complete call
    but sampling continues into another message. The grammar stack is then
    empty and llama-server returns HTTP 500 instead of the tool call.
    """
    path = root / "common" / "chat.cpp"
    old = (
        "            return p.zero_or_more(start + analysis) + start + "
        "(tool_calls | (final_msg + trailing_calls));\n"
    )
    new = (
        "            return p.zero_or_more(start + analysis) + start + "
        "(tool_calls | (final_msg + trailing_calls)) + p.end();\n"
    )
    replace(path, old, new)


def patch_lazy_prefix_checkpoints(root: Path) -> None:
    """Add HGA's adaptive branch/continuation recurrent-state prefix cache."""
    path = root / "tools" / "server" / "server-context.cpp"

    once(
        path,
        "#include <fstream>\n",
        "#include <unordered_map>\n",
        marker="struct hga_lazy_prefix_cache",
    )

    cache_impl = r'''
struct hga_lazy_prefix_cache {
    static constexpr size_t key_tokens = 256;

    struct checkpoint {
        llama_tokens tokens;
        common_prompt_checkpoint state;
        uint64_t stamp = 0;
    };

    struct family {
        llama_tokens key;
        std::list<llama_tokens> prompts;
        std::list<checkpoint> checkpoints;
        uint64_t stamp = 0;
    };

    struct match {
        size_t intersection = 0;
        checkpoint * restore = nullptr;
        size_t prompts = 0;
        bool indexed = false;
    };

    explicit hga_lazy_prefix_cache(size_t limit) : limit(limit) {}

    static uint64_t hash(const llama_tokens & tokens) {
        uint64_t value = UINT64_C(1469598103934665603);
        for (size_t i = 0; i < key_tokens; ++i) {
            const uint32_t token = (uint32_t) tokens[i];
            for (int shift = 0; shift < 32; shift += 8) {
                value ^= (token >> shift) & 0xff;
                value *= UINT64_C(1099511628211);
            }
        }
        return value;
    }

    static size_t common_prefix(const llama_tokens & a, const llama_tokens & b) {
        const size_t n = std::min(a.size(), b.size());
        size_t i = 0;
        while (i < n && a[i] == b[i]) {
            ++i;
        }
        return i;
    }

    match find(const llama_tokens & tokens, size_t resident_prefix) {
        match result;
        if (tokens.size() < key_tokens) {
            return result;
        }

        const auto it = families.find(hash(tokens));
        if (it == families.end() || common_prefix(it->second.key, tokens) < key_tokens) {
            return result;
        }

        auto & cur = it->second;
        cur.stamp = ++clock;
        result.indexed = true;
        result.prompts = cur.prompts.size();
        for (const auto & prompt : cur.prompts) {
            const size_t n = common_prefix(prompt, tokens);
            if (n >= key_tokens && n < prompt.size() && n < tokens.size()) {
                result.intersection = std::max(result.intersection, n);
            }
        }

        for (auto & ckpt : cur.checkpoints) {
            if (ckpt.tokens.size() <= resident_prefix &&
                    ckpt.tokens.size() <= tokens.size() &&
                    common_prefix(ckpt.tokens, tokens) == ckpt.tokens.size() &&
                    (!result.restore || ckpt.tokens.size() > result.restore->tokens.size())) {
                result.restore = &ckpt;
            }
        }
        if (result.restore) {
            result.restore->stamp = ++clock;
        }
        return result;
    }

    void observe(const llama_tokens & tokens) {
        if (tokens.size() < key_tokens) {
            return;
        }

        const uint64_t key_hash = hash(tokens);
        auto it = families.find(key_hash);
        if (it != families.end() && common_prefix(it->second.key, tokens) < key_tokens) {
            return;
        }
        if (it == families.end()) {
            while (families.size() >= limit) {
                auto oldest = std::min_element(families.begin(), families.end(), [](const auto & a, const auto & b) {
                    return a.second.stamp < b.second.stamp;
                });
                n_checkpoints -= oldest->second.checkpoints.size();
                families.erase(oldest);
            }
            family fresh;
            fresh.key.assign(tokens.begin(), tokens.begin() + key_tokens);
            fresh.stamp = ++clock;
            it = families.emplace(key_hash, std::move(fresh)).first;
        }

        auto & cur = it->second;
        cur.stamp = ++clock;
        const bool duplicate = std::any_of(cur.prompts.begin(), cur.prompts.end(), [&](const auto & prompt) {
            return prompt == tokens;
        });
        if (!duplicate) {
            while (cur.prompts.size() >= limit) {
                cur.prompts.pop_front();
            }
            cur.prompts.push_back(tokens);
        }
    }

    void store(const llama_tokens & tokens, common_prompt_checkpoint && state) {
        if (tokens.size() < key_tokens) {
            return;
        }

        auto it = families.find(hash(tokens));
        if (it == families.end() || common_prefix(it->second.key, tokens) < key_tokens) {
            return;
        }

        auto & cur = it->second;
        for (auto & ckpt : cur.checkpoints) {
            if (ckpt.tokens == tokens) {
                ckpt.state = std::move(state);
                ckpt.stamp = ++clock;
                return;
            }
        }

        while (n_checkpoints >= limit) {
            family * oldest_family = nullptr;
            std::list<checkpoint>::iterator oldest_checkpoint;
            for (auto & item : families) {
                for (auto ckpt = item.second.checkpoints.begin(); ckpt != item.second.checkpoints.end(); ++ckpt) {
                    if (!oldest_family || ckpt->stamp < oldest_checkpoint->stamp) {
                        oldest_family = &item.second;
                        oldest_checkpoint = ckpt;
                    }
                }
            }
            if (!oldest_family) {
                break;
            }
            oldest_family->checkpoints.erase(oldest_checkpoint);
            --n_checkpoints;
        }

        cur.checkpoints.push_back({tokens, std::move(state), ++clock});
        cur.stamp = clock;
        ++n_checkpoints;
    }

    size_t family_count() const {
        return families.size();
    }

    size_t checkpoint_count() const {
        return n_checkpoints;
    }

private:
    size_t limit;
    size_t n_checkpoints = 0;
    uint64_t clock = 0;
    std::unordered_map<uint64_t, family> families;
};

'''
    once(
        path,
        "// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283\n",
        cache_impl,
        after=False,
        marker="struct hga_lazy_prefix_cache",
    )

    once(
        path,
        "    server_prompt prompt;\n",
        "\n"
        "    size_t hga_lazy_checkpoint_at = 0;\n",
        marker="hga_lazy_checkpoint_at",
    )

    once(
        path,
        "    std::unique_ptr<server_prompt_cache> prompt_cache;\n",
        "    std::unique_ptr<hga_lazy_prefix_cache> hga_lazy_cache;\n",
        marker="unique_ptr<hga_lazy_prefix_cache>",
    )

    once(
        path,
        "        SRV_TRC(\"%s\", \"for more info see https://github.com/ggml-org/llama.cpp/pull/16391\\n\");\n",
        r'''

        if (params_base.hga_enabled) {
            const char * env = getenv("HGA_LAZY_PREFIX_CACHE");
            const int limit = env ? atoi(env) : 0;
            if (limit > 0) {
                hga_lazy_cache = std::make_unique<hga_lazy_prefix_cache>((size_t) limit);
                SRV_INF("HGA lazy prefix cache enabled, entries = %d, key tokens = %zu\n", limit, hga_lazy_cache->key_tokens);
            }
        }
''',
        marker="HGA lazy prefix cache enabled",
    )

    capture_impl = r'''
    void create_hga_lazy_checkpoint(server_slot & slot) {
        if (!hga_lazy_cache || slot.hga_lazy_checkpoint_at == 0 ||
                slot.prompt.n_tokens() != (int) slot.hga_lazy_checkpoint_at) {
            return;
        }

        const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
        const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
        if (pos_min < 0 || pos_max < 0) {
            slot.hga_lazy_checkpoint_at = 0;
            return;
        }

        common_prompt_checkpoint state;
        state.id_task = slot.task->id;
        state.update_pos(slot.prompt.n_tokens(), pos_min, pos_max);
        state.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        state.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        common_speculative_get_state(spec.get(), slot.id, state.data_spec);

        llama_tokens prefix = slot.task->tokens.get_text_tokens();
        prefix.resize(slot.hga_lazy_checkpoint_at);
        const size_t checkpoint_tokens = slot.hga_lazy_checkpoint_at;
        const float checkpoint_mib = (float) state.size() / 1024 / 1024;
        hga_lazy_cache->store(prefix, std::move(state));
        slot.hga_lazy_checkpoint_at = 0;

        SLT_INF(slot, "hga-prefix: STORE tokens=%zu size=%.3f MiB checkpoints=%zu\n",
                checkpoint_tokens, checkpoint_mib, hga_lazy_cache->checkpoint_count());
    }

    void create_hga_completion_checkpoint(const server_slot & slot) {
        if (!hga_lazy_cache || !slot.task ||
                slot.task->type != SERVER_TASK_TYPE_COMPLETION ||
                !slot.task->params.cache_prompt || slot.prompt.tokens.has_mtmd ||
                slot.prompt.n_tokens() < (int) hga_lazy_cache->key_tokens) {
            return;
        }

        const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
        const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
        if (pos_min < 0 || pos_max < 0) {
            return;
        }

        common_prompt_checkpoint state;
        state.id_task = slot.task->id;
        state.update_pos(slot.prompt.n_tokens(), pos_min, pos_max);
        state.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        state.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        common_speculative_get_state(spec.get(), slot.id, state.data_spec);

        const auto tokens = slot.prompt.tokens.get_text_tokens();
        const float checkpoint_mib = (float) state.size() / 1024 / 1024;
        hga_lazy_cache->observe(tokens);
        hga_lazy_cache->store(tokens, std::move(state));

        SLT_INF(slot, "hga-prefix: FINISH tokens=%zu size=%.3f MiB checkpoints=%zu\n",
                tokens.size(), checkpoint_mib, hga_lazy_cache->checkpoint_count());
    }

'''
    once(
        path,
        "    // returns false to decline the task, it is offered again after the decode is done\n",
        capture_impl,
        after=False,
        marker="create_hga_lazy_checkpoint",
    )

    once(
        path,
        "            slot.callback_on_reset = [this](const server_slot & slot) {\n"
        "                // flush the generated token stats before reset()\n",
        "                create_hga_completion_checkpoint(slot);\n",
        marker="create_hga_completion_checkpoint(slot)",
    )

    old_cache = r'''                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);
'''
    new_cache = r'''                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                if (hga_lazy_cache && !input_tokens.has_mtmd) {
                                    slot.hga_lazy_checkpoint_at = 0;
                                    const auto text_tokens = input_tokens.get_text_tokens();
                                    auto lazy = hga_lazy_cache->find(text_tokens, n_past);
                                    size_t lazy_cached = 0;

                                    if (lazy.restore) {
                                        lazy.restore->state.load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        lazy.restore->state.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        common_speculative_set_state(spec.get(), slot.id, lazy.restore->state.data_spec);
                                        n_past = lazy.restore->tokens.size();
                                        lazy_cached = lazy.restore->tokens.size();
                                        SLT_INF(slot, "hga-prefix: HIT tokens=%d size=%.3f MiB\n", n_past,
                                                (float) lazy.restore->state.size() / 1024 / 1024);
                                    }

                                    if (lazy.intersection > lazy_cached) {
                                        slot.hga_lazy_checkpoint_at = lazy.intersection;
                                        SLT_INF(slot, "hga-prefix: INTERSECT tokens=%zu cached=%d\n", lazy.intersection, n_past);
                                    }

                                    hga_lazy_cache->observe(text_tokens);
                                    SLT_INF(slot, "hga-prefix: INDEX tokens=%zu families=%zu indexed=%d prompts=%zu intersection=%zu\n",
                                            text_tokens.size(), hga_lazy_cache->family_count(), (int) lazy.indexed,
                                            lazy.prompts, lazy.intersection);
                                }
'''
    replace(path, old_cache, new_cache)

    once(
        path,
        "                    // If using an alora, there may be uncached tokens that come\n",
        "                    create_hga_lazy_checkpoint(slot);\n\n",
        after=False,
        marker="create_hga_lazy_checkpoint(slot)",
    )

    old_push = r'''                        slot.prompt.tokens.push_back(cur_tok);

                        // break at the last user message, or at user messages at least min step past the last checkpoint
'''
    new_push = r'''                        slot.prompt.tokens.push_back(cur_tok);

                        if (slot.hga_lazy_checkpoint_at > 0 &&
                                slot.prompt.n_tokens() == (int) slot.hga_lazy_checkpoint_at) {
                            break;
                        }

                        // break at the last user message, or at user messages at least min step past the last checkpoint
'''
    replace(path, old_push, new_push)


def patch_graph_recipe_cache(root: Path) -> None:
    """Retain immutable host graph recipes across HGA scheduler teardown."""
    graph_h = root / "src" / "llama-graph.h"
    graph_cpp = root / "src" / "llama-graph.cpp"
    context_h = root / "src" / "llama-context.h"
    context_cpp = root / "src" / "llama-context.cpp"

    once(
        graph_h,
        "    void reset();\n",
        "\n"
        "    void capture_graph_recipe(ggml_backend_sched_t sched);\n"
        "    void restore_graph_recipe();\n"
        "    void apply_graph_recipe(ggml_backend_sched_t sched) const;\n"
        "    bool has_graph_recipe() const;\n",
        marker="capture_graph_recipe",
    )
    once(
        graph_h,
        "    void reset();\n",
        "\n    void * alloc_custom_userdata(size_t size);\n",
        marker="alloc_custom_userdata",
    )
    once(
        graph_h,
        "    int64_t max_nodes;\n",
        "\n"
        "    std::vector<ggml_tensor *> graph_recipe_tensors;\n"
        "    std::vector<ggml_tensor *> graph_recipe_sources;\n"
        "    std::vector<void *> graph_recipe_buffers;\n"
        "    std::vector<void *> graph_recipe_data;\n"
        "    std::vector<void *> graph_recipe_extra;\n"
        "    std::vector<ggml_backend_t> graph_recipe_backends;\n",
        marker="graph_recipe_backends",
    )
    once(
        graph_h,
        "    std::vector<ggml_backend_t> graph_recipe_backends;\n",
        "    std::vector<std::unique_ptr<uint8_t[]>> custom_userdata;\n",
        marker="std::vector<std::unique_ptr<uint8_t[]>> custom_userdata",
    )
    maybe_replace(
        graph_h,
        "            cparams.causal_attn             == other.cparams.causal_attn             &&\n",
        "            cparams.causal_attn             == other.cparams.causal_attn             &&\n"
        "            cparams.hga_phase               == other.cparams.hga_phase               &&\n"
        "            cparams.op_offload              == other.cparams.op_offload              &&\n",
    )
    once(
        graph_cpp,
        "    fused_nodes.clear();\n",
        "\n"
        "    graph_recipe_tensors.clear();\n"
        "    graph_recipe_sources.clear();\n"
        "    graph_recipe_buffers.clear();\n"
        "    graph_recipe_data.clear();\n"
        "    graph_recipe_extra.clear();\n"
        "    graph_recipe_backends.clear();\n",
        marker="graph_recipe_sources.clear()",
    )
    once(
        graph_cpp,
        "    graph_recipe_backends.clear();\n",
        "    custom_userdata.clear();\n",
        marker="custom_userdata.clear()",
    )
    once(
        graph_cpp,
        "void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {\n",
        """void * llm_graph_result::alloc_custom_userdata(size_t size) {
    auto data = std::make_unique<uint8_t[]>(size);
    void * ptr = data.get();
    custom_userdata.emplace_back(std::move(data));
    return ptr;
}

""",
        after=False,
        marker="void * llm_graph_result::alloc_custom_userdata",
    )
    once(
        graph_cpp,
        "void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {\n",
        """void llm_graph_result::capture_graph_recipe(ggml_backend_sched_t sched) {
    GGML_ASSERT(graph_recipe_tensors.empty());

    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx_compute.get()); tensor != nullptr; tensor = ggml_get_next_tensor(ctx_compute.get(), tensor)) {
        graph_recipe_tensors.push_back(tensor);
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            graph_recipe_sources.push_back(tensor->src[i]);
        }
        graph_recipe_buffers.push_back(tensor->buffer);
        graph_recipe_data.push_back(tensor->data);
        graph_recipe_extra.push_back(tensor->extra);
        graph_recipe_backends.push_back(ggml_backend_sched_get_tensor_backend(sched, tensor));
    }
}

void llm_graph_result::restore_graph_recipe() {
    GGML_ASSERT(graph_recipe_sources.size() == graph_recipe_tensors.size() * GGML_MAX_SRC);
    for (size_t i = 0; i < graph_recipe_tensors.size(); ++i) {
        ggml_tensor * tensor = graph_recipe_tensors[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            tensor->src[j] = graph_recipe_sources[i * GGML_MAX_SRC + j];
        }
        tensor->buffer = static_cast<ggml_backend_buffer_t>(graph_recipe_buffers[i]);
        tensor->data = graph_recipe_data[i];
        tensor->extra = graph_recipe_extra[i];
    }
}

void llm_graph_result::apply_graph_recipe(ggml_backend_sched_t sched) const {
    GGML_ASSERT(graph_recipe_backends.size() == graph_recipe_tensors.size());
    for (size_t i = 0; i < graph_recipe_tensors.size(); ++i) {
        if (graph_recipe_backends[i]) {
            ggml_backend_sched_set_tensor_backend(sched, graph_recipe_tensors[i], graph_recipe_backends[i]);
        }
    }
}

bool llm_graph_result::has_graph_recipe() const {
    return !graph_recipe_tensors.empty();
}

""",
        after=False,
        marker="void llm_graph_result::capture_graph_recipe",
    )

    once(
        context_h,
        "    void hga_vram_log(const char * tag, uint32_t n_tokens = 0);\n",
        "    void hga_graph_cache_detach();\n",
        marker="hga_graph_cache_detach",
    )
    once(
        context_h,
        "    llm_graph_cb graph_get_cb() const;\n",
        "\n    void graph_recipe_cache_store(llm_graph_result_ptr & slot, std::vector<llm_graph_result_ptr> & cache);\n",
        marker="graph_recipe_cache_store",
    )
    once(
        context_h,
        "    llm_graph_result_ptr gf_res_reserve;\n",
        "\n"
        "    std::vector<llm_graph_result_ptr> graph_recipe_cache;\n"
        "    std::vector<llm_graph_result_ptr> graph_reserve_recipe_cache;\n"
        "    bool graph_recipe_cache_enabled = false;\n"
        "    size_t graph_recipe_cache_max = 8;\n"
        "    uint64_t graph_recipe_cache_hits = 0;\n"
        "    uint64_t graph_recipe_cache_builds = 0;\n"
        "    uint64_t graph_reserve_recipe_cache_hits = 0;\n"
        "    uint64_t graph_reserve_recipe_cache_builds = 0;\n",
        marker="graph_reserve_recipe_cache_builds",
    )
    once(
        context_cpp,
        """        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\\n", __func__);
        }
""",
        """
        const char * HGA_GRAPH_RECIPE_CACHE = getenv("HGA_GRAPH_RECIPE_CACHE");
        graph_recipe_cache_enabled = !graph_reuse_disable && cparams.hga_enabled && !(HGA_GRAPH_RECIPE_CACHE && HGA_GRAPH_RECIPE_CACHE[0] == '0');
        const char * HGA_GRAPH_RECIPE_CACHE_MAX = getenv("HGA_GRAPH_RECIPE_CACHE_MAX");
        if (HGA_GRAPH_RECIPE_CACHE_MAX) {
            graph_recipe_cache_max = std::max<size_t>(2, std::strtoul(HGA_GRAPH_RECIPE_CACHE_MAX, nullptr, 10));
        }
        if (graph_recipe_cache_enabled) {
            LLAMA_LOG_INFO("%s: HGA immutable graph recipe cache enabled (max %zu compute + reserve recipes)\\n", __func__, graph_recipe_cache_max);
        }
        std::fprintf(stderr, "hga-graph: immutable recipe cache=%s max=%zu\\n", graph_recipe_cache_enabled ? "on" : "off", graph_recipe_cache_max);
""",
        marker="HGA_GRAPH_RECIPE_CACHE_MAX",
    )
    once(
        context_cpp,
        "llm_graph_result * llama_context::process_ubatch(",
        """void llama_context::graph_recipe_cache_store(llm_graph_result_ptr & slot, std::vector<llm_graph_result_ptr> & cache) {
    if (!graph_recipe_cache_enabled || !slot || !slot->has_graph_recipe()) {
        return;
    }

    slot->restore_graph_recipe();
    if (cache.size() >= graph_recipe_cache_max) {
        cache.erase(cache.begin());
    }
    cache.emplace_back(std::move(slot));
}

void llama_context::hga_graph_cache_detach() {
    graph_recipe_cache_store(gf_res_prev, graph_recipe_cache);
    graph_recipe_cache_store(gf_res_reserve, graph_reserve_recipe_cache);
    gf_res_prev.reset();
    gf_res_reserve.reset();
}

""",
        after=False,
        marker="void llama_context::graph_recipe_cache_store",
    )

    process_old = """    } else {
        res->reset();

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        hga_vram_log("ubatch before alloc_graph", ubatch.n_tokens);
"""
    process_new = """    } else {
        const int64_t max_nodes = res->get_max_nodes();
        if (graph_recipe_cache_enabled && cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }
        graph_recipe_cache_store(gf_res_prev, graph_recipe_cache);
        if (!gf_res_prev) {
            gf_res_prev.reset(new llm_graph_result(max_nodes));
        } else {
            gf_res_prev->reset();
        }

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        bool recipe_hit = false;
        if (graph_recipe_cache_enabled) {
            for (size_t i = 0; i < graph_recipe_cache.size(); ++i) {
                auto & candidate = graph_recipe_cache[i];
                const auto candidate_params = graph_params(candidate.get(), ubatch, mctx, gtype);
                if (candidate->can_reuse(candidate_params)) {
                    gf_res_prev = std::move(candidate);
                    graph_recipe_cache.erase(graph_recipe_cache.begin() + i);
                    res = gf_res_prev.get();
                    gf = res->get_gf();
                    res->apply_graph_recipe(sched.get());
                    recipe_hit = true;
                    ++graph_recipe_cache_hits;
                    ++n_reused;
                    LLAMA_LOG_INFO("%s: HGA graph recipe hit n_tokens=%u nodes=%d hits=%" PRIu64 " builds=%" PRIu64 "\\n", __func__, ubatch.n_tokens, ggml_graph_n_nodes(gf), graph_recipe_cache_hits, graph_recipe_cache_builds);
                    std::fprintf(stderr, "hga-graph: compute HIT n_tokens=%u nodes=%d hits=%" PRIu64 " builds=%" PRIu64 "\\n", ubatch.n_tokens, ggml_graph_n_nodes(gf), graph_recipe_cache_hits, graph_recipe_cache_builds);
                    break;
                }
            }
        }

        if (!recipe_hit) {
            res = gf_res_prev.get();
            const auto build_params = graph_params(res, ubatch, mctx, gtype);
            const int64_t t_start_us = ggml_time_us();
            gf = model.build_graph(build_params);
            ++graph_recipe_cache_builds;
            LLAMA_LOG_INFO("%s: HGA graph recipe build n_tokens=%u nodes=%d time=%.3f ms builds=%" PRIu64 "\\n", __func__, ubatch.n_tokens, gf ? ggml_graph_n_nodes(gf) : 0, (ggml_time_us() - t_start_us)/1000.0, graph_recipe_cache_builds);
            if (graph_recipe_cache_enabled) {
                std::fprintf(stderr, "hga-graph: compute BUILD n_tokens=%u nodes=%d time=%.3f ms builds=%" PRIu64 "\\n", ubatch.n_tokens, gf ? ggml_graph_n_nodes(gf) : 0, (ggml_time_us() - t_start_us)/1000.0, graph_recipe_cache_builds);
            }

            if (!gf) {
                LLAMA_LOG_ERROR("%s: failed to initialize graph\\n", __func__);
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
            res->capture_graph_recipe(sched.get());
        }

        hga_vram_log("ubatch before alloc_graph", ubatch.n_tokens);
"""
    replace(context_cpp, process_old, process_new)

    once(
        context_cpp,
        "    gf_res_prev.reset(new llm_graph_result(max_nodes));\n",
        "    hga_graph_cache_detach();\n",
        after=False,
        marker="hga_graph_cache_detach();\n    gf_res_prev.reset(new llm_graph_result(max_nodes))",
    )

    reserve_reset_old = """    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();
"""
    reserve_reset_new = """    const int64_t reserve_max_nodes = gf_res_reserve->get_max_nodes();
    graph_recipe_cache_store(gf_res_prev, graph_recipe_cache);
    graph_recipe_cache_store(gf_res_reserve, graph_reserve_recipe_cache);
    if (!gf_res_prev) {
        gf_res_prev.reset(new llm_graph_result(reserve_max_nodes));
    } else {
        gf_res_prev->reset();
    }
    if (!gf_res_reserve) {
        gf_res_reserve.reset(new llm_graph_result(reserve_max_nodes));
    } else {
        gf_res_reserve->reset();
    }

    ggml_backend_sched_reset(sched.get());
"""
    replace(context_cpp, reserve_reset_old, reserve_reset_new)

    reserve_build_old = """    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);
"""
    reserve_build_new = """    auto * res = gf_res_reserve.get();
    ggml_cgraph * gf = nullptr;
    bool recipe_hit = false;
    if (graph_recipe_cache_enabled) {
        for (size_t i = 0; i < graph_reserve_recipe_cache.size(); ++i) {
            auto & candidate = graph_reserve_recipe_cache[i];
            const auto candidate_params = graph_params(candidate.get(), ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));
            if (candidate->can_reuse(candidate_params)) {
                gf_res_reserve = std::move(candidate);
                graph_reserve_recipe_cache.erase(graph_reserve_recipe_cache.begin() + i);
                res = gf_res_reserve.get();
                gf = res->get_gf();
                res->apply_graph_recipe(sched.get());
                recipe_hit = true;
                ++graph_reserve_recipe_cache_hits;
                LLAMA_LOG_INFO("%s: HGA reserve recipe hit n_tokens=%u nodes=%d hits=%" PRIu64 " builds=%" PRIu64 "\\n", __func__, n_tokens, ggml_graph_n_nodes(gf), graph_reserve_recipe_cache_hits, graph_reserve_recipe_cache_builds);
                std::fprintf(stderr, "hga-graph: reserve HIT n_tokens=%u nodes=%d hits=%" PRIu64 " builds=%" PRIu64 "\\n", n_tokens, ggml_graph_n_nodes(gf), graph_reserve_recipe_cache_hits, graph_reserve_recipe_cache_builds);
                break;
            }
        }
    }

    if (!recipe_hit) {
        res = gf_res_reserve.get();
        const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));
        const int64_t t_start_us = ggml_time_us();
        gf = model.build_graph(gparams);
        ++graph_reserve_recipe_cache_builds;
        LLAMA_LOG_INFO("%s: HGA reserve recipe build n_tokens=%u nodes=%d time=%.3f ms builds=%" PRIu64 "\\n", __func__, n_tokens, gf ? ggml_graph_n_nodes(gf) : 0, (ggml_time_us() - t_start_us)/1000.0, graph_reserve_recipe_cache_builds);
        if (graph_recipe_cache_enabled) {
            std::fprintf(stderr, "hga-graph: reserve BUILD n_tokens=%u nodes=%d time=%.3f ms builds=%" PRIu64 "\\n", n_tokens, gf ? ggml_graph_n_nodes(gf) : 0, (ggml_time_us() - t_start_us)/1000.0, graph_reserve_recipe_cache_builds);
        }
        if (gf) {
            res->capture_graph_recipe(sched.get());
        }
    }
"""
    replace(context_cpp, reserve_build_old, reserve_build_new)
    print("  patched immutable HGA graph recipe cache")


def patch_cuda_vmm_shrink(root: Path) -> None:
    """Unmap CUDA VMM physical pages when a large compute pool is empty.

    ggml_cuda_pool_vmm::free only drops pool_used. pool_size (cuMemCreate
    mappings) stays until the pool destructor, so cudaMemGetInfo still
    counts the PREFILL scratch after sched.reset(). DECODE then looks like
    it still holds the 128-token graph, and a second llama_context (MTP)
    cannot reuse that VRAM.

    Unmap only large arenas (>= 16 MiB) after a device sync: tiny per-op
    scratch is alloc/freed while kernels are still queued. HGA_VMM_KEEP=1
    disables this.
    """
    path = root / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.cu"
    free_head = (
        "    void free(void * ptr, size_t size) override {\n"
        "#ifdef DEBUG_CUDA_MALLOC\n"
        '        printf("cuda pool[%d]: freed %llu bytes at %llx\\n", device, (unsigned long long) size, ptr);\n'
        "#endif\n"
        "\n"
        "        pool_used -= size;\n"
        "\n"
        "        // all deallocations must be in reverse order of the allocations\n"
        "        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));\n"
    )
    upstream = free_head + "    }\n"
    v1 = (
        free_head
        + "\n"
        + "        // HGA: unmap physical pages when the pool is empty so destroying the\n"
        + "        // PREFILL scheduler actually returns VRAM (cuMemCreate is sticky).\n"
        + "        if (pool_used == 0 && pool_size > 0 && pool_addr != 0) {\n"
        + '            const char * keep = getenv("HGA_VMM_KEEP");\n'
        + "            if (!(keep && keep[0] && keep[0] != '0')) {\n"
        + '                GGML_LOG_INFO("cuda vmm pool[%d]: unmap %.2f MiB (prefill compute graph dropped)\\n",\n'
        + "                        device, pool_size / 1024.0 / 1024.0);\n"
        + "#if defined(GGML_USE_HIP)\n"
        + "                for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {\n"
        + "                    CU_CHECK(cuMemUnmap(mapping.first, mapping.second));\n"
        + "                }\n"
        + "                mappings.clear();\n"
        + "#else\n"
        + "                CU_CHECK(cuMemUnmap(pool_addr, pool_size));\n"
        + "#endif\n"
        + "                pool_size = 0;\n"
        + "            }\n"
        + "        }\n"
        + "    }\n"
    )
    v2 = (
        free_head
        + "\n"
        + "        // HGA: unmap large empty arenas so destroying the PREFILL scheduler\n"
        + "        // returns VRAM (cuMemCreate is sticky). Skip tiny scratch: those\n"
        + "        // free()s race with still-queued kernels (CONCAT illegal access).\n"
        + "        if (size >= (1ull << 20) || pool_used == 0) {\n"
        + '            const char * keep = getenv("HGA_VMM_KEEP");\n'
        + "            const bool keeping = keep && keep[0] && keep[0] != '0';\n"
        + '            GGML_LOG_INFO("cuda vmm pool[%d]: free %.2f MiB -> used=%.2f size=%.2f%s\\n",\n'
        + "                    device, size / 1024.0 / 1024.0,\n"
        + "                    pool_used / 1024.0 / 1024.0, pool_size / 1024.0 / 1024.0,\n"
        + "                    (pool_used == 0 && pool_size >= (16ull << 20))\n"
        + '                        ? (keeping ? " KEEP (no unmap)" : " will-unmap") : "");\n'
        + "        }\n"
        + "        if (pool_used == 0 && pool_size >= (16ull << 20) && pool_addr != 0) {\n"
        + '            /* Unmap only when HGA_VMM_UNMAP=1 (armed around drop-prefill).\n'
        + '             * Unmapping on every empty pool (HGA_VMM_KEEP=0) device-syncs and\n'
        + '             * remaps 300+ MiB between 512-token prefill chunks. */\n'
        + '            const char * unmap = getenv("HGA_VMM_UNMAP");\n'
        + "            if (unmap && unmap[0] == '1') {\n"
        + "                ggml_cuda_set_device(device);\n"
        + "                CUDA_CHECK(cudaDeviceSynchronize());\n"
        + '                fprintf(stderr, "cuda vmm pool[%d]: UNMAP %.2f MiB (HGA_VMM_UNMAP)\\n",\n'
        + "                        device, pool_size / 1024.0 / 1024.0);\n"
        + "#if defined(GGML_USE_HIP)\n"
        + "                for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {\n"
        + "                    CU_CHECK(cuMemUnmap(mapping.first, mapping.second));\n"
        + "                }\n"
        + "                mappings.clear();\n"
        + "#else\n"
        + "                CU_CHECK(cuMemUnmap(pool_addr, pool_size));\n"
        + "#endif\n"
        + "                pool_size = 0;\n"
        + "            }\n"
        + "        }\n"
        + "    }\n"
    )
    text = path.read_text(encoding="utf-8")
    dtor_old = (
        "            CU_CHECK(cuMemUnmap(pool_addr, pool_size));\n"
        "            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));\n"
    )
    dtor_new = (
        "            if (pool_size > 0) {\n"
        "                CU_CHECK(cuMemUnmap(pool_addr, pool_size));\n"
        "            }\n"
        "            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));\n"
    )
    if dtor_old in text:
        text = text.replace(dtor_old, dtor_new, 1)
        path.write_text(text, encoding="utf-8")
        print(f"  patched {path.name} VMM dtor skip unmap(0)")
        text = path.read_text(encoding="utf-8")
    if "HGA_VMM_UNMAP" in text:
        print(f"  already patched: {path.name} VMM shrink (HGA_VMM_UNMAP present)")
    elif v2 in text:
        print(f"  already patched: {path.name} VMM shrink")
    elif v1 in text:
        path.write_text(text.replace(v1, v2, 1), encoding="utf-8")
        print(f"  updated {path.name} VMM shrink (sync + 16 MiB floor)")
    elif upstream in text:
        path.write_text(text.replace(upstream, v2, 1), encoding="utf-8")
        print(f"  patched {path}")
    else:
        die(f"cuda VMM pool free() not found in {path}")
    patch_cuda_vmm_explicit_shrink(path)


def patch_cuda_vmm_explicit_shrink(path: Path) -> None:
    """Unmap sticky VMM pages even when pool_used is already 0.

    Prefill scratch is alloc'd during graph_compute and freed after each op,
    so pool_used is 0 by the time we drop the scheduler. free() then never
    runs, HGA_VMM_UNMAP never fires, and cudaMemGetInfo still counts the
    mapped arena (~200+ MiB). That is why leftover pair 24-56 is ~35 MiB
    short. Walk every live VMM pool and unmap the unused tail.
    """
    text = path.read_text(encoding="utf-8")
    if "ggml_cuda_hga_vmm_shrink" in text:
        print(f"  already patched: {path.name} explicit VMM shrink")
        return

    proto = (
        "struct ggml_cuda_pool_vmm;\n"
        "static void hga_vmm_register(ggml_cuda_pool_vmm * p);\n"
        "static void hga_vmm_unregister(ggml_cuda_pool_vmm * p);\n"
        "\n"
        "struct ggml_cuda_pool_vmm : public ggml_cuda_pool {\n"
    )
    if "struct ggml_cuda_pool_vmm : public ggml_cuda_pool {\n" not in text:
        die(f"ggml_cuda_pool_vmm struct not found in {path}")
    text = text.replace(
        "struct ggml_cuda_pool_vmm : public ggml_cuda_pool {\n",
        proto,
        1,
    )

    ctor_old = (
        "    explicit ggml_cuda_pool_vmm(int device) :\n"
        "        device(device),\n"
        "        physical_device(ggml_cuda_get_physical_device(device)),\n"
        "        granularity(ggml_cuda_info().devices[device].vmm_granularity) {\n"
        "    }\n"
    )
    ctor_new = (
        "    explicit ggml_cuda_pool_vmm(int device) :\n"
        "        device(device),\n"
        "        physical_device(ggml_cuda_get_physical_device(device)),\n"
        "        granularity(ggml_cuda_info().devices[device].vmm_granularity) {\n"
        "        hga_vmm_register(this);\n"
        "    }\n"
    )
    if ctor_old not in text:
        die(f"ggml_cuda_pool_vmm ctor not found in {path}")
    text = text.replace(ctor_old, ctor_new, 1)

    dtor_old = (
        "    ~ggml_cuda_pool_vmm() {\n"
        "        if (pool_addr != 0) {\n"
    )
    dtor_new = (
        "    ~ggml_cuda_pool_vmm() {\n"
        "        hga_vmm_unregister(this);\n"
        "        if (pool_addr != 0) {\n"
    )
    if dtor_old not in text:
        die(f"ggml_cuda_pool_vmm dtor not found in {path}")
    text = text.replace(dtor_old, dtor_new, 1)

    # cuMemUnmap(addr, 0) is illegal after we shrink pool_size to 0.
    dtor_unmap_old = (
        "            CU_CHECK(cuMemUnmap(pool_addr, pool_size));\n"
        "            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));\n"
    )
    dtor_unmap_new = (
        "            if (pool_size > 0) {\n"
        "                CU_CHECK(cuMemUnmap(pool_addr, pool_size));\n"
        "            }\n"
        "            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));\n"
    )
    if dtor_unmap_old in text:
        text = text.replace(dtor_unmap_old, dtor_unmap_new, 1)

    impl = r'''
static std::mutex g_hga_vmm_mu;
static std::vector<ggml_cuda_pool_vmm *> g_hga_vmm_pools;

static void hga_vmm_register(ggml_cuda_pool_vmm * p) {
    std::lock_guard<std::mutex> lk(g_hga_vmm_mu);
    g_hga_vmm_pools.push_back(p);
}

static void hga_vmm_unregister(ggml_cuda_pool_vmm * p) {
    std::lock_guard<std::mutex> lk(g_hga_vmm_mu);
    g_hga_vmm_pools.erase(std::remove(g_hga_vmm_pools.begin(), g_hga_vmm_pools.end(), p),
                          g_hga_vmm_pools.end());
}

static void hga_vmm_shrink_one(ggml_cuda_pool_vmm * p) {
    if (!p) {
        return;
    }
    const double used_m = p->pool_used / 1024.0 / 1024.0;
    const double size_m = p->pool_size / 1024.0 / 1024.0;
    if (p->pool_addr == 0 || p->pool_size == 0) {
        fprintf(stderr, "cuda vmm pool[%d]: shrink skip used=%.2f size=%.2f (empty)\n",
                p->device, used_m, size_m);
        return;
    }
    ggml_cuda_set_device(p->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    size_t keep = 0;
    if (p->pool_used > 0) {
        keep = p->granularity * ((p->pool_used + p->granularity - 1) / p->granularity);
    }
    if (keep >= p->pool_size) {
        fprintf(stderr, "cuda vmm pool[%d]: shrink keep %.2f (used=%.2f size=%.2f) no unmap\n",
                p->device, keep / 1024.0 / 1024.0, used_m, size_m);
        return;
    }
    const size_t drop = p->pool_size - keep;
    fprintf(stderr, "cuda vmm pool[%d]: SHRINK unmap %.2f MiB  used=%.2f keep=%.2f was_size=%.2f\n",
            p->device, drop / 1024.0 / 1024.0, used_m, keep / 1024.0 / 1024.0, size_m);
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> kept;
    for (std::pair<CUdeviceptr, size_t> & mapping : p->mappings) {
        const size_t map_off = (size_t) ((char *) mapping.first - (char *) p->pool_addr);
        if (map_off >= keep) {
            CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
        } else {
            kept.push_back(mapping);
        }
    }
    p->mappings.swap(kept);
#else
    CU_CHECK(cuMemUnmap((CUdeviceptr) ((char *) p->pool_addr + keep), drop));
#endif
    p->pool_size = keep;
}

#endif // defined(GGML_USE_VMM)

extern "C" void ggml_cuda_hga_vmm_shrink(void) {
#if defined(GGML_USE_VMM)
    std::lock_guard<std::mutex> lk(g_hga_vmm_mu);
    fprintf(stderr, "cuda vmm shrink: %zu pools\n", g_hga_vmm_pools.size());
    for (ggml_cuda_pool_vmm * p : g_hga_vmm_pools) {
        hga_vmm_shrink_one(p);
    }
#else
    fprintf(stderr, "cuda vmm shrink: VMM not compiled\n");
#endif
}

#define HGA_VMM_SHRINK_DEFINED 1
#if defined(GGML_USE_VMM) && !defined(HGA_VMM_SHRINK_ENDIF_EATEN)
#endif
'''
    # The impl above accidentally nested an extra endif. Insert a clean block
    # in place of the original `#endif // defined(GGML_USE_VMM)` that closes
    # the vmm struct, then define the C API outside.
    marker = (
        "};\n"
        "#endif // defined(GGML_USE_VMM)\n"
        "\n"
        "std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device"
    )
    impl_clean = (
        "};\n"
        "\n"
        "static std::mutex g_hga_vmm_mu;\n"
        "static std::vector<ggml_cuda_pool_vmm *> g_hga_vmm_pools;\n"
        "\n"
        "static void hga_vmm_register(ggml_cuda_pool_vmm * p) {\n"
        "    std::lock_guard<std::mutex> lk(g_hga_vmm_mu);\n"
        "    g_hga_vmm_pools.push_back(p);\n"
        "}\n"
        "\n"
        "static void hga_vmm_unregister(ggml_cuda_pool_vmm * p) {\n"
        "    std::lock_guard<std::mutex> lk(g_hga_vmm_mu);\n"
        "    g_hga_vmm_pools.erase(std::remove(g_hga_vmm_pools.begin(), g_hga_vmm_pools.end(), p),\n"
        "                          g_hga_vmm_pools.end());\n"
        "}\n"
        "\n"
        "static void hga_vmm_shrink_one(ggml_cuda_pool_vmm * p) {\n"
        "    if (!p) {\n"
        "        return;\n"
        "    }\n"
        "    const double used_m = p->pool_used / 1024.0 / 1024.0;\n"
        "    const double size_m = p->pool_size / 1024.0 / 1024.0;\n"
        "    if (p->pool_addr == 0 || p->pool_size == 0) {\n"
        '        fprintf(stderr, "cuda vmm pool[%d]: shrink skip used=%.2f size=%.2f (empty)\\n",\n'
        "                p->device, used_m, size_m);\n"
        "        return;\n"
        "    }\n"
        "    ggml_cuda_set_device(p->device);\n"
        "    CUDA_CHECK(cudaDeviceSynchronize());\n"
        "    size_t keep = 0;\n"
        "    if (p->pool_used > 0) {\n"
        "        keep = p->granularity * ((p->pool_used + p->granularity - 1) / p->granularity);\n"
        "    }\n"
        "    if (keep >= p->pool_size) {\n"
        '        fprintf(stderr, "cuda vmm pool[%d]: shrink keep %.2f (used=%.2f size=%.2f) no unmap\\n",\n'
        "                p->device, keep / 1024.0 / 1024.0, used_m, size_m);\n"
        "        return;\n"
        "    }\n"
        "    const size_t drop = p->pool_size - keep;\n"
        '    fprintf(stderr, "cuda vmm pool[%d]: SHRINK unmap %.2f MiB  used=%.2f keep=%.2f was_size=%.2f\\n",\n'
        "            p->device, drop / 1024.0 / 1024.0, used_m, keep / 1024.0 / 1024.0, size_m);\n"
        "#if defined(GGML_USE_HIP)\n"
        "    std::vector<std::pair<CUdeviceptr, size_t>> kept;\n"
        "    for (std::pair<CUdeviceptr, size_t> & mapping : p->mappings) {\n"
        "        const size_t map_off = (size_t) ((char *) mapping.first - (char *) p->pool_addr);\n"
        "        if (map_off >= keep) {\n"
        "            CU_CHECK(cuMemUnmap(mapping.first, mapping.second));\n"
        "        } else {\n"
        "            kept.push_back(mapping);\n"
        "        }\n"
        "    }\n"
        "    p->mappings.swap(kept);\n"
        "#else\n"
        "    CU_CHECK(cuMemUnmap((CUdeviceptr) ((char *) p->pool_addr + keep), drop));\n"
        "#endif\n"
        "    p->pool_size = keep;\n"
        "}\n"
        "\n"
        "#endif // defined(GGML_USE_VMM)\n"
        "\n"
        "extern \"C\" void ggml_cuda_hga_vmm_shrink(void) {\n"
        "#if defined(GGML_USE_VMM)\n"
        "    std::lock_guard<std::mutex> lk(g_hga_vmm_mu);\n"
        '    fprintf(stderr, "cuda vmm shrink: %zu pools\\n", g_hga_vmm_pools.size());\n'
        "    for (ggml_cuda_pool_vmm * p : g_hga_vmm_pools) {\n"
        "        hga_vmm_shrink_one(p);\n"
        "    }\n"
        "#else\n"
        '    fprintf(stderr, "cuda vmm shrink: VMM not compiled\\n");\n'
        "#endif\n"
        "}\n"
        "\n"
        "std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device"
    )
    if marker not in text:
        die(f"VMM struct endif marker not found in {path}")
    text = text.replace(marker, impl_clean, 1)
    path.write_text(text, encoding="utf-8")
    print(f"  patched {path.name} explicit VMM shrink (registry + tail unmap)")


def patch_cuda_alloc_ledger(root: Path) -> None:
    """Keep the last 20 physical CUDA allocations, including ggml VMM growth."""
    path = root / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.cu"
    text = path.read_text(encoding="utf-8")
    marker = "HGA CUDA ALLOC LEDGER (last 20 physical allocations)"
    if marker in text:
        generic = 'ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device, "backend-buffer");'
        malformed = (
            'const char * hga_scope = ggml_backend_hga_alloc_scope_get();\n'
            '    ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device,\n'
            '            hga_scope && hga_scope[0] ? hga_scope : "backend-buffer");'
        )
        attributed = (
            'ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device,\n'
            '            ggml_backend_hga_alloc_scope_get() && ggml_backend_hga_alloc_scope_get()[0]\n'
            '                ? ggml_backend_hga_alloc_scope_get() : "backend-buffer");'
        )
        if malformed in text:
            text = text.replace(malformed, attributed, 1)
            path.write_text(text, encoding="utf-8")
            print("  repaired ggml-cuda.cu allocation caller attribution expression")
            text = path.read_text(encoding="utf-8")
        if generic in text:
            text = text.replace(generic, attributed, 1)
            path.write_text(text, encoding="utf-8")
            print("  upgraded ggml-cuda.cu allocation ledger with caller attribution")
            text = path.read_text(encoding="utf-8")
        low_old = """                hga_cuda_alloc_record("vmm-grow", device, reserve_size, free_before,
                        hga_cuda_free_bytes(), create_rc == CUDA_SUCCESS);
                if (create_rc != CUDA_SUCCESS) {
                    hga_cuda_alloc_dump("vmm-grow");
                }
"""
        low_new = """                const size_t free_after = hga_cuda_free_bytes();
                hga_cuda_alloc_record("vmm-grow", device, reserve_size, free_before,
                        free_after, create_rc == CUDA_SUCCESS);
                static bool low_memory_dumped = false;
                if (!low_memory_dumped && free_after < 768ull * 1024 * 1024) {
                    low_memory_dumped = true;
                    hga_cuda_alloc_dump("low-memory VMM grow");
                }
                if (create_rc != CUDA_SUCCESS) {
                    hga_cuda_alloc_dump("vmm-grow");
                }
"""
        if "low-memory VMM grow" not in text:
            if low_old not in text:
                die("ggml-cuda.cu: existing allocation ledger lacks VMM record anchor")
            path.write_text(text.replace(low_old, low_new, 1), encoding="utf-8")
            print("  upgraded ggml-cuda.cu allocation ledger with low-memory dump")
        else:
            print("  already patched: ggml-cuda.cu physical allocation ledger")
        return

    error_anchor = """[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
"""
    ledger = r'''/* HGA CUDA ALLOC LEDGER (last 20 physical allocations).
 * Opt-in: HGA_CUDA_ALLOC_LEDGER=1. */
struct hga_cuda_alloc_rec {
    uint64_t seq = 0;
    char kind[24] = {};
    int device = -1;
    size_t requested = 0;
    size_t free_before = 0;
    size_t free_after = 0;
    int ok = 0;
};

static std::mutex g_hga_cuda_alloc_mutex;
static hga_cuda_alloc_rec g_hga_cuda_alloc_ring[20];
static uint64_t g_hga_cuda_alloc_seq = 0;

static bool hga_cuda_alloc_ledger_enabled() {
    const char * e = getenv("HGA_CUDA_ALLOC_LEDGER");
    return e && e[0] && e[0] != '0';
}

static size_t hga_cuda_free_bytes() {
    size_t free_b = 0, total_b = 0;
    (void) cudaMemGetInfo(&free_b, &total_b);
    return free_b;
}

static void hga_cuda_alloc_record(const char * kind, int device, size_t requested,
                                  size_t free_before, size_t free_after, bool ok) {
    if (!hga_cuda_alloc_ledger_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_hga_cuda_alloc_mutex);
    const uint64_t seq = ++g_hga_cuda_alloc_seq;
    hga_cuda_alloc_rec & r = g_hga_cuda_alloc_ring[(seq - 1) % 20];
    r.seq = seq;
    snprintf(r.kind, sizeof(r.kind), "%s", kind ? kind : "?");
    r.device = device;
    r.requested = requested;
    r.free_before = free_before;
    r.free_after = free_after;
    r.ok = ok ? 1 : 0;
}

static void hga_cuda_alloc_dump(const char * why) {
    if (!hga_cuda_alloc_ledger_enabled()) {
        return;
    }
    hga_cuda_alloc_rec rows[20];
    uint64_t last = 0;
    {
        std::lock_guard<std::mutex> lock(g_hga_cuda_alloc_mutex);
        memcpy(rows, g_hga_cuda_alloc_ring, sizeof(rows));
        last = g_hga_cuda_alloc_seq;
    }
    const uint64_t first = last > 20 ? last - 19 : 1;
    fprintf(stderr, "hga-cuda-alloc: BEGIN why=%s last=%" PRIu64 "\n",
            why ? why : "?", last);
    fprintf(stderr, "hga-cuda-alloc:  seq kind                     dev request_MiB free_before free_after delta_MiB ok\n");
    for (uint64_t seq = first; seq <= last; ++seq) {
        const hga_cuda_alloc_rec & r = rows[(seq - 1) % 20];
        if (r.seq != seq) {
            continue;
        }
        const double mib = 1024.0 * 1024.0;
        fprintf(stderr, "hga-cuda-alloc: %4" PRIu64 " %-24s %3d %11.2f %11.2f %10.2f %9.2f %d\n",
                r.seq, r.kind, r.device, r.requested / mib,
                r.free_before / mib, r.free_after / mib,
                ((double) r.free_before - (double) r.free_after) / mib, r.ok);
    }
    fprintf(stderr, "hga-cuda-alloc: END\n");
    fflush(stderr);
}

'''
    if error_anchor not in text:
        die("ggml-cuda.cu: error hook anchor missing for allocation ledger")
    text = text.replace(error_anchor, ledger + error_anchor, 1)

    old = """    GGML_LOG_ERROR("  %s\\n", stmt);
    // abort with GGML_ABORT to get a stack trace
"""
    new = """    GGML_LOG_ERROR("  %s\\n", stmt);
    hga_cuda_alloc_dump("CUDA_CHECK");
    // abort with GGML_ABORT to get a stack trace
"""
    if old not in text:
        die("ggml-cuda.cu: CUDA error body missing for allocation ledger")
    text = text.replace(old, new, 1)

    old = "static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {"
    new = "static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device, const char * who) {"
    if old not in text:
        die("ggml-cuda.cu: device malloc signature missing for allocation ledger")
    text = text.replace(old, new, 1)
    old = """    ggml_cuda_set_device(device);
    cudaError_t err;
"""
    new = """    ggml_cuda_set_device(device);
    const bool trace_alloc = hga_cuda_alloc_ledger_enabled();
    const size_t free_before = trace_alloc ? hga_cuda_free_bytes() : 0;
    cudaError_t err;
"""
    if old not in text:
        die("ggml-cuda.cu: device malloc body missing for allocation ledger")
    text = text.replace(old, new, 1)
    old = """    }
    return err;
}

#if defined(GGML_USE_HIP)
"""
    new = """    }
    if (trace_alloc) {
        hga_cuda_alloc_record(who, device, size, free_before,
                hga_cuda_free_bytes(), err == cudaSuccess);
        if (err != cudaSuccess) {
            hga_cuda_alloc_dump(who);
        }
    }
    return err;
}

#if defined(GGML_USE_HIP)
"""
    if old not in text:
        die("ggml-cuda.cu: device malloc return missing for allocation ledger")
    text = text.replace(old, new, 1)
    text = text.replace(
        "ggml_cuda_device_malloc(&ptr, look_ahead_size, device);",
        'ggml_cuda_device_malloc(&ptr, look_ahead_size, device, "legacy-pool");',
    )
    text = text.replace(
        "ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);",
        'ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device,\n'
        '            ggml_backend_hga_alloc_scope_get() && ggml_backend_hga_alloc_scope_get()[0]\n'
        '                ? ggml_backend_hga_alloc_scope_get() : "backend-buffer");',
    )

    old = """            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));
"""
    new = """            CUmemGenericAllocationHandle handle;
            const bool trace_alloc = hga_cuda_alloc_ledger_enabled();
            const size_t free_before = trace_alloc ? hga_cuda_free_bytes() : 0;
            const CUresult create_rc = cuMemCreate(&handle, reserve_size, &prop, 0);
            if (trace_alloc) {
                const size_t free_after = hga_cuda_free_bytes();
                hga_cuda_alloc_record("vmm-grow", device, reserve_size, free_before,
                        free_after, create_rc == CUDA_SUCCESS);
                static bool low_memory_dumped = false;
                if (!low_memory_dumped && free_after < 768ull * 1024 * 1024) {
                    low_memory_dumped = true;
                    hga_cuda_alloc_dump("low-memory VMM grow");
                }
                if (create_rc != CUDA_SUCCESS) {
                    hga_cuda_alloc_dump("vmm-grow");
                }
            }
            CU_CHECK(create_rc);
"""
    if old not in text:
        die("ggml-cuda.cu: VMM cuMemCreate anchor missing for allocation ledger")
    text = text.replace(old, new, 1)

    old = """                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
"""
    new = """                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\\n", __func__, node->name, ggml_op_name(node->op));
                    hga_cuda_alloc_dump("unsupported CUDA op");
                }
                GGML_ASSERT(ok);
"""
    if old not in text:
        die("ggml-cuda.cu: unsupported-op anchor missing for allocation ledger")
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    print("  patched ggml-cuda.cu physical allocation ledger")


def patch_cuda_alloc_attribution(root: Path) -> None:
    """Label the major ggml CUDA buffers in the physical allocation ledger."""
    header = root / "ggml" / "include" / "ggml-backend.h"
    htxt = header.read_text(encoding="utf-8")
    scope_decl = (
        "    // HGA diagnostic allocation scope (thread-local; no-op unless consumed by a backend).\n"
        "    GGML_API void         ggml_backend_hga_alloc_scope_set(const char * scope);\n"
        "    GGML_API const char * ggml_backend_hga_alloc_scope_get(void);\n"
    )
    if "ggml_backend_hga_alloc_scope_set" not in htxt:
        anchor = "    GGML_API const char *          ggml_backend_buft_name          (ggml_backend_buffer_type_t buft);\n"
        if anchor not in htxt:
            die("ggml-backend.h: buffer type API anchor missing for allocation attribution")
        header.write_text(htxt.replace(anchor, anchor + scope_decl, 1), encoding="utf-8")
        print("  patched ggml-backend.h allocation scope API")

    backend = root / "ggml" / "src" / "ggml-backend.cpp"
    btxt = backend.read_text(encoding="utf-8")
    if "g_hga_backend_alloc_scope" not in btxt:
        anchor = "// backend buffer type\n\n"
        impl = (
            "// HGA allocation attribution is opt-in at the CUDA ledger consumer.\n"
            "// A TLS scope keeps simultaneous model/context construction independent.\n"
            "static thread_local const char * g_hga_backend_alloc_scope = nullptr;\n\n"
            "void ggml_backend_hga_alloc_scope_set(const char * scope) {\n"
            "    g_hga_backend_alloc_scope = scope;\n"
            "}\n\n"
            "const char * ggml_backend_hga_alloc_scope_get(void) {\n"
            "    return g_hga_backend_alloc_scope;\n"
            "}\n\n"
        )
        if anchor not in btxt:
            die("ggml-backend.cpp: buffer type anchor missing for allocation attribution")
        backend.write_text(btxt.replace(anchor, anchor + impl, 1), encoding="utf-8")
        print("  patched ggml-backend.cpp allocation scope TLS")

    alloc = root / "ggml" / "src" / "ggml-alloc.c"
    atxt = alloc.read_text(encoding="utf-8")
    old = """        size_t chunk_size = talloc->chunks[n]->max_size;
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
"""
    new = """        size_t chunk_size = talloc->chunks[n]->max_size;
        const char * hga_prev_scope = ggml_backend_hga_alloc_scope_get();
        ggml_backend_hga_alloc_scope_set("compute-graph");
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        ggml_backend_hga_alloc_scope_set(hga_prev_scope);
"""
    if "ggml_backend_hga_alloc_scope_set(\"compute-graph\")" not in atxt:
        if old not in atxt:
            die("ggml-alloc.c: compute graph allocation anchor missing")
        alloc.write_text(atxt.replace(old, new, 1), encoding="utf-8")
        print("  patched ggml-alloc.c compute graph allocation attribution")

    def scoped_ctx_alloc(path: Path, needle: str, label: str, expected: int = 1) -> None:
        txt = path.read_text(encoding="utf-8")
        marker = f'ggml_backend_hga_alloc_scope_set("{label}")'
        if marker in txt:
            return
        count = txt.count(needle)
        if count != expected:
            die(f"{path.name}: expected {expected} allocation anchors for {label}, found {count}")
        replacement = (
            f'ggml_backend_hga_alloc_scope_set("{label}");\n'
            f'            {needle}\n'
            '            ggml_backend_hga_alloc_scope_set(nullptr);'
        )
        path.write_text(txt.replace(needle, replacement, expected), encoding="utf-8")
        print(f"  patched {path.name} allocation attribution: {label}")

    scoped_ctx_alloc(
        root / "src" / "llama-kv-cache.cpp",
        "buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer",
        "kv-cache",
    )
    scoped_ctx_alloc(
        root / "src" / "llama-model.cpp",
        "buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft); // real buffer",
        "model-weights",
    )
    scoped_ctx_alloc(
        root / "src" / "llama-memory-recurrent.cpp",
        "ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);",
        "recurrent-cache",
    )


def patch_async_eval_events(root: Path) -> None:
    """Let HGA callbacks express CUDA dependencies without host synchronization."""
    header = root / "ggml" / "include" / "ggml-backend.h"
    source = root / "ggml" / "src" / "ggml-backend.cpp"
    cuda = root / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.cu"

    setter = (
        "    GGML_API void                 ggml_backend_sched_set_eval_callback"
        "(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data);\n"
    )
    setter_async = (
        "    GGML_API void                 ggml_backend_sched_set_eval_callback_async"
        "(ggml_backend_sched_t sched, bool async);\n"
    )
    text = header.read_text(encoding="utf-8")
    if "ggml_backend_sched_set_eval_callback_async" not in text:
        if setter not in text:
            die(f"scheduler callback declaration not found in {header}")
        header.write_text(text.replace(setter, setter + setter_async, 1), encoding="utf-8")
        print(f"  patched {header} async callback declaration")
    else:
        print(f"  already patched: {header.name} async callback declaration")

    text = source.read_text(encoding="utf-8")
    changed = False
    if "bool callback_eval_async;" not in text:
        anchor = "    void * callback_eval_user_data;\n"
        if anchor not in text:
            die(f"scheduler callback fields not found in {source}")
        text = text.replace(anchor, anchor + "    bool callback_eval_async;\n", 1)
        changed = True

    whole_old = "        if (!sched->callback_eval) {\n"
    whole_new = "        if (!sched->callback_eval || sched->callback_eval_async) {\n"
    if whole_new not in text:
        if whole_old not in text:
            die(f"scheduler callback branch not found in {source}")
        text = text.replace(whole_old, whole_new, 1)
        changed = True

    sync_old = (
        "                // TODO: pass backend to the callback, then the user can decide if they want to synchronize\n"
        "                ggml_backend_synchronize(split_backend);\n"
    )
    sync_new = (
        "                // HGA can order its CUDA copy and compute streams with events.\n"
        "                // Other callback users retain the historical host synchronization.\n"
        "                if (!sched->callback_eval_async) {\n"
        "                    ggml_backend_synchronize(split_backend);\n"
        "                }\n"
    )
    if sync_new not in text:
        if sync_old not in text:
            die(f"scheduler callback synchronization not found in {source}")
        text = text.replace(sync_old, sync_new, 1)
        changed = True

    setter_impl = (
        "void ggml_backend_sched_set_eval_callback_async(ggml_backend_sched_t sched, bool async) {\n"
        "    GGML_ASSERT(sched);\n"
        "    sched->callback_eval_async = async;\n"
        "}\n"
    )
    if "void ggml_backend_sched_set_eval_callback_async" not in text:
        anchor = (
            "void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, "
            "ggml_backend_sched_eval_callback callback, void * user_data) {\n"
            "    GGML_ASSERT(sched);\n"
            "    sched->callback_eval = callback;\n"
            "    sched->callback_eval_user_data = user_data;\n"
            "}\n"
        )
        if anchor not in text:
            die(f"scheduler callback setter not found in {source}")
        text = text.replace(anchor, anchor + "\n" + setter_impl, 1)
        changed = True
    if changed:
        source.write_text(text, encoding="utf-8")
        print(f"  patched {source} async callback mode")
    else:
        print(f"  already patched: {source.name} async callback mode")

    common = root / "ggml" / "src" / "ggml-cuda" / "common.cuh"
    text = common.read_text(encoding="utf-8")
    legacy_fields = (
        "    void (* hga_node_callback)(ggml_tensor *, void *) = nullptr;\n"
        "    void * hga_node_callback_user = nullptr;\n"
    )
    if legacy_fields in text:
        common.write_text(text.replace(legacy_fields, "", 1), encoding="utf-8")
        print(f"  removed legacy per-context HGA callback state from {common}")

    text = cuda.read_text(encoding="utf-8")
    anchor = (
        "static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {\n"
        "    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;\n"
        "\n"
        "    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));\n"
        "}\n"
    )
    bridge = (
        "\nextern \"C\" GGML_API bool ggml_cuda_hga_event_record(ggml_backend_t backend, void * event) {\n"
        "    if (!ggml_backend_is_cuda(backend) || event == nullptr) {\n"
        "        return false;\n"
        "    }\n"
        "    ggml_backend_cuda_context * cuda_ctx = g_hga_cuda_active_context != nullptr ?\n"
        "            g_hga_cuda_active_context : (ggml_backend_cuda_context *) backend->context;\n"
        "    return cudaEventRecord((cudaEvent_t) event, cuda_ctx->stream()) == cudaSuccess;\n"
        "}\n"
        "\n"
        "extern \"C\" GGML_API bool ggml_cuda_hga_event_wait(ggml_backend_t backend, void * event) {\n"
        "    if (!ggml_backend_is_cuda(backend) || event == nullptr) {\n"
        "        return false;\n"
        "    }\n"
        "    ggml_backend_cuda_context * cuda_ctx = g_hga_cuda_active_context != nullptr ?\n"
        "            g_hga_cuda_active_context : (ggml_backend_cuda_context *) backend->context;\n"
        "    return cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t) event, 0) == cudaSuccess;\n"
        "}\n"
    )
    changed = False
    if "ggml_cuda_hga_event_record" not in text:
        if anchor not in text:
            die(f"CUDA backend event recorder not found in {cuda}")
        text = text.replace(anchor, anchor + bridge, 1)
        changed = True

    callback_globals = (
        "static void (* g_hga_cuda_node_callback)(ggml_tensor *, void *) = nullptr;\n"
        "static void * g_hga_cuda_node_callback_user = nullptr;\n"
        "static thread_local ggml_backend_cuda_context * g_hga_cuda_active_context = nullptr;\n"
        "\n"
    )
    callback_globals_v1 = (
        "static void (* g_hga_cuda_node_callback)(ggml_tensor *, void *) = nullptr;\n"
        "static void * g_hga_cuda_node_callback_user = nullptr;\n"
        "\n"
    )
    eval_anchor = "static void ggml_cuda_graph_evaluate_and_capture("
    if callback_globals_v1 in text:
        text = text.replace(callback_globals_v1, callback_globals, 1)
        changed = True
    elif callback_globals not in text:
        if eval_anchor not in text:
            die(f"CUDA graph evaluator not found in {cuda}")
        text = text.replace(eval_anchor, callback_globals + eval_anchor, 1)
        changed = True

    setter = (
        "\nextern \"C\" GGML_API bool ggml_cuda_hga_set_node_callback(\n"
        "        ggml_backend_t backend, void (* callback)(ggml_tensor *, void *), void * user) {\n"
        "    if (!ggml_backend_is_cuda(backend)) {\n"
        "        return false;\n"
        "    }\n"
        "    g_hga_cuda_node_callback = callback;\n"
        "    g_hga_cuda_node_callback_user = user;\n"
        "    return true;\n"
        "}\n"
    )
    setter_context = (
        "\nextern \"C\" GGML_API bool ggml_cuda_hga_set_node_callback(\n"
        "        ggml_backend_t backend, void (* callback)(ggml_tensor *, void *), void * user) {\n"
        "    if (!ggml_backend_is_cuda(backend)) {\n"
        "        return false;\n"
        "    }\n"
        "    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;\n"
        "    cuda_ctx->hga_node_callback = callback;\n"
        "    cuda_ctx->hga_node_callback_user = user;\n"
        "    return true;\n"
        "}\n"
    )
    if setter_context in text:
        text = text.replace(setter_context, setter, 1)
        changed = True
    elif "ggml_cuda_hga_set_node_callback" not in text:
        wait_impl = (
            "extern \"C\" GGML_API bool ggml_cuda_hga_event_wait(ggml_backend_t backend, void * event) {\n"
            "    if (!ggml_backend_is_cuda(backend) || event == nullptr) {\n"
            "        return false;\n"
            "    }\n"
            "    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;\n"
            "    return cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t) event, 0) == cudaSuccess;\n"
            "}\n"
        )
        wait_impl_active = (
            "extern \"C\" GGML_API bool ggml_cuda_hga_event_wait(ggml_backend_t backend, void * event) {\n"
            "    if (!ggml_backend_is_cuda(backend) || event == nullptr) {\n"
            "        return false;\n"
            "    }\n"
            "    ggml_backend_cuda_context * cuda_ctx = g_hga_cuda_active_context != nullptr ?\n"
            "            g_hga_cuda_active_context : (ggml_backend_cuda_context *) backend->context;\n"
            "    return cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t) event, 0) == cudaSuccess;\n"
            "}\n"
        )
        if wait_impl in text:
            text = text.replace(wait_impl, wait_impl + setter, 1)
        elif wait_impl_active in text:
            # Fresh upstream has just received the record/wait bridge above,
            # whose waiter honors the active CUDA graph context.
            text = text.replace(wait_impl_active, wait_impl_active + setter, 1)
        else:
            die(f"HGA CUDA event waiter not found in {cuda}")
        changed = True

    inline_old = "                prev_i = i;\n"
    inline_new = (
        "                if (g_hga_cuda_node_callback != nullptr) {\n"
        "                    g_hga_cuda_active_context = cuda_ctx;\n"
        "                    g_hga_cuda_node_callback(node, g_hga_cuda_node_callback_user);\n"
        "                    g_hga_cuda_active_context = nullptr;\n"
        "                }\n"
        "\n"
        "                prev_i = i;\n"
    )
    inline_context = (
        "                if (cuda_ctx->hga_node_callback != nullptr) {\n"
        "                    cuda_ctx->hga_node_callback(node, cuda_ctx->hga_node_callback_user);\n"
        "                }\n"
        "\n"
        "                prev_i = i;\n"
    )
    inline_global_v1 = (
        "                if (g_hga_cuda_node_callback != nullptr) {\n"
        "                    g_hga_cuda_node_callback(node, g_hga_cuda_node_callback_user);\n"
        "                }\n"
        "\n"
        "                prev_i = i;\n"
    )
    if inline_context in text:
        text = text.replace(inline_context, inline_new, 1)
        changed = True
    elif inline_global_v1 in text:
        text = text.replace(inline_global_v1, inline_new, 1)
        changed = True
    elif inline_new not in text:
        if inline_old not in text:
            die(f"CUDA node loop callback point not found in {cuda}")
        text = text.replace(inline_old, inline_new, 1)
        changed = True

    event_context = (
        "    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;\n"
        "    return cudaEventRecord((cudaEvent_t) event, cuda_ctx->stream()) == cudaSuccess;\n"
    )
    event_active = (
        "    ggml_backend_cuda_context * cuda_ctx = g_hga_cuda_active_context != nullptr ?\n"
        "            g_hga_cuda_active_context : (ggml_backend_cuda_context *) backend->context;\n"
        "    return cudaEventRecord((cudaEvent_t) event, cuda_ctx->stream()) == cudaSuccess;\n"
    )
    if event_context in text:
        text = text.replace(event_context, event_active, 1)
        changed = True
    wait_context = (
        "    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;\n"
        "    return cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t) event, 0) == cudaSuccess;\n"
    )
    wait_active = (
        "    ggml_backend_cuda_context * cuda_ctx = g_hga_cuda_active_context != nullptr ?\n"
        "            g_hga_cuda_active_context : (ggml_backend_cuda_context *) backend->context;\n"
        "    return cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t) event, 0) == cudaSuccess;\n"
    )
    if wait_context in text:
        text = text.replace(wait_context, wait_active, 1)
        changed = True
    if changed:
        cuda.write_text(text, encoding="utf-8")
        print(f"  patched {cuda} HGA raw-event/inline-node bridge")
    else:
        print(f"  already patched: {cuda.name} HGA raw-event/inline-node bridge")


def patch_cuda_i8_cast(root: Path) -> None:
    """Enable raw scalar I8 -> F16 copies for compact HGA KV staging."""
    cpy = root / "ggml" / "src" / "ggml-cuda" / "cpy.cu"
    once(
        cpy,
        "    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {\n",
        """    } else if (src0->type == GGML_TYPE_I8 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<int8_t, half>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<int8_t, half>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
""",
        after=False,
        marker="ggml_cpy_scalar_contiguous_cuda<int8_t, half>",
    )

    cuda = root / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.cu"
    once(
        cuda,
        "                ggml_type src1_type = op->src[1]->type;\n",
        """                if (src0_type == GGML_TYPE_I8 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
""",
        marker="src0_type == GGML_TYPE_I8 && src1_type == GGML_TYPE_F16",
    )


def patch_hga_ubatch_pad(root: Path) -> None:
    """Remove legacy physical PREFILL padding that corrupts recurrent KV state.

    A padded ubatch puts repeated dummy tokens into the recurrent cache.  That
    cannot be rolled back once the real suffix is longer than ``n_rs_seq``;
    persistent server slots then abort on their first cache-preserving turn.
    HGA itself already receives the real logical batch descriptor, so graph
    shape reuse must never be implemented by appending model-visible tokens.
    """
    legacy_blocks = (
        """        // HGA: pad a short *whole* batch (dummy n=2) to n_ubatch. Do not pad
        // leftover prompt tokens: GDN cannot seq_rm more than n_rs_seq.
        if (ubatches.size() == 1 && n_ubatch >= 8) {
            for (auto & ub : ubatches) {
                if (ub.n_tokens > 0 && ub.n_tokens < n_ubatch) {
                    hga_ubatch_pad_to(ub, n_ubatch);
                }
            }
        }

""",
        """        // HGA: pad a short *whole* batch (dummy n=2) to n_ubatch. Do not pad
        // leftover 464 of a long prompt: GDN cannot seq_rm 48 pad tokens
        // (rollback only n_rs_seq). 512-token chunks reuse via can_reuse.
        if (ubatches.size() == 1 && n_ubatch >= 8) {
            for (auto & ub : ubatches) {
                if (ub.n_tokens > 0 && ub.n_tokens < n_ubatch) {
                    hga_ubatch_pad_to(ub, n_ubatch);
                }
            }
        }

""",
        """        // HGA: every ubatch is exactly n_ubatch tokens so the ggml graph (and
        // CUDA VMM scratch) is one size. Dummy n=2 and leftover prompt tokens
        // are padded; pad tokens repeat the last real token, output=0.
        for (auto & ub : ubatches) {
            if (ub.n_tokens > 0 && ub.n_tokens < n_ubatch) {
                hga_ubatch_pad_to(ub, n_ubatch);
            }
        }
""",
    )
    for rel in (
        "src/llama-memory-hybrid.cpp",
        "src/llama-memory-hybrid-iswa.cpp",
    ):
        path = root / rel
        once(path, '#include "llama-context.h"\n', '#include "llama-hga.h"\n')
        text = path.read_text(encoding="utf-8")
        changed = False
        for block in legacy_blocks:
            if block in text:
                text = text.replace(block, "", 1)
                changed = True
        if changed:
            path.write_text(text, encoding="utf-8")
            print(f"  removed unsafe HGA physical prefill padding from {path.name}")
        else:
            print(f"  already patched: {path.name} preserves real prefill token counts")


def patch_hga_skip_llama_attn_kv(root: Path) -> None:
    """HGA does not use llama kq_mask / cpy_k / cpy_v. Leave them out of the ggml graph.

    Those inputs were kept only so set_input would not hit a null buffer. They
    are unused (attention is the CPU HGA op) but kq_mask.ne[0]==n_kv still
    keyed can_reuse, so VERIFY rebuilt at every 256-token KV bucket. Same
    tensors would sit in a CUDA graph capture if graphs were enabled.
    """
    graph_cpp = root / "src" / "llama-graph.cpp"
    replace(
        graph_cpp,
        """void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
    mctx->get_attn()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    if (inp_attn->self_k_rot) {
        mctx->get_attn()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        mctx->get_attn()->set_input_v_rot(inp_attn->self_v_rot);
    }
""",
        """void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    // HGA: softmax K/V live in the CPU session. Skip llama k_idxs/v_idxs/kq_mask.
    if (inp_attn && !cparams.hga_runtime) {
        mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
        mctx->get_attn()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);

        mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

        if (inp_attn->self_k_rot) {
            mctx->get_attn()->set_input_k_rot(inp_attn->self_k_rot);
        }

        if (inp_attn->self_v_rot) {
            mctx->get_attn()->set_input_v_rot(inp_attn->self_v_rot);
        }
    }
""",
    )
    replace(
        graph_cpp,
        """    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();
""",
        """    bool res = true;

    if (inp_attn && !params.cparams.hga_runtime) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

        res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);
    }

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();
""",
    )
    replace(
        graph_cpp,
        """    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();
""",
        """    // head/rs_z are GDN ring inputs, not graph topology. Checking them
    // rebuilt the 512-token prefill graph on every ubatch.
    if (!params.cparams.hga_runtime) {
        res &= inp_rs->head == mctx->get_recr()->get_head();
        res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();
    }
""",
    )
    replace(
        graph_cpp,
        """    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);
""",
        """    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    std::unique_ptr<llm_graph_input_attn_kv> inp_attn;
    if (!cparams.hga_runtime) {
        inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());
    }

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);
""",
    )


def patch_generate_k1(root: Path) -> None:
    """Generate/VERIFY and MTP always use one n=K+1 graph (pad short batches)."""
    replace(
        root / "src" / "llama-batch.cpp",
        "        GGML_ASSERT(n_ubatch > n_keep_tail);\n",
        "        GGML_ASSERT(n_ubatch >= n_keep_tail);\n",
    )

    # A short final verify batch is padded to K+1. Its padding cleanup can be
    # followed immediately by speculative rejection cleanup, before a graph
    # consumes the first recurrent rollback. Compose both bounded requests.
    recurrent_cpp = root / "src" / "llama-memory-recurrent.cpp"
    recurrent_text = recurrent_cpp.read_text(encoding="utf-8")
    rollback_new = """            // partial rollback via per-token snapshot index (bounded by n_rs_seq)
            if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
                const llama_pos rollback = cell.pos - (p0 - 1);
                // Padding cleanup and speculative rejection can both roll back
                // before the next graph consumes the first request. Compose
                // the two requests into the older snapshot plane.
                const llama_pos total = (llama_pos) rs_idx[seq_id] + rollback;
                if (rollback >= 1 && total <= (llama_pos) n_rs_seq) {
                    set_rs_idx(seq_id, (uint32_t) total);
                    cell.pos = p0 - 1;
                    return true;
                }
                return false;
            }
"""
    rollback_guarded = """            // partial rollback via per-token snapshot index (bounded by n_rs_seq)
            if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
                const llama_pos rollback = cell.pos - (p0 - 1);
                // pending rollback is single-use
                const bool pending = rs_idx[seq_id] != 0;
                if (!pending && rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
                    set_rs_idx(seq_id, (uint32_t) rollback);
                    cell.pos = p0 - 1;
                    return true;
                }
                return false;
            }
"""
    rollback_single = """            // partial rollback via per-token snapshot index (bounded by n_rs_seq)
            if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
                const llama_pos rollback = cell.pos - (p0 - 1);
                if (rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
                    set_rs_idx(seq_id, (uint32_t) rollback);
                    cell.pos = p0 - 1;
                    return true;
                }
                return false;
            }
"""
    if rollback_new in recurrent_text:
        print("  already patched: llama-memory-recurrent.cpp composable rollback")
    elif rollback_guarded in recurrent_text:
        recurrent_cpp.write_text(
            recurrent_text.replace(rollback_guarded, rollback_new, 1), encoding="utf-8"
        )
        print("  patched llama-memory-recurrent.cpp composable rollback")
    elif rollback_single in recurrent_text:
        recurrent_cpp.write_text(
            recurrent_text.replace(rollback_single, rollback_new, 1), encoding="utf-8"
        )
        print("  patched llama-memory-recurrent.cpp composable rollback")
    else:
        die(f"recurrent rollback block not found in {recurrent_cpp}")

    decode_cpp = root / "src" / "llama-context.cpp"
    replace(
        decode_cpp,
        """    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\\n", __func__);
        return -1;
    }
""",
        """    hga_ubatch_pad_reset();
    hga_swap_ensure((uint32_t) batch_inp.n_tokens);

    hga_llama_batch_pad hga_pad;
    const llama_batch * hga_batch = &batch_inp;
    if (hga_maybe_pad_decode_batch(cparams, batch_inp, (uint32_t) n_embd, hga_pad)) {
        hga_batch = &hga_pad.batch;
    }

    if (!balloc->init(*hga_batch, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\\n", __func__);
        return -1;
    }
""",
    )
    replace(
        decode_cpp,
        """        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());
""",
        """        {
            const uint32_t n_real = hga_ubatch_padded_n_real();
            if (n_real > 0 && n_real < ubatch.n_tokens && ubatch.pos && memory) {
                const llama_pos pad_pos = ubatch.pos[n_real];
                if (ubatch.n_seq_id && ubatch.seq_id && ubatch.n_seq_id[0] > 0) {
                    for (int32_t s = 0; s < ubatch.n_seq_id[0]; ++s) {
                        memory->seq_rm(ubatch.seq_id[0][s], pad_pos, -1);
                    }
                } else {
                    memory->seq_rm(0, pad_pos, -1);
                }
                hga_ubatch_pad_reset();
            }
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());
""",
    )

    spec = root / "common" / "speculative.cpp"
    replace(
        spec,
        "    result.n_outputs_max = params.n_parallel;\n"
        "    result.n_outputs_max_per_seq = 1;\n",
        "    result.n_outputs_max = params.n_parallel;\n"
        "    result.n_outputs_max_per_seq = 1;\n"
        "    {\n"
        "        const bool spec_mtp_out = std::any_of(\n"
        "            params.speculative.types.begin(), params.speculative.types.end(),\n"
        "            [](common_speculative_type t) { return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP; });\n"
        "        if (spec_mtp_out) {\n"
        "            const int32_t per_seq = std::max(1, params_spec.n_max + 1);\n"
        "            result.n_outputs_max = params.n_parallel * per_seq;\n"
        "            result.n_outputs_max_per_seq = per_seq;\n"
        "        }\n"
        "    }\n",
    )
    replace(
        spec,
        "                common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);\n",
        "                common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] },\n"
        "                        batch_in.logits ? batch_in.logits[k] : 0);\n",
    )
    replace(
        spec,
        "        if (chain_heads) {\n"
        "            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);\n"
        "\n"
        "            chain_h.assign(n_seq, {});\n"
        "            for (auto & c : chain_h) {\n"
        "                c.reserve((size_t) (this->params.n_max + 1) * n_embd);\n"
        "            }\n"
        "        }\n",
        "        chain_h.assign(n_seq, {});\n"
        "        for (auto & c : chain_h) {\n"
        "            c.reserve((size_t) (this->params.n_max + 1) * n_embd);\n"
        "        }\n"
        "        if (chain_heads) {\n"
        "            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);\n"
        "        }\n",
    )
    replace(
        spec,
        "    void draft(common_speculative_draft_params_vec & dparams) override {\n"
        "        auto & ctx_dft = params.ctx_dft;\n"
        "\n"
        "        common_batch_clear(batch);\n"
        "\n"
        "        // keep track of which sequences are still drafting\n"
        "        int n_drafting = 0;\n"
        "        std::vector<bool> drafting(n_seq);\n"
        "\n"
        "        const size_t row_bytes = (size_t) n_embd * sizeof(float);\n",
        "    void draft(common_speculative_draft_params_vec & dparams) override {\n"
        "        auto & ctx_dft = params.ctx_dft;\n"
        "        // HGA: MTP attention is the same full-attn block as the trunk. Incremental\n"
        "        // n=1 does not give it the other tokens in the verify/draft window, and it\n"
        "        // rebuilds a different ggml graph. Recompute the whole window (pad to K+1).\n"
        "        const bool recompute_window = !is_mem_shared;\n"
        "\n"
        "        common_batch_clear(batch);\n"
        "\n"
        "        // keep track of which sequences are still drafting\n"
        "        int n_drafting = 0;\n"
        "        std::vector<bool> drafting(n_seq);\n"
        "\n"
        "        const size_t row_bytes = (size_t) n_embd * sizeof(float);\n",
    )
    replace(
        spec,
        "            if (chain_heads) {\n"
        "                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());\n"
        "            }\n"
        "        }\n"
        "\n"
        "        int i = 0;\n",
        "            if (recompute_window) {\n"
        "                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());\n"
        "            }\n"
        "        }\n"
        "\n"
        "        int i = 0;\n",
    )
    replace(
        spec,
        "            if (chain_heads) {\n"
        "                auto * mem_dft = llama_get_memory(ctx_dft);\n"
        "                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {\n"
        "                    if (drafting[seq_id]) {\n"
        "                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].n_past, -1);\n"
        "                    }\n"
        "                }\n"
        "                llama_set_nextn_layer_offset(ctx_dft, i);\n"
        "            }\n",
        "            if (recompute_window) {\n"
        "                auto * mem_dft = llama_get_memory(ctx_dft);\n"
        "                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {\n"
        "                    if (drafting[seq_id]) {\n"
        "                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].n_past, -1);\n"
        "                    }\n"
        "                }\n"
        "                if (chain_heads) {\n"
        "                    llama_set_nextn_layer_offset(ctx_dft, i);\n"
        "                }\n"
        "            }\n",
    )
    replace(
        spec,
        "                if (chain_heads) {\n"
        "                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546\n"
        "                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);\n"
        "\n"
        "                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far\n"
        "                    for (int t = 0; t < n_rows; ++t) {\n"
        "                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];\n"
        "                        common_batch_add(batch, tok, dp.n_past + t, { seq_id }, t == n_rows - 1);\n"
        "                        std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd,\n"
        "                                    chain_h[seq_id].data() + (size_t) t * n_embd, row_bytes);\n"
        "                    }\n"
        "                } else if (is_mem_shared) {\n",
        "                if (recompute_window) {\n"
        "                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);\n"
        "\n"
        "                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far\n"
        "                    for (int t = 0; t < n_rows; ++t) {\n"
        "                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];\n"
        "                        common_batch_add(batch, tok, dp.n_past + t, { seq_id }, true);\n"
        "                        std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd,\n"
        "                                    chain_h[seq_id].data() + (size_t) t * n_embd, row_bytes);\n"
        "                    }\n"
        "                } else if (is_mem_shared) {\n",
    )


def patch_spec_step_prof(root: Path) -> None:
    """Per-step wall times in llama-speculative-simple (draft / verify / process / sample)."""
    path = root / "examples" / "speculative-simple" / "speculative-simple.cpp"
    text = path.read_text(encoding="utf-8")
    if "hga-prof spec step" in text:
        print("  already patched: speculative-simple step prof")
        return
    dec_start_anchor = "    const auto t_dec_start = ggml_time_us();\n\n\n    while (true) {\n"
    if dec_start_anchor not in text:
        # Current upstream keeps one blank line here; older versions kept two.
        dec_start_anchor = "    const auto t_dec_start = ggml_time_us();\n\n    while (true) {\n"
    replace(
        path,
        dec_start_anchor,
        """    const auto t_dec_start = ggml_time_us();

    int n_steps = 0;
    int64_t t_draft_sum = 0;
    int64_t t_verify_sum = 0;
    int64_t t_process_sum = 0;
    int64_t t_sample_sum = 0;
    int64_t t_rest_sum = 0;

    while (true) {
        const int64_t t_step0 = ggml_time_us();
        int64_t t_draft_us = 0;
        int64_t t_verify_us = 0;
        int64_t t_process_us = 0;
        int64_t t_sample_us = 0;
""",
    )
    replace(
        path,
        "            common_speculative_draft(spec);\n",
        "            const int64_t t_draft0 = ggml_time_us();\n"
        "            common_speculative_draft(spec);\n"
        "            t_draft_us = ggml_time_us() - t_draft0;\n",
    )
    replace(
        path,
        """            llama_decode(ctx_tgt, batch_tgt);
        }

        // feed the batch to the speculative implementation(s) - this drives the draft model, MTP, Eagle3, etc.
        if (!common_speculative_process(spec, batch_tgt)) {
            LOG_ERR("%s", "failed to process speculative batch\\n");
            break;
        }
""",
        """            const int64_t t_verify0 = ggml_time_us();
            llama_decode(ctx_tgt, batch_tgt);
            t_verify_us = ggml_time_us() - t_verify0;
        }

        // feed the batch to the speculative implementation(s) - this drives the draft model, MTP, Eagle3, etc.
        {
            const int64_t t_process0 = ggml_time_us();
            if (!common_speculative_process(spec, batch_tgt)) {
                LOG_ERR("%s", "failed to process speculative batch\\n");
                break;
            }
            t_process_us = ggml_time_us() - t_process0;
        }
""",
    )
    replace(
        path,
        "        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);\n",
        "        const int64_t t_sample0 = ggml_time_us();\n"
        "        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);\n"
        "        t_sample_us = ggml_time_us() - t_sample0;\n",
    )
    replace(
        path,
        """        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }
""",
        """        {
            const int64_t t_step = ggml_time_us() - t_step0;
            const int64_t t_known = t_draft_us + t_verify_us + t_process_us + t_sample_us;
            const int64_t t_rest = t_step > t_known ? t_step - t_known : 0;
            n_steps++;
            t_draft_sum += t_draft_us;
            t_verify_sum += t_verify_us;
            t_process_sum += t_process_us;
            t_sample_sum += t_sample_us;
            t_rest_sum += t_rest;
            fprintf(stderr,
                    "hga-prof spec step %d: wall=%.1f ms  draft=%.1f verify=%.1f process=%.1f sample=%.1f rest=%.1f  "
                    "n_draft=%zu n_acc=%zu n_tgt_batch=%d\\n",
                    n_steps, t_step / 1000.0, t_draft_us / 1000.0, t_verify_us / 1000.0,
                    t_process_us / 1000.0, t_sample_us / 1000.0, t_rest / 1000.0,
                    n_draft, ids.size(), batch_tgt.n_tokens);
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }
""",
    )
    replace(
        path,
        '    LOG_INF("accept    = %.3f%%\\n", 100.0f * n_accept / n_drafted);\n',
        """    LOG_INF("accept    = %.3f%%\\n", 100.0f * n_accept / n_drafted);

    {
        const double gen_ms = (t_dec_end - t_dec_start) / 1000.0;
        const double inv = gen_ms > 0.0 ? 100.0 / gen_ms : 0.0;
        fprintf(stderr,
                "hga-prof spec TOTAL steps=%d  generate=%.1f ms (%.2f steps/s)  "
                "draft=%.1f (%.0f%%) verify=%.1f (%.0f%%) process=%.1f (%.0f%%) "
                "sample=%.1f (%.0f%%) rest=%.1f (%.0f%%)\\n",
                n_steps, gen_ms, n_steps / std::max(0.001, gen_ms / 1000.0),
                t_draft_sum / 1000.0, t_draft_sum / 1000.0 * inv,
                t_verify_sum / 1000.0, t_verify_sum / 1000.0 * inv,
                t_process_sum / 1000.0, t_process_sum / 1000.0 * inv,
                t_sample_sum / 1000.0, t_sample_sum / 1000.0 * inv,
                t_rest_sum / 1000.0, t_rest_sum / 1000.0 * inv);
    }
""",
    )


def patch_offload_rs(root: Path) -> None:
    """Split KV offload from GDN R/S offload. --no-kv-offload must not pin R/S on CPU."""
    hybrid_h = root / "src" / "llama-memory-hybrid.h"
    hybrid_c = root / "src" / "llama-memory-hybrid.cpp"
    iswa_h = root / "src" / "llama-memory-hybrid-iswa.h"
    iswa_c = root / "src" / "llama-memory-hybrid-iswa.cpp"

    replace(
        hybrid_h,
        """                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
""",
        """                 uint32_t   n_rs_seq,
                     bool   offload_attn,
                     bool   offload_recr,
                     bool   unified,
""",
    )
    replace(
        iswa_h,
        """                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
""",
        """                 uint32_t   n_rs_seq,
                     bool   offload_attn,
                     bool   offload_recr,
                     bool   unified,
""",
    )
    replace(
        hybrid_c,
        """                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache(
        model,
        model.hparams,
        type_k,
        type_v,
        v_trans,
        offload,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        swa_type,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )) {}
""",
        """                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload_attn,
                     bool   offload_recr,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache(
        model,
        model.hparams,
        type_k,
        type_v,
        v_trans,
        offload_attn,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        swa_type,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload_recr,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )) {
    LLAMA_LOG_INFO("%s: attn KV offload = %d, recurrent R/S offload = %d\\n",
            __func__, (int) offload_attn, (int) offload_recr);
}
""",
    )
    replace(
        iswa_c,
        """                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache_iswa(
        model,
        type_k,
        type_v,
        v_trans,
        offload,
        swa_full,
        unified,
        kv_size,
        n_seq_max,
        n_ubatch,
        n_pad,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )) {}
""",
        """                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload_attn,
                     bool   offload_recr,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache_iswa(
        model,
        type_k,
        type_v,
        v_trans,
        offload_attn,
        swa_full,
        unified,
        kv_size,
        n_seq_max,
        n_ubatch,
        n_pad,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload_recr,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )) {
    LLAMA_LOG_INFO("%s: attn KV offload = %d, recurrent R/S offload = %d\\n",
            __func__, (int) offload_attn, (int) offload_recr);
}
""",
    )

    replace(
        root / "src" / "llama-model.cpp",
        """                            /* offload           */ cparams.offload_kqv,
                            /* unified           */ cparams.kv_unified,
""",
        """                            /* offload_attn      */ cparams.offload_kqv,
                            /* offload_recr      */ cparams.offload_rs,
                            /* unified           */ cparams.kv_unified,
""",
        count=-1,
    )

    replace(
        root / "src" / "llama-cparams.h",
        "    bool offload_kqv;\n    bool flash_attn;\n",
        "    bool offload_kqv;\n    bool offload_rs = true; // GDN R/S; independent of KV (--no-kv-offload)\n    bool flash_attn;\n",
    )
    replace(
        root / "include" / "llama.h",
        "        bool offload_kqv; // offload the KQV ops (including the KV cache) to GPU\n        bool no_perf;     // measure performance timings\n",
        "        bool offload_kqv; // offload the KQV ops (including the KV cache) to GPU\n"
        "        bool offload_rs;  // offload Gated DeltaNet R/S independently of KV\n"
        "        bool no_perf;     // measure performance timings\n",
    )
    replace(
        root / "src" / "llama-context.cpp",
        "    cparams.offload_kqv             = params.offload_kqv;\n",
        "    cparams.offload_kqv             = params.offload_kqv;\n"
        "    cparams.offload_rs              = params.offload_rs;\n",
    )
    replace(
        root / "src" / "llama-context.cpp",
        "        /*.offload_kqv                 =*/ true,\n        /*.no_perf                     =*/ true,\n",
        "        /*.offload_kqv                 =*/ true,\n"
        "        /*.offload_rs                  =*/ true,\n"
        "        /*.no_perf                     =*/ true,\n",
    )
    replace(
        root / "common" / "common.h",
        "    bool no_kv_offload     = false; // disable KV offloading\n",
        "    bool no_kv_offload     = false; // disable KV offloading\n"
        "    bool offload_rs        = true;  // GDN R/S on the layer device (independent of KV)\n",
    )
    replace(
        root / "common" / "common.cpp",
        "    cparams.offload_kqv       = !params.no_kv_offload;\n",
        "    cparams.offload_kqv       = !params.no_kv_offload;\n"
        "    cparams.offload_rs        = params.offload_rs;\n",
    )
    once(
        root / "common" / "arg.cpp",
        '    ).set_env("LLAMA_ARG_KV_OFFLOAD"));\n',
        """
    add_opt(common_arg(
        {"--offload-rs"},
        {"--no-offload-rs"},
        "offload Gated DeltaNet recurrent state R/S to the layer device (default: enabled). Independent of --no-kv-offload.",
        [](common_params & params, bool value) {
            params.offload_rs = value;
        }
    ).set_env("LLAMA_ARG_OFFLOAD_RS"));
""",
    )
    replace(
        root / "src" / "llama-memory-recurrent.cpp",
        '        LLAMA_LOG_DEBUG("%s, layer %3d: dev = %s\\n", __func__, i, dev_name);\n',
        '        LLAMA_LOG_INFO("%s, layer %3d: dev = %s\\n", __func__, i, dev_name);\n',
    )



def patch_chunked_gpu_load(root: Path) -> None:
    """Stream GPU tensors from the GGUF in small chunks instead of whole-tensor RAM.

    `--load-mode none` already has an async pinned-staging path. If that path
    is unavailable the upstream fallback allocated one host buffer the size of
    the current tensor (lm_head is ~1 GiB).  Always chunk, and log the chosen
    path at INFO so a 16 GB-VRAM / modest-RAM host can load without swapping.
    """
    path = root / "src" / "llama-model-loader.cpp"
    if "HGA_CHUNKED_GPU_LOAD" in path.read_text(encoding="utf-8"):
        print(f"  already patched: {path.name} chunked GPU load")
        return

    replace(
        path,
        "    // 4 staging buffers for async uploads, each sized 1MB seems to be a good default for single NVMe drives.\n"
        "    // NVMe raid configurations might require more / larger buffers.\n"
        "    constexpr size_t n_buffers = 4;\n",
        "    // HGA_CHUNKED_GPU_LOAD: 4 staging buffers. 16 MiB each is a small RAM\n"
        "    // ceiling while remaining sequential-read friendly on NVMe.\n"
        "    constexpr size_t n_buffers = 4;\n",
    )
    replace(
        path,
        "    // Buffer size: balance between memory usage and I/O efficiency\n"
        "    // 64MB works well for NVMe drives\n"
        "    const size_t buffer_size = alignment != 1 ? 64 * 1024 * 1024 + 2 * alignment : 1 * 1024 * 1024;\n",
        "    // Buffer size: keep host staging tiny. Direct I/O wants 64 MiB;\n"
        "    // regular files use 16 MiB (was 1 MiB).\n"
        "    const size_t buffer_size = alignment != 1 ? 64 * 1024 * 1024 + 2 * alignment : 16 * 1024 * 1024;\n",
    )
    replace(
        path,
        "    if (upload_backend) {\n"
        "        LLAMA_LOG_DEBUG(\"%s: using async uploads for device %s, buffer type %s, backend %s\\n\", __func__,\n"
        "            ggml_backend_dev_name(ggml_backend_get_device(upload_backend)),\n"
        "            ggml_backend_buft_name(ggml_backend_buffer_get_type(bufs.at(0))),\n"
        "            ggml_backend_name(upload_backend));\n"
        "    }\n",
        "    if (upload_backend) {\n"
        "        LLAMA_LOG_INFO(\"%s: HGA chunked GPU load: async pinned staging %.1f MiB x %zu\\n\", __func__,\n"
        "            buffer_size / (1024.0 * 1024.0), n_buffers);\n"
        "        LLAMA_LOG_DEBUG(\"%s: using async uploads for device %s, buffer type %s, backend %s\\n\", __func__,\n"
        "            ggml_backend_dev_name(ggml_backend_get_device(upload_backend)),\n"
        "            ggml_backend_buft_name(ggml_backend_buffer_get_type(bufs.at(0))),\n"
        "            ggml_backend_name(upload_backend));\n"
        "    } else {\n"
        "        LLAMA_LOG_INFO(\"%s: HGA chunked GPU load: sync 16 MiB file staging (async path unavailable)\\n\", __func__);\n"
        "    }\n",
    )
    replace(
        path,
        "                } else {\n"
        "                    read_buf.resize(n_size);\n"
        "                    file->seek(weight->offs, SEEK_SET);\n"
        "                    file->read_raw(read_buf.data(), n_size);\n"
        "                    ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);\n"
        "                    if (check_tensors && !ggml_validate_row_data(cur->type, read_buf.data(), n_size)) {\n"
        "                        throw std::runtime_error(format(\"tensor '%s' has invalid data\", ggml_get_name(cur)));\n"
        "                    }\n"
        "                }\n",
        "                } else {\n"
        "                    /* HGA: never stage a whole GPU tensor in RAM. */\n"
        "                    constexpr size_t hga_chunk = 16ull * 1024ull * 1024ull;\n"
        "                    read_buf.resize(std::min(hga_chunk, n_size ? n_size : hga_chunk));\n"
        "                    file->seek(weight->offs, SEEK_SET);\n"
        "                    size_t copied = 0;\n"
        "                    while (copied < n_size) {\n"
        "                        const size_t n = std::min(read_buf.size(), n_size - copied);\n"
        "                        file->read_raw(read_buf.data(), n);\n"
        "                        ggml_backend_tensor_set(cur, read_buf.data(), copied, n);\n"
        "                        copied += n;\n"
        "                    }\n"
        "                    if (check_tensors) {\n"
        "                        LLAMA_LOG_WARN(\"%s: skipping host validation of GPU tensor '%s' (chunked load)\\n\",\n"
        "                            __func__, ggml_get_name(cur));\n"
        "                    }\n"
        "                }\n",
    )


def patch_server_critical_path_prof(root: Path) -> None:
    """Runtime-gated server timers around draft, target, process, and sampling."""
    path = root / "tools" / "server" / "server-context.cpp"
    text = path.read_text(encoding="utf-8")
    if "hga-server-prof target_decode" in text:
        print("  already patched: server critical-path profiler")
        return

    replace(
        path,
        "    int64_t n_post_decode = 0;\n"
        "    int64_t n_sampl       = 0;\n",
        "    int64_t n_post_decode = 0;\n"
        "    int64_t n_sampl       = 0;\n"
        "\n"
        "    static bool hga_profile_server() {\n"
        "        static const bool enabled = []() {\n"
        "            const char * env = std::getenv(\"HGA_PROFILE_SERVER\");\n"
        "            return env && env[0] && env[0] != '0';\n"
        "        }();\n"
        "        return enabled;\n"
        "    }\n",
    )
    replace(
        path,
        "        try {\n"
        "            scoped_timer t(t_pre_decode, n_pre_decode);\n"
        "            pre_decode();\n"
        "            batch.render();\n",
        "        try {\n"
        "            const int64_t hga_prof_t0 = hga_profile_server() ? ggml_time_us() : 0;\n"
        "            scoped_timer t(t_pre_decode, n_pre_decode);\n"
        "            pre_decode();\n"
        "            batch.render();\n"
        "            if (hga_prof_t0) {\n"
        "                SRV_INF(\"hga-server-prof pre_decode+render %.3f ms batch=%d\\n\",\n"
        "                        (ggml_time_us() - hga_prof_t0) / 1000.0, batch.size());\n"
        "            }\n",
    )
    replace(
        path,
        "            try {\n"
        "                scoped_timer t(t_post_decode, n_post_decode);\n"
        "                post_decode(n_tokens, off, batch_view);\n",
        "            try {\n"
        "                const int64_t hga_prof_t0 = hga_profile_server() ? ggml_time_us() : 0;\n"
        "                scoped_timer t(t_post_decode, n_post_decode);\n"
        "                post_decode(n_tokens, off, batch_view);\n"
        "                if (hga_prof_t0) {\n"
        "                    SRV_INF(\"hga-server-prof post_decode %.3f ms batch=%d\\n\",\n"
        "                            (ggml_time_us() - hga_prof_t0) / 1000.0, n_tokens);\n"
        "                }\n",
    )
    replace(
        path,
        "        if (!drafting.empty()) {\n"
        "            queue_tasks.yield_to_queue([&]() {\n"
        "                common_speculative_draft(spec.get());\n"
        "            });\n"
        "        }\n",
        "        if (!drafting.empty()) {\n"
        "            const int64_t hga_prof_t0 = hga_profile_server() ? ggml_time_us() : 0;\n"
        "            queue_tasks.yield_to_queue([&]() {\n"
        "                common_speculative_draft(spec.get());\n"
        "            });\n"
        "            if (hga_prof_t0) {\n"
        "                SRV_INF(\"hga-server-prof speculative_draft %.3f ms slots=%zu\\n\",\n"
        "                        (ggml_time_us() - hga_prof_t0) / 1000.0, drafting.size());\n"
        "            }\n"
        "        }\n",
    )
    replace(
        path,
        "        int ret = 0;\n"
        "        queue_tasks.yield_to_queue([&]() {\n"
        "            ret = llama_decode(ctx_tgt, batch_view);\n"
        "            if (ret == 0 && has_output) {\n"
        "                llama_synchronize(ctx_tgt);\n"
        "            }\n"
        "        });\n",
        "        int ret = 0;\n"
        "        const int64_t hga_prof_target_t0 = hga_profile_server() ? ggml_time_us() : 0;\n"
        "        queue_tasks.yield_to_queue([&]() {\n"
        "            ret = llama_decode(ctx_tgt, batch_view);\n"
        "            if (ret == 0 && has_output) {\n"
        "                llama_synchronize(ctx_tgt);\n"
        "            }\n"
        "        });\n"
        "        if (hga_prof_target_t0) {\n"
        "            SRV_INF(\"hga-server-prof target_decode %.3f ms batch=%d output=%d\\n\",\n"
        "                    (ggml_time_us() - hga_prof_target_t0) / 1000.0,\n"
        "                    batch_view.n_tokens, (int) has_output);\n"
        "        }\n",
    )
    replace(
        path,
        "        if (spec) {\n"
        "            bool ok = true;\n"
        "            queue_tasks.yield_to_queue([&]() {\n"
        "                ok = common_speculative_process(spec.get(), batch_view);\n"
        "            });\n",
        "        if (spec) {\n"
        "            bool ok = true;\n"
        "            const int64_t hga_prof_t0 = hga_profile_server() ? ggml_time_us() : 0;\n"
        "            queue_tasks.yield_to_queue([&]() {\n"
        "                ok = common_speculative_process(spec.get(), batch_view);\n"
        "            });\n"
        "            if (hga_prof_t0) {\n"
        "                SRV_INF(\"hga-server-prof speculative_process %.3f ms batch=%d\\n\",\n"
        "                        (ggml_time_us() - hga_prof_t0) / 1000.0, batch_view.n_tokens);\n"
        "            }\n",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("llama_cpp", type=Path, help="path to llama.cpp checkout")
    args = ap.parse_args()
    root: Path = args.llama_cpp.resolve()
    if not (root / "src" / "models" / "qwen35.cpp").is_file():
        die(f"{root} does not look like llama.cpp (missing src/models/qwen35.cpp)")

    # Release pin: HGA is validated against llama.cpp v0.3.0.  Git-backed
    # checkouts are matched by commit (v0.3.0 == b10621).  Copies with no .git
    # (offline host) are accepted as-is.
    if (root / ".git").exists():
        assert_llama_release_pin(root)
    elif not (root / "src" / "CMakeLists.txt").is_file():
        die(f"{root} does not look like a llama.cpp clone (missing src/CMakeLists.txt)")

    src_hga_h = HERE / "cpp" / "include" / "hga.h"
    src_hga_c = HERE / "cpp" / "src" / "hga.cpp"
    src_l2_h = HERE / "cpp" / "include" / "hga_l2.h"
    src_l2_c = HERE / "cpp" / "src" / "hga_l2.cpp"
    glue_h = HERE / "llama.cpp-hga" / "llama-hga.h"
    glue_c = HERE / "llama.cpp-hga" / "llama-hga.cpp"
    swap_h = HERE / "llama.cpp-hga" / "hga-weight-swap.h"
    swap_c = HERE / "llama.cpp-hga" / "hga-weight-swap.cpp"
    kv_h = HERE / "llama.cpp-hga" / "hga-kv-gemv.h"
    kv_c = HERE / "llama.cpp-hga" / "hga-kv-gemv.cpp"
    for p in (src_hga_h, src_hga_c, src_l2_h, src_l2_c, glue_h, glue_c, swap_h, swap_c, kv_h, kv_c):
        if not p.is_file():
            die(f"missing {p}")

    patch_muse_glimmer_tool_termination(root)

    # Do not preserve the checkout machine's mtimes. turing1's clock can be a
    # few minutes ahead, causing Ninja to treat a freshly deployed HGA source
    # as older than its object file and silently keep a stale binary.
    shutil.copyfile(src_hga_h, root / "src" / "hga.h")
    shutil.copyfile(src_hga_c, root / "src" / "hga.cpp")
    shutil.copyfile(src_l2_h, root / "src" / "hga_l2.h")
    shutil.copyfile(src_l2_c, root / "src" / "hga_l2.cpp")
    shutil.copyfile(glue_h, root / "src" / "llama-hga.h")
    shutil.copyfile(glue_c, root / "src" / "llama-hga.cpp")
    shutil.copyfile(swap_h, root / "src" / "hga-weight-swap.h")
    shutil.copyfile(swap_c, root / "src" / "hga-weight-swap.cpp")
    shutil.copyfile(kv_h, root / "src" / "hga-kv-gemv.h")
    shutil.copyfile(kv_c, root / "src" / "hga-kv-gemv.cpp")
    print("copied hga.{h,cpp} hga_l2.{h,cpp} llama-hga.{h,cpp} hga-weight-swap.{h,cpp} hga-kv-gemv.{h,cpp}")

    # Repair an older patch that closed llama_context_default_params too early.
    ctx_cpp = root / "src" / "llama-context.cpp"
    broken = (
        "        /*.ctx_other                   =*/ nullptr,\n"
        "    };\n"
        "        /*.hga_enabled                 =*/ false,\n"
    )
    fixed = (
        "        /*.ctx_other                   =*/ nullptr,\n"
        "        /*.hga_enabled                 =*/ false,\n"
    )
    t = ctx_cpp.read_text(encoding="utf-8")
    if broken in t:
        ctx_cpp.write_text(t.replace(broken, fixed, 1), encoding="utf-8")
        print("  repaired llama_context_default_params initializer")

    arg_cpp = root / "common" / "arg.cpp"
    at = arg_cpp.read_text(encoding="utf-8")
    old_f = "[](common_params & params, float value) { params.hga_frac_l1 = value; }"
    new_f = "[](common_params & params, const std::string & value) { params.hga_frac_l1 = std::stof(value); }"
    if old_f in at:
        at = at.replace(old_f, new_f, 1)
        at = at.replace(
            "[](common_params & params, float value) { params.hga_frac_l2 = value; }",
            "[](common_params & params, const std::string & value) { params.hga_frac_l2 = std::stof(value); }",
            1,
        )
        arg_cpp.write_text(at, encoding="utf-8")
        print("  repaired --hga-frac-* parsers (string, not float)")

    once(
        root / "src" / "CMakeLists.txt",
        "            llama-graph.cpp\n",
        "            llama-hga.cpp\n            hga.cpp\n",
    )
    once(
        root / "src" / "CMakeLists.txt",
        "            llama-hga.cpp\n            hga.cpp\n",
        "            hga-weight-swap.cpp\n",
    )
    once(
        root / "src" / "CMakeLists.txt",
        "            hga-weight-swap.cpp\n",
        "            hga_l2.cpp\n            hga-kv-gemv.cpp\n",
    )
    once(
        root / "src" / "CMakeLists.txt",
        "target_link_libraries(llama PUBLIC ggml)\n",
        """
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(llama PRIVATE OpenMP::OpenMP_CXX)
endif()
""",
    )
    once(
        root / "src" / "CMakeLists.txt",
        """if (OpenMP_CXX_FOUND)
    target_link_libraries(llama PRIVATE OpenMP::OpenMP_CXX)
endif()
""",
        """
# The HGA router is copied into the llama target rather than ggml-cpu, so it
# does not inherit ggml-cpu's per-ISA flags.  Keep the flags source-local: the
# rest of llama remains portable.  HGA's kernels compile _mm512_* unconditionally
# once __AVX512F__ is set, so only enable AVX-512 when the build host actually has
# it - otherwise llama.cpp dies with SIGILL on a CPU without AVX-512.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(HGA_CPU_COMPILE_OPTIONS
        -O3 -fno-strict-aliasing -mavx2 -mfma -mf16c)
    if (EXISTS "/proc/cpuinfo")
        file(READ "/proc/cpuinfo" _hga_cpuinfo)
        string(FIND "${_hga_cpuinfo}" " avx512f" _hga_has512)
        if (_hga_has512 GREATER -1)
            set(HGA_CPU_COMPILE_OPTIONS
                -O3 -fno-strict-aliasing -mavx2 -mfma -mf16c
                -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512cd)
        endif()
    endif()
    set_source_files_properties(hga.cpp hga_l2.cpp hga-kv-gemv.cpp
        PROPERTIES COMPILE_OPTIONS "${HGA_CPU_COMPILE_OPTIONS}")
endif()
""",
        marker="set(HGA_CPU_COMPILE_OPTIONS",
    )

    once(
        root / "include" / "llama.h",
        "        struct llama_context * ctx_other;\n",
        """        // Hierarchical Global Attention (CPU exact-token routing on full-attn layers)
        bool    hga_enabled;
        int32_t hga_levels;
        int32_t hga_chunk_size;
        int32_t hga_group_size;
        int32_t hga_keep_first;
        int32_t hga_keep_last;
        float   hga_frac_l1;
        float   hga_frac_l2;
""",
        marker="        bool    hga_enabled;",
    )

    once(
        root / "src" / "llama-context.cpp",
        "        /*.ctx_other                   =*/ nullptr,\n",
        """        /*.hga_enabled                 =*/ false,
        /*.hga_levels                  =*/ 2,
        /*.hga_chunk_size              =*/ 64,
        /*.hga_group_size              =*/ 16,
        /*.hga_keep_first              =*/ 2,
        /*.hga_keep_last               =*/ 7,
        /*.hga_frac_l1                 =*/ 0.08f,
        /*.hga_frac_l2                 =*/ 0.04f,
""",
        marker="        /*.hga_enabled                 =*/ false,",
    )

    cparams_h = root / "src" / "llama-cparams.h"
    if "bool hga_enabled" in cparams_h.read_text(encoding="utf-8"):
        print("  already patched: llama-cparams.h")
    else:
        once(
            cparams_h,
            "    llama_context * ctx_other;\n",
            """    bool hga_enabled = false;
    int32_t hga_levels = 2;
    int32_t hga_chunk_size = 64;
    int32_t hga_group_size = 16;
    int32_t hga_keep_first = 2;
    int32_t hga_keep_last = 7;
    float hga_frac_l1 = 0.08f;
    float hga_frac_l2 = 0.04f;
    bool hga_i8 = true;
    void * hga_runtime = nullptr;
""",
        )

    once(
        root / "src" / "llama-context.cpp",
        '#include "llama.h"\n',
        '#include "llama-hga.h"\n',
    )

    once(
        root / "src" / "llama-context.cpp",
        "    cparams.offload_kqv             = params.offload_kqv;\n",
        "    hga_cparams_from_ctx_params(cparams, params);\n",
    )

    once(
        root / "src" / "llama-context.cpp",
        "    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;\n",
        "    hga_runtime_init(cparams, hparams);\n",
    )

    once(
        root / "src" / "llama-context.cpp",
        "llama_context::~llama_context() {\n    // wait for any pending asynchronous copies into the output buffers before they are freed\n    synchronize();\n",
        "    hga_runtime_free(cparams);\n",
    )
    once(
        root / "src" / "llama-context.cpp",
        "llama_context::~llama_context() {\n    // wait for any pending asynchronous copies into the output buffers before they are freed\n    synchronize();\n",
        "    hga_swap_free();\n",
    )

    ctx_h = root / "src" / "llama-context.h"
    ht = ctx_h.read_text(encoding="utf-8")
    if "void hga_vram_log" in ht:
        print("  already patched: llama-context.h hga_vram_log")
    elif "void hga_swap_free();" in ht:
        ctx_h.write_text(
            ht.replace(
                "    void hga_swap_free();\n",
                "    void hga_swap_free();\n    void hga_vram_log(const char * tag, uint32_t n_tokens = 0);\n",
                1,
            ),
            encoding="utf-8",
        )
        print("  patched llama-context.h: hga_vram_log")
    else:
        once(
            ctx_h,
            "    void sched_reserve();\n\n    void synchronize();\n",
            """
    // HGA two-mode QKV / lm_head staging (see planned_refactoring.md)
    void hga_swap_init();
    void hga_swap_ensure(uint32_t n_tokens);
    void hga_swap_free();
    void hga_vram_log(const char * tag, uint32_t n_tokens = 0);
""",
        )

    replace(
        root / "src" / "llama-context.cpp",
        """        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\\n", __func__);
        }

        sched_reserve();
""",
        """        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\\n", __func__);
        }

        hga_swap_init();
        sched_reserve();
""",
    )
    replace(
        root / "src" / "llama-context.cpp",
        "        hga_swap_init();\n        sched_reserve();\n",
        "        hga_swap_init();\n        sched_reserve();\n        hga_vram_log(\"ctor after sched_reserve\", 0);\n",
    )

    replace(
        root / "src" / "llama-context.cpp",
        "    output_swaps.clear();\n\n    sched_reserve();\n",
        "    output_swaps.clear();\n\n    hga_swap_ensure(n_tokens_all);\n    sched_reserve();\n",
    )

    replace(
        root / "src" / "llama-context.cpp",
        """    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    sched_reserve();

    n_queued_tokens += n_tokens;
""",
        """    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    hga_swap_ensure(n_tokens);
    sched_reserve();

    n_queued_tokens += n_tokens;
""",
    )

    replace(
        root / "src" / "llama-context.cpp",
        """        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
    }
""",
        """        hga_vram_log("ubatch before alloc_graph", ubatch.n_tokens);
        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            hga_vram_log("ubatch alloc_graph FAILED", ubatch.n_tokens);
            LLAMA_LOG_ERROR("%s: failed to allocate graph\\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
        hga_vram_log("ubatch after alloc_graph", ubatch.n_tokens);
    }
""",
    )
    ctx_gc = root / "src" / "llama-context.cpp"
    ctx_gc_text = ctx_gc.read_text(encoding="utf-8")
    if "hga-prof graph PREFILL" in ctx_gc_text:
        print("  already patched: llama-context.cpp graph wall profile")
    else:
        graph_compute_profile = '''    const char * hga_profile_graph_env = std::getenv("HGA_PROFILE_GRAPH");
    const bool hga_profile_graph = cparams.hga_enabled && hga_profile_graph_env &&
            hga_profile_graph_env[0] && hga_profile_graph_env[0] != '0';
    const int64_t hga_graph_t0 = hga_profile_graph ? ggml_time_us() : 0;
    hga_vram_log("ubatch before graph_compute", ubatch.n_tokens);
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    hga_vram_log("ubatch after graph_compute", ubatch.n_tokens);
    if (hga_profile_graph) {
        ggml_backend_sched_synchronize(sched.get());
        const double hga_graph_ms = (ggml_time_us() - hga_graph_t0) / 1000.0;
        static double hga_graph_prefill_ms = 0.0;
        static double hga_graph_decode_ms = 0.0;
        static int hga_graph_prefill_n = 0;
        static int hga_graph_decode_n = 0;
        static uint32_t hga_graph_prefill_tok = 0;
        const bool hga_graph_prefill = ubatch.n_tokens >= 8;
        if (hga_graph_prefill) {
            hga_graph_prefill_ms += hga_graph_ms;
            hga_graph_prefill_n += 1;
            hga_graph_prefill_tok += ubatch.n_tokens;
            std::fprintf(stderr, "hga-prof graph PREFILL #%d n=%u wall=%.2f ms  sum=%.1f ms\\n",
                    hga_graph_prefill_n, ubatch.n_tokens, hga_graph_ms, hga_graph_prefill_ms);
        } else {
            hga_graph_decode_ms += hga_graph_ms;
            hga_graph_decode_n += 1;
            std::fprintf(stderr, "hga-prof graph DECODE #%d n=%u wall=%.2f ms  sum=%.1f ms\\n",
                    hga_graph_decode_n, ubatch.n_tokens, hga_graph_ms, hga_graph_decode_ms);
            if (hga_graph_prefill_n > 0) {
                static bool hga_graph_prefill_dumped = false;
                if (!hga_graph_prefill_dumped) {
                    hga_graph_prefill_dumped = true;
                    const double tps = hga_graph_prefill_ms > 0.0
                            ? 1000.0 * hga_graph_prefill_tok / hga_graph_prefill_ms : 0.0;
                    std::fprintf(stderr, "hga-prof graph TOTAL prefill graphs=%d tokens=%u wall=%.1f ms (%.1f tok/s graph-only)\\n",
                            hga_graph_prefill_n, hga_graph_prefill_tok, hga_graph_prefill_ms, tps);
                }
            }
        }
    }
'''
        # Older HGA worktrees already have graph timing or vram logs here,
        # while a clean current upstream checkout has only graph_compute().
        profiled_call = """    const bool hga_profile_graph = cparams.hga_enabled && std::getenv("HGA_PROFILE_GRAPH");
    const int64_t hga_graph_t0 = hga_profile_graph ? ggml_time_us() : 0;
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
"""
        vram_call = """    hga_vram_log("ubatch before graph_compute", ubatch.n_tokens);
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    hga_vram_log("ubatch after graph_compute", ubatch.n_tokens);
"""
        if profiled_call in ctx_gc_text:
            replace(ctx_gc, profiled_call, graph_compute_profile)
        elif vram_call in ctx_gc_text:
            replace(ctx_gc, vram_call, graph_compute_profile)
        else:
            replace(
                ctx_gc,
                "    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);\n",
                graph_compute_profile,
            )

    once(
        root / "common" / "common.h",
        "    enum llama_flash_attn_type   flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_AUTO; // whether to use Flash Attention\n",
        """
    bool    hga_enabled    = false;
    int32_t hga_levels     = 2;
    int32_t hga_chunk_size = 64;
    int32_t hga_group_size = 16;
    int32_t hga_keep_first = 2;
    int32_t hga_keep_last  = 7;
    float   hga_frac_l1    = 0.08f;
    float   hga_frac_l2    = 0.04f;
""",
        marker="    bool    hga_enabled    = false;",
    )

    once(
        root / "common" / "common.cpp",
        "    cparams.kv_unified        = params.kv_unified;\n",
        """
    cparams.hga_enabled    = params.hga_enabled;
    cparams.hga_levels     = params.hga_levels;
    cparams.hga_chunk_size = params.hga_chunk_size;
    cparams.hga_group_size = params.hga_group_size;
    cparams.hga_keep_first = params.hga_keep_first;
    cparams.hga_keep_last  = params.hga_keep_last;
    cparams.hga_frac_l1    = params.hga_frac_l1;
    cparams.hga_frac_l2    = params.hga_frac_l2;
""",
    )

    once(
        root / "common" / "arg.cpp",
        '    ).set_env("LLAMA_ARG_KV_OFFLOAD"));\n',
        """
    add_opt(common_arg(
        {"--hga"},
        {"--no-hga"},
        "Hierarchical Global Attention on Qwen3.5/3.8 full-attention layers (CPU exact-token routing). Default is HGA-2 (--hga-levels 2).",
        [](common_params & params, bool value) {
            params.hga_enabled = value;
        }
    ).set_env("LLAMA_ARG_HGA"));
    add_opt(common_arg(
        {"--hga-levels"}, "N",
        "HGA hierarchy depth (default 2): 1 = whole 64-token chunks (~8% tokens), 2 = groups of 16 (~4% tokens, HGA-2). Same chunk count.",
        [](common_params & params, int value) {
            params.hga_levels = value;
        }
    ).set_env("LLAMA_ARG_HGA_LEVELS"));
    add_opt(common_arg(
        {"--hga-chunk"}, "N",
        string_format("HGA chunk size (default: %d)", params.hga_chunk_size),
        [](common_params & params, int value) { params.hga_chunk_size = value; }
    ));
    add_opt(common_arg(
        {"--hga-group"}, "N",
        string_format("HGA group size (default: %d)", params.hga_group_size),
        [](common_params & params, int value) { params.hga_group_size = value; }
    ));
    add_opt(common_arg(
        {"--hga-keep-first"}, "N",
        string_format("HGA sink chunks (default: %d)", params.hga_keep_first),
        [](common_params & params, int value) { params.hga_keep_first = value; }
    ));
    add_opt(common_arg(
        {"--hga-keep-last"}, "N",
        string_format("HGA local chunks (default: %d)", params.hga_keep_last),
        [](common_params & params, int value) { params.hga_keep_last = value; }
    ));
    add_opt(common_arg(
        {"--hga-frac-l1"}, "F",
        string_format("1-level extra mid-chunk fraction, exclusive of windows (default: %.2f)", (double) params.hga_frac_l1),
        [](common_params & params, const std::string & value) { params.hga_frac_l1 = std::stof(value); }
    ));
    add_opt(common_arg(
        {"--hga-frac-l2"}, "F",
        string_format("2-level extra mid-group fraction, exclusive of windows (default: %.2f)", (double) params.hga_frac_l2),
        [](common_params & params, const std::string & value) { params.hga_frac_l2 = std::stof(value); }
    ));
""",
        marker='        {"--hga"},',
    )

    # Default routing is HGA-2. Re-runnable on trees that still default to 1.
    maybe_replace(
        root / "src" / "llama-context.cpp",
        "        /*.hga_levels                  =*/ 1,\n",
        "        /*.hga_levels                  =*/ 2,\n",
    )
    maybe_replace(
        root / "src" / "llama-cparams.h",
        "    int32_t hga_levels = 1;\n",
        "    int32_t hga_levels = 2;\n",
    )
    maybe_replace(
        root / "common" / "common.h",
        "    int32_t hga_levels     = 1;\n",
        "    int32_t hga_levels     = 2;\n",
    )
    maybe_replace(
        root / "common" / "arg.cpp",
        "HGA hierarchy depth: 1 = whole 64-token chunks (~8% tokens), 2 = groups of 16 (~4% tokens). Same chunk count.",
        "HGA hierarchy depth (default 2): 1 = whole 64-token chunks (~8% tokens), 2 = groups of 16 (~4% tokens, HGA-2). Same chunk count.",
    )
    maybe_replace(
        root / "common" / "arg.cpp",
        "Hierarchical Global Attention on Qwen3.5/3.8 full-attention layers (CPU exact-token routing)",
        "Hierarchical Global Attention on Qwen3.5/3.8 full-attention layers (CPU exact-token routing). Default is HGA-2 (--hga-levels 2).",
    )

    # INT8 KV + QK dots (re-runnable on an already-patched tree).
    once(root / "include" / "llama.h", "        float   hga_frac_l2;\n", "        bool    hga_i8;\n")
    once(
        root / "src" / "llama-context.cpp",
        "        /*.hga_frac_l2                 =*/ 0.04f,\n",
        "        /*.hga_i8                      =*/ true,\n",
    )
    once(root / "src" / "llama-cparams.h", "    float hga_frac_l2 = 0.04f;\n", "    bool hga_i8 = true;\n")
    once(
        root / "src" / "llama-cparams.h",
        "    void * hga_runtime = nullptr;\n",
        """    int32_t hga_phase = 0; // 0 none, 1 prefill (QKV CUDA), 2 decode (Q+lm_head CUDA)
    void * hga_swap = nullptr;
    uint32_t hga_n_ubatch_orig = 0;
    bool hga_seen_prefill = false;
""",
    )
    once(
        root / "src" / "llama-cparams.h",
        "    bool hga_seen_prefill = false;\n",
        "    bool hga_seen_large_prefill = false;\n",
    )
    once(root / "common" / "common.h", "    float   hga_frac_l2    = 0.04f;\n", "    bool    hga_i8         = true;\n")
    once(
        root / "common" / "common.cpp",
        "    cparams.hga_frac_l2    = params.hga_frac_l2;\n",
        "    cparams.hga_i8         = params.hga_i8;\n",
    )
    once(
        root / "common" / "arg.cpp",
        """    add_opt(common_arg(
        {"--hga-frac-l2"}, "F",
        string_format("2-level extra mid-group fraction, exclusive of windows (default: %.2f)", (double) params.hga_frac_l2),
        [](common_params & params, const std::string & value) { params.hga_frac_l2 = std::stof(value); }
    ));
""",
        """    add_opt(common_arg(
        {"--hga-i8"},
        {"--hga-f16"},
        "HGA KV + QK dots: INT8 (default, AVX-512 integer MAC) or F16 reference",
        [](common_params & params, bool value) { params.hga_i8 = value; }
    ));
""",
    )

    once(root / "include" / "llama.h", "        bool    hga_i8;\n",
         "        bool    hga_wave;\n        float   hga_frac_retr;\n        float   hga_frac_est;\n")
    once(
        root / "src" / "llama-context.cpp",
        "        /*.hga_i8                      =*/ true,\n",
        "        /*.hga_wave                    =*/ false,\n        /*.hga_frac_retr               =*/ 0.018f,\n        /*.hga_frac_est                =*/ 0.232f,\n",
    )
    once(root / "src" / "llama-cparams.h", "    bool hga_i8 = true;\n",
         "    bool hga_wave = false;\n    float hga_frac_retr = 0.018f;\n    float hga_frac_est = 0.232f;\n")
    once(root / "common" / "common.h", "    bool    hga_i8         = true;\n",
         "    bool    hga_wave       = false;\n    float   hga_frac_retr  = 0.018f;\n    float   hga_frac_est   = 0.232f;\n")
    once(
        root / "common" / "common.cpp",
        "    cparams.hga_i8         = params.hga_i8;\n",
        "    cparams.hga_wave       = params.hga_wave;\n    cparams.hga_frac_retr  = params.hga_frac_retr;\n    cparams.hga_frac_est   = params.hga_frac_est;\n",
    )
    once(
        root / "common" / "arg.cpp",
        """    add_opt(common_arg(
        {"--hga-i8"},
        {"--hga-f16"},
        "HGA KV + QK dots: INT8 (default, AVX-512 integer MAC) or F16 reference",
        [](common_params & params, bool value) { params.hga_i8 = value; }
    ));
""",
        """    add_opt(common_arg(
        {"--hga-wave"},
        {"--hga-hier"},
        "Decode KV routing: RetroInfer-style wave index (segmented k-means + estimation) or hierarchical summaries (default)",
        [](common_params & params, bool value) { params.hga_wave = value; }
    ));
    add_opt(common_arg(
        {"--hga-frac-retr"}, "F",
        string_format("Wave retrieval budget as a fraction of clusters (default: %.3f)", (double) params.hga_frac_retr),
        [](common_params & params, const std::string & value) { params.hga_frac_retr = std::stof(value); }
    ));
    add_opt(common_arg(
        {"--hga-frac-est"}, "F",
        string_format("Wave estimation budget as a fraction of clusters (default: %.3f)", (double) params.hga_frac_est),
        [](common_params & params, const std::string & value) { params.hga_frac_est = std::stof(value); }
    ));
""",
    )

    qwen = root / "src" / "models" / "qwen35.cpp"
    once(qwen, '#include "models.h"\n', '#include "llama-hga.h"\n')
    qwen_txt = qwen.read_text(encoding="utf-8")
    # The diagnostic probe is added after all normal qwen35 rewrites. Remove
    # it first so exact anchors below remain stable when apply_hga.py is rerun
    # on an already-patched checkout.
    qwen_without_probe = re.sub(
        r'^\s*hga_pin_gpu_prefill_probe\(sched, \w+, cparams\.hga_phase\);\n',
        '',
        qwen_txt,
        flags=re.MULTILINE,
    )
    if qwen_without_probe != qwen_txt:
        qwen.write_text(qwen_without_probe, encoding="utf-8")
        qwen_txt = qwen_without_probe
        print("  temporarily removed qwen35.cpp prefill pin probe")

    # Older installer revisions ran the generic MTP pack insertion before the
    # specialized Q/K placement rewrite.  Each rerun then matched the freshly
    # inserted simple pin and expanded it again.  Repair such checkouts once,
    # before applying the idempotent form below.
    if qwen_txt.count('cb(Qcur, "mtp_Qcur_reshaped", il);') > 1:
        q_block = '''    cb(Qcur, "mtp_Qcur_reshaped", il);
    /* Same as trunk VERIFY: n_tokens>1 cannot D2H the strided Q+gate view. */
    if (cparams.hga_runtime && n_tokens > 1 && hga_decode_pack(cparams.hga_phase)) {
        Qcur = ggml_cont(ctx0, Qcur);
        hga_pin_gpu(sched, Qcur);
        cb(Qcur, "mtp_Qcur_cont", il);
    }
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);
    /* Prefill catch-up: q-norm CUDA. Draft/verify: CPU RMSNorm like the trunk. */
    if (cparams.hga_phase == HGA_SWAP_PREFILL) {
        hga_pin_gpu(sched, Qcur);
    } else if (hga_decode_pack(cparams.hga_phase) && backend_cpu) {
        ggml_backend_sched_set_tensor_backend(sched, Qcur, backend_cpu);
    } else {
        hga_pin_gpu_pack(sched, Qcur, cparams.hga_phase);
    }
'''
        qwen_txt, n_q = re.subn(
            r'    cb\(Qcur, "mtp_Qcur_reshaped", il\);\n.*?(?=    ggml_tensor \* gate =)',
            q_block + "\n",
            qwen_txt,
            count=1,
            flags=re.DOTALL,
        )
        if n_q != 1:
            die("qwen35.cpp: could not repair repeated MTP Q placement blocks")

        k_block = '''    cb(Kcur, "mtp_Kcur_normed", il);
    if (cparams.hga_phase == HGA_SWAP_PREFILL) {
        hga_pin_gpu(sched, Kcur);
    } else if (hga_decode_pack(cparams.hga_phase) && backend_cpu) {
        ggml_backend_sched_set_tensor_backend(sched, Kcur, backend_cpu);
    } else {
        hga_pin_gpu_pack(sched, Kcur, cparams.hga_phase);
    }
'''
        qwen_txt, n_k = re.subn(
            r'    cb\(Kcur, "mtp_Kcur_normed", il\);\n.*?(?=    ggml_tensor \* Vcur =)',
            k_block + "\n",
            qwen_txt,
            count=1,
            flags=re.DOTALL,
        )
        if n_k != 1:
            die("qwen35.cpp: could not repair repeated MTP K placement blocks")
        qwen.write_text(qwen_txt, encoding="utf-8")
        print("  repaired qwen35.cpp repeated MTP Q/K placement blocks")
    # Decode K/V linears stay on CUDA. Do not re-insert host GEMV.
    gemv_block = """    ggml_tensor * Kcur = nullptr;
    ggml_tensor * Vcur = nullptr;
    if (cparams.hga_phase == HGA_SWAP_DECODE && cparams.hga_enabled) {
        if (!hga_build_kv_gemv_pair(this, model.layers[il].wk, model.layers[il].wv, cur, &Kcur, &Vcur, il)) {
            Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
            Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
        }
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
    } else {
        Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
        cb(Kcur, "Kcur", il);
        Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
        cb(Vcur, "Vcur", il);
    }"""
    gemv_block_no_il = gemv_block.replace(
        "hga_build_kv_gemv_pair(this, model.layers[il].wk, model.layers[il].wv, cur, &Kcur, &Vcur, il)",
        "hga_build_kv_gemv_pair(this, model.layers[il].wk, model.layers[il].wv, cur, &Kcur, &Vcur)",
    )
    gemv_block_turing1 = """    ggml_tensor * Kcur = nullptr;
    ggml_tensor * Vcur = nullptr;
    if (cparams.hga_enabled && cparams.hga_phase == HGA_SWAP_DECODE) {
        if (!hga_build_kv_gemv_pair(this, model.layers[il].wk, model.layers[il].wv, cur, &Kcur, &Vcur, il)) {
            Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
            Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
        }
        if (!Kcur) {
            Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
        }
    } else {
        Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
        Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    }
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);"""
    gpu_kv_mm = """    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);
    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);"""
    # Upstream added a readability blank line between the K and V projections.
    # Keep accepting the original compact form for existing patched trees.
    gpu_kv_mm_spaced = gpu_kv_mm.replace(
        '    cb(Kcur, "Kcur", il);\n',
        '    cb(Kcur, "Kcur", il);\n\n',
        1,
    )
    dropped_gemv = False
    for variant, tag in (
        (gemv_block, "GEMV"),
        (gemv_block_no_il, "GEMV no-il"),
        (gemv_block_turing1, "GEMV turing1"),
    ):
        if variant in qwen_txt:
            qwen.write_text(qwen_txt.replace(variant, gpu_kv_mm, 1), encoding="utf-8")
            print(f"  upgraded qwen35.cpp: drop host K/V {tag}, GPU mul_mat")
            qwen_txt = qwen.read_text(encoding="utf-8")
            dropped_gemv = True
            break
    if not dropped_gemv and "hga_build_kv_gemv_pair" in qwen_txt:
        qwen_txt2, n = re.subn(
            r"    ggml_tensor \* Kcur = nullptr;\n"
            r"    ggml_tensor \* Vcur = nullptr;\n"
            r"    if \(cparams\..*?HGA_SWAP_DECODE.*?"
            r"    cb\(Kcur, \"Kcur\", il\);\n"
            r"    cb\(Vcur, \"Vcur\", il\);",
            gpu_kv_mm,
            qwen_txt,
            count=1,
            flags=re.S,
        )
        if n != 1:
            die("qwen35.cpp host K/V GEMV present but no replacement matched")
        qwen.write_text(qwen_txt2, encoding="utf-8")
        print("  upgraded qwen35.cpp: drop host K/V GEMV (regex), GPU mul_mat")
        qwen_txt = qwen_txt2
    if "hga_build_kv_gemv_pair" in qwen_txt or "hga_build_kv_proj" in qwen_txt:
        die("qwen35.cpp still references host K/V GEMV after upgrade")

    # build_norm returns MUL(RMS_NORM(Q), weight).  Retain the RMS root so it
    # can join the independent K/V CUDA roots before the CPU HGA chain.
    qnorm_line = (
        "    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, "
        "nullptr, LLM_NORM_RMS, il);"
    )
    qnorm_root = """    /* build_norm returns MUL(RMS_NORM(Q), weight).  Keep the GPU RMS root so
     * decode/verify can expand it together with K/V before the CPU branch. */
    ggml_tensor * Qcur_rms = Qcur->src[0];
    GGML_ASSERT(Qcur_rms != nullptr);"""
    qwen_txt = qwen.read_text(encoding="utf-8")
    if "ggml_tensor * Qcur_rms = Qcur->src[0];" not in qwen_txt:
        if qnorm_line not in qwen_txt:
            die("qwen35.cpp: Q norm anchor missing for grouped expansion")
        qwen_txt = qwen_txt.replace(qnorm_line, qnorm_line + "\n" + qnorm_root, 1)
        qwen.write_text(qwen_txt, encoding="utf-8")
        print("  patched qwen35.cpp: retained Q RMS expansion root")

    # HGA's q/k weight multiply, RoPE, and attention are CPU in DECODE/VERIFY.
    # Expanding the raw Q projection is insufficient because its GPU RMS node
    # remains under the final HGA root, yielding CPU(V)->CUDA(Q RMS)->CPU(HGA).
    # Expand Q RMS and K/V first so each dense layer has one CPU HGA interval.
    qkv_expand_marker = "Expanding only Qcur_full leaves Qcur_rms"
    qkv_expand_legacy = """    /* Q/K/V are independent CUDA projections of the same normalized input.
     * Expand all three before the CPU q/k-norm + RoPE + HGA chain is added.
     * Otherwise ggml's dependency-first traversal executes Q(CUDA)->qnorm(CPU),
     * returns to K(CUDA)->knorm(CPU), then V(CUDA)->HGA(CPU): six scheduler
     * splits per dense layer (98 for VERIFY).  Early expansion groups the
     * three projections into one CUDA segment and leaves one CPU HGA segment. */
    if (hga_decode_pack(cparams.hga_phase)) {
        ggml_build_forward_expand(gf, Qcur_full);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);
    }"""
    qkv_expand = """    /* Q RMS-normalization and K/V projection are independent CUDA work.
     * Expand all three roots before the CPU q/k-weight + RoPE + HGA chain.
     * Expanding only Qcur_full leaves Qcur_rms below the final HGA root and
     * produces CPU(V copy)->CUDA(Q RMS)->CPU(HGA) in every dense layer. */
    if (hga_decode_pack(cparams.hga_phase)) {
        ggml_build_forward_expand(gf, Qcur_rms);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);
    }"""
    qwen_txt = qwen.read_text(encoding="utf-8")
    if qkv_expand_marker in qwen_txt:
        print("  already patched: qwen35.cpp grouped decode Q/K/V expansion")
    elif qkv_expand_legacy in qwen_txt:
        qwen.write_text(qwen_txt.replace(qkv_expand_legacy, qkv_expand, 1), encoding="utf-8")
        print("  upgraded qwen35.cpp: grouped Q RMS/K/V expansion")
    elif gpu_kv_mm in qwen_txt or gpu_kv_mm_spaced in qwen_txt:
        kv_projection_block = gpu_kv_mm if gpu_kv_mm in qwen_txt else gpu_kv_mm_spaced
        qkv_grouped = kv_projection_block + "\n\n" + qkv_expand
        qwen.write_text(qwen_txt.replace(kv_projection_block, qkv_grouped, 1), encoding="utf-8")
        print("  patched qwen35.cpp: grouped decode Q/K/V expansion")
    else:
        die("qwen35.cpp: Q/K/V projection block missing for grouped expansion")
    # Repair a tree on which an older patcher run inserted the same expansion
    # twice.  Keep this normalization because third_party/ is intentionally
    # persistent across deploys.
    qwen_txt = qwen.read_text(encoding="utf-8")
    qkv_expand_twice = qkv_expand + "\n\n" + qkv_expand
    if qkv_expand_twice in qwen_txt:
        while qkv_expand_twice in qwen_txt:
            qwen_txt = qwen_txt.replace(qkv_expand_twice, qkv_expand, 1)
        qwen.write_text(qwen_txt, encoding="utf-8")
        print("  repaired qwen35.cpp: removed duplicate grouped Q/K/V expansion")
    once(qwen, "    // Apply MRoPE\n", "    ggml_tensor * Kraw = Kcur; // pre-RoPE, after k-norm (HGA summaries)\n")
    replace(
        qwen,
        """    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
""",
        """    if (cparams.hga_enabled) {
        cur = hga_build_full_attn(this, inp, Qcur, Kcur, Vcur, Kraw, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
""",
    )
    # Repair older 5-arg hga_build_full_attn call (missing hybrid KV inp).
    qt = qwen.read_text(encoding="utf-8")
    old_call = "cur = hga_build_full_attn(this, Qcur, Kcur, Vcur, Kraw, kq_scale, il);"
    new_call = "cur = hga_build_full_attn(this, inp, Qcur, Kcur, Vcur, Kraw, kq_scale, il);"
    if old_call in qt:
        qwen.write_text(qt.replace(old_call, new_call, 1), encoding="utf-8")
        print("  repaired hga_build_full_attn to keep hybrid KV inputs live")

    prefill_gpu_pins = """
    /* All attn linears on CUDA. HGA still CPU: K/V *activations* D2H after this. */
    if (cparams.hga_phase == HGA_SWAP_PREFILL || cparams.hga_phase == HGA_SWAP_DECODE) {
        hga_pin_gpu(sched, Qcur_full);
        hga_pin_gpu(sched, Kcur);
        hga_pin_gpu(sched, Vcur);
    }
"""
    old_prefill_only = """    /* Prefill: force Q/K/V GEMMs onto CUDA. Otherwise the CPU-pinned HGA op
     * pulls those mul_mats onto the host (copying 640 MiB of weights). */
    if (cparams.hga_phase == HGA_SWAP_PREFILL) {
        if (ggml_backend_t gpu = hga_sched_gpu_backend(sched)) {
            ggml_backend_sched_set_tensor_backend(sched, Qcur_full, gpu);
            ggml_backend_sched_set_tensor_backend(sched, Kcur, gpu);
            ggml_backend_sched_set_tensor_backend(sched, Vcur, gpu);
        }
    }
"""
    qt_pre = qwen.read_text(encoding="utf-8")
    old_decode_q_only = """    if (cparams.hga_phase == HGA_SWAP_PREFILL) {
        hga_pin_gpu(sched, Qcur_full);
        hga_pin_gpu(sched, Kcur);
        hga_pin_gpu(sched, Vcur);
    } else if (cparams.hga_phase == HGA_SWAP_DECODE) {
        hga_pin_gpu(sched, Qcur_full);
    }"""
    new_qkv_gpu = """    if (cparams.hga_phase == HGA_SWAP_PREFILL || cparams.hga_phase == HGA_SWAP_DECODE) {
        hga_pin_gpu(sched, Qcur_full);
        hga_pin_gpu(sched, Kcur);
        hga_pin_gpu(sched, Vcur);
    }"""
    old_decode_nested = """    /* Prefill: force Q/K/V GEMMs onto CUDA. Decode: Q stays on CUDA. */
    if (cparams.hga_phase == HGA_SWAP_PREFILL || cparams.hga_phase == HGA_SWAP_DECODE) {
        if (ggml_backend_t gpu = hga_sched_gpu_backend(sched)) {
            ggml_backend_sched_set_tensor_backend(sched, Qcur_full, gpu);
            if (cparams.hga_phase == HGA_SWAP_PREFILL) {
                ggml_backend_sched_set_tensor_backend(sched, Kcur, gpu);
                ggml_backend_sched_set_tensor_backend(sched, Vcur, gpu);
            } else if (cparams.hga_enabled && (cparams.hga_phase == HGA_SWAP_PREFILL || cparams.hga_phase == HGA_SWAP_DECODE)) {
                ggml_backend_sched_set_tensor_backend(sched, Kcur, gpu);
            }
        }
    }"""
    if old_decode_q_only in qt_pre:
        qwen.write_text(qt_pre.replace(old_decode_q_only, new_qkv_gpu, 1), encoding="utf-8")
        print("  upgraded qwen35.cpp: decode K/V GEMM on CUDA")
        qt_pre = qwen.read_text(encoding="utf-8")
    elif old_decode_nested in qt_pre:
        qwen.write_text(qt_pre.replace(old_decode_nested, new_qkv_gpu, 1), encoding="utf-8")
        print("  upgraded qwen35.cpp: decode K/V GEMM on CUDA (nested pins)")
        qt_pre = qwen.read_text(encoding="utf-8")
    if "hga_pin_gpu_pack(sched, Qcur_full" in qt_pre:
        print("  already patched: qwen35.cpp QKV GPU pack pins")
    elif "else if (cparams.hga_phase == HGA_SWAP_DECODE)" in qt_pre and "hga_pin_gpu(sched, Qcur_full)" in qt_pre:
        print("  already patched: qwen35.cpp decode Q GPU pin")
    elif old_prefill_only in qt_pre:
        qwen.write_text(qt_pre.replace(old_prefill_only, prefill_gpu_pins, 1), encoding="utf-8")
        print("  upgraded qwen35.cpp Q pin for decode")
    elif "hga_pin_gpu(sched, Qcur_full)" not in qt_pre and "hga_sched_gpu_backend" not in qt_pre:
        needle_v = """        Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
        cb(Vcur, "Vcur", il);
    }

    // Apply K normalization
"""
        insert_v = """        Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
        cb(Vcur, "Vcur", il);
    }
""" + prefill_gpu_pins + """
    // Apply K normalization
"""
        # Current upstream has unconditional K/V projections (and a blank
        # line before the K-normalization comment); older HGA trees retain
        # the conditional projection block above.
        upstream_needle_v = """    cb(Vcur, "Vcur", il);

    // Apply K normalization
"""
        upstream_insert_v = """    cb(Vcur, "Vcur", il);""" + prefill_gpu_pins + """

    // Apply K normalization
"""
        upstream_v_projection = """    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);"""
        if needle_v in qt_pre:
            qwen.write_text(qt_pre.replace(needle_v, insert_v, 1), encoding="utf-8")
        elif upstream_needle_v in qt_pre:
            qwen.write_text(qt_pre.replace(upstream_needle_v, upstream_insert_v, 1), encoding="utf-8")
        elif upstream_v_projection in qt_pre:
            # The grouped-expansion hook may already sit between Vcur and the
            # following K-normalization comment, so anchor on Vcur itself.
            qwen.write_text(
                qt_pre.replace(upstream_v_projection, upstream_v_projection + prefill_gpu_pins, 1),
                encoding="utf-8",
            )
        else:
            die("qwen35.cpp: Vcur/k-norm anchor not found for prefill GPU pins")
        print("  patched qwen35.cpp prefill GPU QKV pins")
    qt_qkv = qwen.read_text(encoding="utf-8")
    if new_qkv_gpu not in qt_qkv and "hga_pin_gpu_pack(sched, Qcur_full" not in qt_qkv:
        die("qwen35.cpp missing GPU Q/K/V GEMM pins after upgrade")

    new_qwen_pins = """
    /* Linears + K-norm stay on CUDA. Do not pin GPU K/V views to CPU — that
     * aliases device memory as a host pointer. HGA copies activations instead. */
    if (!cparams.offload_kqv) {
        if (cparams.hga_phase != HGA_SWAP_PREFILL && cparams.hga_phase != HGA_SWAP_DECODE) {
            ggml_backend_sched_set_tensor_backend(sched, Qcur, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, Kcur, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, Vcur, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, Kraw, backend_cpu);
        }
    }
"""
    old_qwen_pins = """
    /* Keep the whole dense-attn chain on CPU (QKV + q/k RMSNorm + RoPE + HGA).
     * o_proj / FFN / GDN stay on CUDA because their weights live there. */
    if (!cparams.offload_kqv) {
        ggml_backend_sched_set_tensor_backend(sched, Qcur, backend_cpu);
        ggml_backend_sched_set_tensor_backend(sched, Kcur, backend_cpu);
        ggml_backend_sched_set_tensor_backend(sched, Vcur, backend_cpu);
        ggml_backend_sched_set_tensor_backend(sched, Kraw, backend_cpu);
    }
"""
    crash_inner = """        if (cparams.hga_phase == HGA_SWAP_DECODE || cparams.hga_phase == HGA_SWAP_PREFILL) {
            ggml_backend_sched_set_tensor_backend(sched, Kcur, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, Vcur, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, Kraw, backend_cpu);
        } else {
            ggml_backend_sched_set_tensor_backend(sched, Qcur, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, Kcur, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, Vcur, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, Kraw, backend_cpu);
        }"""
    decode_only_inner = """        if (cparams.hga_phase == HGA_SWAP_DECODE) {
            ggml_backend_sched_set_tensor_backend(sched, Kcur, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, Vcur, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, Kraw, backend_cpu);
        } else if (cparams.hga_phase != HGA_SWAP_PREFILL) {
            ggml_backend_sched_set_tensor_backend(sched, Qcur, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, Kcur, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, Vcur, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, Kraw, backend_cpu);
        }"""
    none_inner = """        if (cparams.hga_phase != HGA_SWAP_PREFILL && cparams.hga_phase != HGA_SWAP_DECODE) {
            ggml_backend_sched_set_tensor_backend(sched, Qcur, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, Kcur, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, Vcur, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, Kraw, backend_cpu);
        }"""
    qt_pins = qwen.read_text(encoding="utf-8")
    if crash_inner in qt_pins:
        qwen.write_text(qt_pins.replace(crash_inner, none_inner, 1), encoding="utf-8")
        print("  upgraded qwen35.cpp: drop CPU pin of GPU K/V views")
        qt_pins = qwen.read_text(encoding="utf-8")
    elif decode_only_inner in qt_pins:
        qwen.write_text(qt_pins.replace(decode_only_inner, none_inner, 1), encoding="utf-8")
        print("  upgraded qwen35.cpp: NONE-only CPU pins")
        qt_pins = qwen.read_text(encoding="utf-8")
    if old_qwen_pins in qt_pins:
        qwen.write_text(qt_pins.replace(old_qwen_pins, new_qwen_pins, 1), encoding="utf-8")
        print("  updated qwen35.cpp pins for two-mode packing")
    once(
        qwen,
        """    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);
""",
        new_qwen_pins,
        marker="hga_phase != HGA_SWAP_PREFILL && cparams.hga_phase != HGA_SWAP_DECODE",
    )

    once(
        qwen,
        """    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);
""",
        """    if (cparams.hga_phase == HGA_SWAP_VERIFY) {
        Qcur = ggml_cont(ctx0, Qcur);
        hga_pin_gpu(sched, Qcur);
        cb(Qcur, "Qcur_cont", il);
    }
""",
    )

    qnorm_both = """    cb(Qcur, "Qcur_normed", il);
    if (cparams.hga_enabled &&
        (cparams.hga_phase == HGA_SWAP_DECODE || cparams.hga_phase == HGA_SWAP_PREFILL)) {
        hga_pin_gpu(sched, Qcur);
    }
"""
    qnorm_pack = """    cb(Qcur, "Qcur_normed", il);
    hga_pin_gpu_pack(sched, Qcur, cparams.hga_phase);
"""
    qnorm_split = """    cb(Qcur, "Qcur_normed", il);
    /* Prefill: q-norm CUDA. Decode: CPU — HGA needs Q anyway (one split). */
    hga_pin_gpu_prefill(sched, Qcur, cparams.hga_phase);
    if (cparams.hga_phase == HGA_SWAP_DECODE && backend_cpu) {
        ggml_backend_sched_set_tensor_backend(sched, Qcur, backend_cpu);
    }
"""
    qnorm_verify_gpu = """    cb(Qcur, "Qcur_normed", il);
    /* Prefill: q-norm stays CUDA (large activations). Decode n=1: CPU RMSNorm
     * (HGA already needs Q on the host). Verify K+1: GPU q-norm+rope, then
     * one contiguous D2H into HGA — do not pin the strided Q view to CPU. */
    if (cparams.hga_phase == HGA_SWAP_PREFILL || cparams.hga_phase == HGA_SWAP_VERIFY) {
        hga_pin_gpu(sched, Qcur);
    } else if (cparams.hga_phase == HGA_SWAP_DECODE && backend_cpu) {
        ggml_backend_sched_set_tensor_backend(sched, Qcur, backend_cpu);
    }
"""
    qnorm_verify_decode = """    cb(Qcur, "Qcur_normed", il);
    /* Prefill: q-norm CUDA. Decode and verify: CPU RMSNorm (HGA needs Q). */
    if (cparams.hga_phase == HGA_SWAP_PREFILL) {
        hga_pin_gpu(sched, Qcur);
    } else if (hga_decode_pack(cparams.hga_phase) && backend_cpu) {
        ggml_backend_sched_set_tensor_backend(sched, Qcur, backend_cpu);
    }
"""
    qt_qn = qwen.read_text(encoding="utf-8")
    if qnorm_verify_decode in qt_qn:
        print("  already patched: qwen35.cpp q-norm verify copies decode CPU")
    elif qnorm_verify_gpu in qt_qn:
        qwen.write_text(qt_qn.replace(qnorm_verify_gpu, qnorm_verify_decode, 1), encoding="utf-8")
        print("  upgraded qwen35.cpp: VERIFY q-norm copies DECODE (CPU)")
    elif qnorm_split in qt_qn:
        qwen.write_text(qt_qn.replace(qnorm_split, qnorm_verify_decode, 1), encoding="utf-8")
        print("  upgraded qwen35.cpp: q-norm verify copies decode CPU")
    elif qnorm_both in qt_qn:
        qwen.write_text(qt_qn.replace(qnorm_both, qnorm_verify_decode, 1), encoding="utf-8")
        print("  upgraded qwen35.cpp: q-norm prefill GPU, decode/verify CPU")
    elif qnorm_pack in qt_qn:
        qwen.write_text(qt_qn.replace(qnorm_pack, qnorm_verify_decode, 1), encoding="utf-8")
        print("  upgraded qwen35.cpp: q-norm prefill GPU, decode/verify CPU")
    else:
        once(
            qwen,
            '    cb(Qcur, "Qcur_normed", il);\n',
            """    if (cparams.hga_phase == HGA_SWAP_PREFILL) {
        hga_pin_gpu(sched, Qcur);
    } else if (hga_decode_pack(cparams.hga_phase) && backend_cpu) {
        ggml_backend_sched_set_tensor_backend(sched, Qcur, backend_cpu);
    }
""",
        )
    knorm_gpu = """    if (cparams.hga_phase == HGA_SWAP_PREFILL || cparams.hga_phase == HGA_SWAP_VERIFY) {
        hga_pin_gpu(sched, Kcur);
    } else if (cparams.hga_phase == HGA_SWAP_DECODE && backend_cpu) {
        ggml_backend_sched_set_tensor_backend(sched, Kcur, backend_cpu);
    }
"""
    knorm_verify_decode = """    if (cparams.hga_phase == HGA_SWAP_PREFILL) {
        hga_pin_gpu(sched, Kcur);
    } else if (hga_decode_pack(cparams.hga_phase) && backend_cpu) {
        ggml_backend_sched_set_tensor_backend(sched, Kcur, backend_cpu);
    }
"""
    knorm_decode_only = """    if (cparams.hga_phase == HGA_SWAP_DECODE && backend_cpu) {
        ggml_backend_sched_set_tensor_backend(sched, Kcur, backend_cpu);
    }
"""
    knorm_messy = """    cb(Kcur, "Kcur_normed", il);
    if (cparams.hga_enabled && cparams.hga_phase == HGA_SWAP_DECODE &&
            cparams.hga_enabled && (cparams.hga_phase == HGA_SWAP_PREFILL || cparams.hga_phase == HGA_SWAP_DECODE)) {
        hga_pin_gpu(sched, Kcur);
    }
"""
    knorm_clean = """    cb(Kcur, "Kcur_normed", il);
""" + knorm_verify_decode
    qt_kn = qwen.read_text(encoding="utf-8")
    if knorm_verify_decode in qt_kn:
        print("  already patched: qwen35.cpp K-norm verify copies decode CPU")
    elif knorm_messy in qt_kn:
        qwen.write_text(qt_kn.replace(knorm_messy, knorm_clean, 1), encoding="utf-8")
        print("  upgraded qwen35.cpp: K-norm CPU on decode")
    elif knorm_gpu in qt_kn:
        qwen.write_text(qt_kn.replace(knorm_gpu, knorm_verify_decode, 1), encoding="utf-8")
        print("  upgraded qwen35.cpp: VERIFY K-norm copies DECODE (CPU)")
    else:
        once(
            qwen,
            '    cb(Kcur, "Kcur_normed", il);\n',
            knorm_verify_decode,
        )
    knorm_old_decode = """    cb(Kcur, "Kcur_normed", il);
    if (cparams.hga_phase == HGA_SWAP_DECODE && backend_cpu) {
        ggml_backend_sched_set_tensor_backend(sched, Kcur, backend_cpu);
    }
"""
    knorm_new = """    cb(Kcur, "Kcur_normed", il);
""" + knorm_verify_decode
    qt_kn2 = qwen.read_text(encoding="utf-8")
    if knorm_old_decode in qt_kn2:
        qwen.write_text(qt_kn2.replace(knorm_old_decode, knorm_new, 1), encoding="utf-8")
        print("  upgraded qwen35.cpp: K-norm verify copies decode CPU")
    once(
        qwen,
        '    cb(gate, "gate_reshaped", il);\n',
        """    if (cparams.hga_enabled &&
        (cparams.hga_phase == HGA_SWAP_DECODE || cparams.hga_phase == HGA_SWAP_PREFILL)) {
        hga_pin_gpu(sched, gate);
    }
""",
        marker="hga_pin_gpu_pack(sched, gate",
    )
    qt_kraw = qwen.read_text(encoding="utf-8")
    old_rope_q = """    /* Decode: q-norm + rope stay on CUDA with Q. HGA copies ~24 KiB.
     * Leaving q-norm on the host while Q/gate are CUDA is the extra split. */
    if (cparams.hga_enabled && cparams.hga_phase == HGA_SWAP_DECODE) {
        hga_pin_gpu(sched, Qcur);
    }"""
    old_rope_q2 = """    /* Decode: q-norm + rope stay on CUDA with Q. HGA copies ~24 KiB. */
    if (cparams.hga_enabled && cparams.hga_phase == HGA_SWAP_DECODE) {
        hga_pin_gpu(sched, Qcur);
    }"""
    new_rope_qk = """    /* Decode q/k-norm+rope stay on CPU with HGA. No extra post-RoPE GPU pin. */"""
    if old_rope_q in qt_kraw:
        qwen.write_text(qt_kraw.replace(old_rope_q, new_rope_qk, 1), encoding="utf-8")
        print("  upgraded qwen35.cpp: post-RoPE Q+K GPU pins")
    elif old_rope_q2 in qt_kraw:
        qwen.write_text(qt_kraw.replace(old_rope_q2, new_rope_qk, 1), encoding="utf-8")
        print("  upgraded qwen35.cpp: post-RoPE Q+K GPU pins")
    elif "hga_pin_gpu(sched, Kcur)" in qt_kraw and "q/k-norm + rope stay on CUDA" in qt_kraw:
        print("  already patched: qwen35.cpp post-RoPE Q+K GPU pins")
    else:
        rope_anchor = """            ggml_backend_sched_set_tensor_backend(sched, Kraw, backend_cpu);
        }
    }

    // Attention computation
"""
        qt_now = qwen.read_text(encoding="utf-8")
        if rope_anchor in qt_now:
            qwen.write_text(qt_now.replace(rope_anchor, """            ggml_backend_sched_set_tensor_backend(sched, Kraw, backend_cpu);
        }
    }
""" + new_rope_qk + """

    // Attention computation
""", 1), encoding="utf-8")
            print("  patched qwen35.cpp post-RoPE comment")
        else:
            print("  already patched: qwen35.cpp post-RoPE (no Kraw insert)")

    qt_chk = qwen.read_text(encoding="utf-8")
    if crash_inner in qt_chk:
        die("qwen35.cpp still CPU-pins GPU K/V views")

    # Explicit H2D of HGA output so o_proj/FFN stay on CUDA.
    once(
        qwen,
        '    cb(cur, "attn_pregate", il);\n',
        """    if (cparams.hga_enabled &&
        (cparams.hga_phase == HGA_SWAP_DECODE || cparams.hga_phase == HGA_SWAP_PREFILL)) {
        cur = hga_copy_to_gpu(this, cur, "hga_attn_gpu");
    }
""",
        marker="hga_copy_to_gpu(this, cur, \"hga_attn_gpu\")",
    )
    once(
        qwen,
        '    cb(gate_sigmoid, "gate_sigmoid", il);\n',
        """    if (cparams.hga_enabled &&
        (cparams.hga_phase == HGA_SWAP_DECODE || cparams.hga_phase == HGA_SWAP_PREFILL)) {
        hga_pin_gpu(sched, gate_sigmoid);
    }
""",
        marker="hga_pin_gpu_pack(sched, gate_sigmoid",
    )

    qt_g = qwen.read_text(encoding="utf-8")
    attn_tail_end = "ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear"
    attn_tail_new = """    cb(cur, "attn_gated", il);
    if (cparams.hga_enabled &&
        (cparams.hga_phase == HGA_SWAP_DECODE || cparams.hga_phase == HGA_SWAP_PREFILL)) {
        hga_pin_gpu(sched, cur);
    }

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);
    if (cparams.hga_enabled &&
        (cparams.hga_phase == HGA_SWAP_DECODE || cparams.hga_phase == HGA_SWAP_PREFILL)) {
        hga_pin_gpu(sched, cur);
    }

    return cur;
}

"""
    start = qt_g.find('    cb(cur, "attn_gated", il);')
    end = qt_g.find(attn_tail_end)
    if start < 0 or end < 0 or end < start:
        die("qwen35.cpp: attn_gated / build_layer_attn_linear anchors not found")
    current_tail = qt_g[start:end]
    if "hga_pin_gpu_pack" in current_tail or (
            "hga_copy_to_gpu" in qt_g and current_tail.count("hga_pin_gpu(sched, cur)") >= 2):
        print("  already patched: qwen35.cpp GPU gate-mul + o_proj")
    else:
        qwen.write_text(qt_g[:start] + attn_tail_new + qt_g[end:], encoding="utf-8")
        print("  patched qwen35.cpp: explicit HGA H2D + GPU o_proj")

    pin_prefill_host = """    if (cparams.hga_enabled && cparams.hga_phase == HGA_SWAP_PREFILL && backend_cpu) {
        ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
    }"""
    pin_prefill_if_host = """    if (cparams.hga_enabled && cparams.hga_phase == HGA_SWAP_PREFILL && backend_cpu &&
        (!model.output || !model.output->buffer || ggml_backend_buffer_is_host(model.output->buffer))) {
        ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
    }"""
    pin_spec_host = """    if (cparams.hga_enabled && backend_cpu &&
            (cparams.hga_phase == HGA_SWAP_PREFILL || hga_lmhead_on_host(cparams)) &&
            (!model.output || !model.output->buffer || ggml_backend_buffer_is_host(model.output->buffer))) {
        ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
    }"""
    pin_gpu_head = """    if (cparams.hga_enabled && backend_cpu &&
            (!model.output || !model.output->buffer || ggml_backend_buffer_is_host(model.output->buffer))) {
        ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
    }"""
    qt_pin = qwen.read_text(encoding="utf-8")
    if pin_gpu_head in qt_pin:
        print("  already patched: qwen35.cpp logits pin (host weight only)")
    elif pin_spec_host in qt_pin:
        qwen.write_text(qt_pin.replace(pin_spec_host, pin_gpu_head), encoding="utf-8")
        print("  updated qwen35.cpp: logits pin only when lm_head is host")
    else:
        if pin_prefill_host in qt_pin:
            qwen.write_text(qt_pin.replace(pin_prefill_host, pin_gpu_head), encoding="utf-8")
            print("  updated qwen35.cpp: logits pin for host lm_head")
            qt_pin = qwen.read_text(encoding="utf-8")
        if pin_prefill_if_host in qt_pin:
            qwen.write_text(qt_pin.replace(pin_prefill_if_host, pin_gpu_head), encoding="utf-8")
            print("  updated qwen35.cpp: logits pin for host lm_head")
        replace(
            qwen,
            '    cb(cur, "result_output", -1);\n    res->t_logits = cur;\n',
            """    cb(cur, "result_output", -1);
    if (cparams.hga_enabled && backend_cpu &&
            (!model.output || !model.output->buffer || ggml_backend_buffer_is_host(model.output->buffer))) {
        ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
    }
    res->t_logits = cur;
""",
        )
        replace(
            qwen,
            '    cb(cur, "result_output", -1);\n\n    res->t_logits = cur;\n',
            """    cb(cur, "result_output", -1);
    if (cparams.hga_enabled && backend_cpu &&
            (!model.output || !model.output->buffer || ggml_backend_buffer_is_host(model.output->buffer))) {
        ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
    }

    res->t_logits = cur;
""",
        )

    # Decode-only extras. Prefill keeps the older pin set (~200 t/s).
    # Marker is needle+insert so identical pin lines at different sites still apply.
    def pin_once(needle: str, insert: str) -> None:
        once(qwen, needle, insert, marker=needle + insert)

    def pack_to_decode(needle_pack: str, needle_dec: str) -> None:
        text = qwen.read_text(encoding="utf-8")
        if needle_dec in text:
            return
        if needle_pack not in text:
            return
        qwen.write_text(text.replace(needle_pack, needle_dec, 1), encoding="utf-8")
        print("  upgraded qwen35.cpp: extra norm pin decode-only")

    pin_dec = "hga_pin_gpu_decode(sched, cur, cparams.hga_phase);"
    pin_pack = "hga_pin_gpu_pack(sched, cur, cparams.hga_phase);"
    pack_to_decode(
        '        cb(cur, "attn_norm", il);\n        ' + pin_pack,
        '        cb(cur, "attn_norm", il);\n        ' + pin_dec)
    pack_to_decode(
        '        cb(attn_post_norm, "attn_post_norm", il);\n        hga_pin_gpu_pack(sched, attn_post_norm, cparams.hga_phase);',
        '        cb(attn_post_norm, "attn_post_norm", il);\n        hga_pin_gpu_decode(sched, attn_post_norm, cparams.hga_phase);')
    pack_to_decode(
        '    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);\n    ' + pin_pack,
        '    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);\n    ' + pin_dec)
    pack_to_decode(
        '    cb(cur, "result_norm", -1);\n    ' + pin_pack,
        '    cb(cur, "result_norm", -1);\n    ' + pin_dec)
    pack_to_decode(
        '    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);\n    hga_pin_gpu_pack(sched, normalized, cparams.hga_phase);',
        '    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);\n    hga_pin_gpu_decode(sched, normalized, cparams.hga_phase);')
    knorm_cpu = """    cb(Kcur, "Kcur_normed", il);
    if (cparams.hga_phase == HGA_SWAP_DECODE && backend_cpu) {
        ggml_backend_sched_set_tensor_backend(sched, Kcur, backend_cpu);
    }"""
    pack_to_decode(
        '    cb(Kcur, "Kcur_normed", il);\n    hga_pin_gpu_pack(sched, Kcur, cparams.hga_phase);',
        knorm_cpu)
    pack_to_decode(
        '    cb(Kcur, "Kcur_normed", il);\n    hga_pin_gpu_decode(sched, Kcur, cparams.hga_phase);',
        knorm_cpu)
    pack_to_decode(
        """    if (cparams.hga_enabled &&
        (cparams.hga_phase == HGA_SWAP_DECODE || cparams.hga_phase == HGA_SWAP_PREFILL)) {
        hga_pin_gpu(sched, Kcur);
    }""",
        """    if (cparams.hga_phase == HGA_SWAP_DECODE && backend_cpu) {
        ggml_backend_sched_set_tensor_backend(sched, Kcur, backend_cpu);
    }""")
    pack_to_decode(
        """    /* q/k-norm + rope stay on CUDA with the linears. */
    if (cparams.hga_enabled &&
        (cparams.hga_phase == HGA_SWAP_DECODE || cparams.hga_phase == HGA_SWAP_PREFILL)) {
        hga_pin_gpu(sched, Qcur);
        hga_pin_gpu(sched, Kcur);
    }""",
        "    /* Decode q/k-norm+rope stay on CPU with HGA. */")
    pack_to_decode(
        """    /* Decode: q/k-rope stay on CUDA with the linears. Prefill keeps the
     * original q-norm + QKV GEMM pins only. */
    hga_pin_gpu_decode(sched, Qcur, cparams.hga_phase);
    hga_pin_gpu_decode(sched, Kcur, cparams.hga_phase);""",
        "    /* Decode q/k-norm+rope stay on CPU with HGA. */")
    pack_to_decode(
        """    hga_pin_gpu_decode(sched, Qcur, cparams.hga_phase);
    hga_pin_gpu_decode(sched, Kcur, cparams.hga_phase);""",
        "    /* Decode q/k-norm+rope stay on CPU with HGA. */")
    leftover_knorm = """    hga_pin_gpu_decode(sched, Kcur, cparams.hga_phase);
    if (cparams.hga_enabled &&
        (cparams.hga_phase == HGA_SWAP_DECODE || cparams.hga_phase == HGA_SWAP_PREFILL)) {
        hga_pin_gpu(sched, Kcur);
    }
"""
    pack_to_decode(leftover_knorm, knorm_gpu)
    pack_to_decode(
        """    if (cparams.hga_phase == HGA_SWAP_DECODE && backend_cpu) {
        ggml_backend_sched_set_tensor_backend(sched, Kcur, backend_cpu);
    }
    hga_pin_gpu_decode(sched, Kcur, cparams.hga_phase);
""",
        """    if (cparams.hga_phase == HGA_SWAP_DECODE && backend_cpu) {
        ggml_backend_sched_set_tensor_backend(sched, Kcur, backend_cpu);
    }
""")

    pin_once('        cb(cur, "attn_norm", il);\n',
             "        hga_pin_gpu_decode(sched, cur, cparams.hga_phase);\n")
    pin_once('        cb(attn_post_norm, "attn_post_norm", il);\n',
             "        hga_pin_gpu_decode(sched, attn_post_norm, cparams.hga_phase);\n")
    pin_once('    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);\n',
             "    hga_pin_gpu_decode(sched, cur, cparams.hga_phase);\n")
    pin_once('    cb(cur, "result_norm", -1);\n',
             "    hga_pin_gpu_decode(sched, cur, cparams.hga_phase);\n")
    pin_once('    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);\n',
             "    hga_pin_gpu_decode(sched, normalized, cparams.hga_phase);\n")
    # MTP is a separate one-block dense-attention graph. It needs its own HGA
    # path as well: otherwise every draft token grows an un-routed KV history.
    # Do this after the generic MTP norm pins above, so the block is stable on
    # both a fresh upstream tree and an already HGA-patched tree.
    # Keep this independent from the full MTP replacement below. That makes a
    # rerun repair a worktree that was interrupted after replacing the
    # attention call but before adding Kraw.
    mtp_kraw = """    hga_pin_gpu_pack(sched, Vcur, cparams.hga_phase);

    /* HGA summaries use post-norm, pre-RoPE keys just like the trunk. */
    ggml_tensor * Kraw = Kcur;
"""
    mtp_kraw_legacy = """    hga_pin_gpu_pack(sched, Vcur, cparams.hga_phase);

    /* HGA summaries use the post-norm, pre-RoPE keys just like the trunk. */
    ggml_tensor * Kraw = Kcur;
"""
    mtp_txt = qwen.read_text(encoding="utf-8")
    if mtp_kraw + mtp_kraw_legacy in mtp_txt:
        qwen.write_text(mtp_txt.replace(mtp_kraw + mtp_kraw_legacy, mtp_kraw, 1), encoding="utf-8")
        print("  repaired qwen35.cpp: duplicate MTP Kraw declaration")
    elif mtp_kraw_legacy in mtp_txt:
        qwen.write_text(mtp_txt.replace(mtp_kraw_legacy, mtp_kraw, 1), encoding="utf-8")
        print("  upgraded qwen35.cpp: legacy MTP Kraw declaration")
    once(
        qwen,
        '    cb(Vcur, "mtp_Vcur", il);\n',
        mtp_kraw,
        marker="HGA summaries use post-norm, pre-RoPE keys just like the trunk.",
    )
    mtp_txt = qwen.read_text(encoding="utf-8")
    if "mtp_hga_attn_gpu" in mtp_txt:
        print("  already patched: qwen35.cpp MTP HGA")
    else:
        once(
            qwen,
            '    cb(Qcur_full, "mtp_Qcur_full", il);\n',
            '    hga_pin_gpu_pack(sched, Qcur_full, cparams.hga_phase);\n',
            marker='mtp_Qcur_full", il);\n    hga_pin_gpu_pack',
        )
        mtp_old_attn = """    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);
"""
        mtp_new_attn = """    if (cparams.hga_runtime) {
        cur = hga_build_full_attn(this, inp_attn, Qcur, Kcur, Vcur, Kraw,
                                  kq_scale, il);
    } else {
        cur = build_attn(inp_attn,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "mtp_attn_pregate", il);
    if (cparams.hga_runtime && hga_gpu_pack(cparams.hga_phase)) {
        cur = hga_copy_to_gpu(this, cur, "mtp_hga_attn_gpu");
    }

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    hga_pin_gpu_pack(sched, gate_sigmoid, cparams.hga_phase);
    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    hga_pin_gpu_pack(sched, cur, cparams.hga_phase);
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);
    hga_pin_gpu_pack(sched, cur, cparams.hga_phase);
"""
        replace(qwen, mtp_old_attn, mtp_new_attn)
        print("  patched qwen35.cpp: MTP HGA attention")

    # The MTP HGA graph is always decode-packed: its projections and post-HGA
    # chain remain on CUDA, while hga_build_full_attn makes the compact D2H/H2D
    # boundary.  Normalize older decode+pack combinations before adding packs,
    # so rerunning the installer never leaves duplicate scheduler pins.
    mtp_pack_sites = (
        ('    cb(h_norm, "mtp_hnorm", il);\n', "h_norm"),
        ('    cb(e_norm, "mtp_enorm", il);\n', "e_norm"),
        ('    cb(cur, "mtp_attn_norm", il);\n', "cur"),
        ('    cb(cur, "mtp_attn_post_norm", il);\n', "cur"),
        ('    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);\n', "cur"),
    )
    for mtp_site, mtp_tensor in mtp_pack_sites:
        mtp_decode = f"    hga_pin_gpu_decode(sched, {mtp_tensor}, cparams.hga_phase);\n"
        mtp_pack = f"    hga_pin_gpu_pack(sched, {mtp_tensor}, cparams.hga_phase);\n"
        maybe_replace(qwen, mtp_site + mtp_decode + mtp_pack, mtp_site + mtp_pack)
        maybe_replace(qwen, mtp_site + mtp_pack + mtp_decode, mtp_site + mtp_pack)
        maybe_replace(qwen, mtp_site + mtp_decode, mtp_site + mtp_pack)
        once(qwen, mtp_site, mtp_pack, marker=mtp_site + mtp_pack)

    # MTP packing must follow the trunk: large catch-up = PREFILL (no
    # contiguous D2H of the whole prompt), draft/verify-sized = DECODE.  Find
    # the position inputs inside graph_mtp instead of matching the following
    # comment: that comment changes after the dense-KV removal patch below and
    # made a second installer run silently omit this safety-critical phase.
    mtp_phase = (
        "    if (cparams.hga_runtime) {\n"
        "        const_cast<llama_cparams &>(cparams).hga_phase =\n"
        "                n_tokens > 8 ? HGA_SWAP_PREFILL : HGA_SWAP_DECODE;\n"
        "    }\n\n"
    )
    qwen_txt = qwen.read_text(encoding="utf-8")
    if mtp_phase in qwen_txt:
        print("  already patched: qwen35.cpp MTP HGA phase")
    else:
        mtp_start = qwen_txt.find("llama_model_qwen35::graph_mtp::graph_mtp")
        mtp_pos = (
            "    ggml_tensor * inp_pos     = build_inp_pos();\n"
            "    ggml_tensor * inp_out_ids = build_inp_out_ids();\n\n"
        )
        mtp_pos_at = qwen_txt.find(mtp_pos, mtp_start)
        if mtp_start < 0 or mtp_pos_at < 0:
            die("qwen35.cpp: graph_mtp position-input anchor not found")
        insert_at = mtp_pos_at + len(mtp_pos)
        qwen_txt = qwen_txt[:insert_at] + mtp_phase + qwen_txt[insert_at:]
        qwen.write_text(qwen_txt, encoding="utf-8")
        print("  patched qwen35.cpp: MTP HGA phase")
    once(
        qwen,
        '    cb(Qcur, "mtp_Qcur_normed", il);\n',
        '    hga_pin_gpu_pack(sched, Qcur, cparams.hga_phase);\n',
        marker='cb(Qcur, "mtp_Qcur_reshaped", il);',
    )
    maybe_replace(
        qwen,
        '    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);\n'
        '    cb(Qcur, "mtp_Qcur_normed", il);\n'
        '    hga_pin_gpu_pack(sched, Qcur, cparams.hga_phase);\n',
        '    cb(Qcur, "mtp_Qcur_reshaped", il);\n'
        '    if (cparams.hga_runtime && n_tokens > 1 && hga_decode_pack(cparams.hga_phase)) {\n'
        '        Qcur = ggml_cont(ctx0, Qcur);\n'
        '        hga_pin_gpu(sched, Qcur);\n'
        '        cb(Qcur, "mtp_Qcur_cont", il);\n'
        '    }\n'
        '    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);\n'
        '    cb(Qcur, "mtp_Qcur_normed", il);\n'
        '    if (cparams.hga_phase == HGA_SWAP_PREFILL) {\n'
        '        hga_pin_gpu(sched, Qcur);\n'
        '    } else if (hga_decode_pack(cparams.hga_phase) && backend_cpu) {\n'
        '        ggml_backend_sched_set_tensor_backend(sched, Qcur, backend_cpu);\n'
        '    } else {\n'
        '        hga_pin_gpu_pack(sched, Qcur, cparams.hga_phase);\n'
        '    }\n',
    )
    maybe_replace(
        qwen,
        '    cb(gate, "mtp_gate", il);\n\n'
        '    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);\n',
        '    cb(gate, "mtp_gate", il);\n'
        '    hga_pin_gpu_pack(sched, gate, cparams.hga_phase);\n\n'
        '    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);\n',
    )
    once(
        qwen,
        '    cb(Kcur, "mtp_Kcur_normed", il);\n',
        '    hga_pin_gpu_pack(sched, Kcur, cparams.hga_phase);\n',
        marker=(
            'cb(Kcur, "mtp_Kcur_normed", il);\n'
            '    if (cparams.hga_phase == HGA_SWAP_PREFILL)'
        ),
    )
    maybe_replace(
        qwen,
        '    cb(Kcur, "mtp_Kcur_normed", il);\n'
        '    hga_pin_gpu_pack(sched, Kcur, cparams.hga_phase);\n',
        '    cb(Kcur, "mtp_Kcur_normed", il);\n'
        '    if (cparams.hga_phase == HGA_SWAP_PREFILL) {\n'
        '        hga_pin_gpu(sched, Kcur);\n'
        '    } else if (hga_decode_pack(cparams.hga_phase) && backend_cpu) {\n'
        '        ggml_backend_sched_set_tensor_backend(sched, Kcur, backend_cpu);\n'
        '    } else {\n'
        '        hga_pin_gpu_pack(sched, Kcur, cparams.hga_phase);\n'
        '    }\n',
    )
    # Optional hard scheduler-placement experiment.  This is deliberately
    # applied to every named Qwen module tensor, rather than only QKV/o_proj:
    # HGA_PREFILL_PIN_ALL=1 therefore prevents op_offload from choosing CPU for
    # any named PREFILL node.  The helper is a no-op outside PREFILL and HGA's
    # internal CPU routing/staging nodes retain their explicit assignments.
    qwen_txt = qwen.read_text(encoding="utf-8")
    cb_pattern = re.compile(r'^(\s*)cb\((\w+),([^\n]+)\);\n', re.MULTILINE)
    qwen_pinned, n_probe = cb_pattern.subn(
        lambda m: (
            f'{m.group(1)}cb({m.group(2)},{m.group(3)});\n'
            f'{m.group(1)}hga_pin_gpu_prefill_probe('
            f'sched, {m.group(2)}, cparams.hga_phase);\n'
        ),
        qwen_txt,
    )
    if n_probe < 50:
        die(f"qwen35.cpp prefill pin probe found only {n_probe} cb() sites")
    qwen.write_text(qwen_pinned, encoding="utf-8")
    print(f"  patched qwen35.cpp: PREFILL pin probe on {n_probe} named tensors")

    # HGA owns the MTP layer's routed KV state.  Keep llama_kv_cache for its
    # slot/sequence bookkeeping, but allocate no dense K/V tensors in that
    # otherwise one-layer MTP cache.  The graph above likewise skips
    # build_attn_inp_kv() in HGA mode, so no cpy_k/cpy_v can address it.
    maybe_replace(
        root / "src" / "models" / "qwen35.cpp",
        "    auto * inp_attn = build_attn_inp_kv();\n\n    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);\n",
        "    /* HGA owns MTP KV.  Do not create dense-KV graph inputs. */\n"
        "    llm_graph_input_attn_kv * inp_attn = cparams.hga_runtime\n"
        "            ? nullptr\n"
        "            : build_attn_inp_kv();\n\n"
        "    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);\n",
    )
    maybe_replace(
        root / "src" / "llama-model.cpp",
        """                    if (mtp_on_hybrid_qwen || mtp_on_hybrid_nemotron) {
                        filter = [&](uint32_t il) { return il >= hparams.n_layer(); };
                    }
""",
        """                    if (mtp_on_hybrid_qwen || mtp_on_hybrid_nemotron) {
                        // The MTP HGA runtime owns its routed KV state.  Leave
                        // this cache's cells available for sequence operations,
                        // but do not allocate the duplicate dense K/V layer.
                        if (cparams.hga_enabled) {
                            filter = [](uint32_t) { return false; };
                        } else {
                            filter = [&](uint32_t il) { return il >= hparams.n_layer(); };
                        }
                    }
""",
    )

    maybe_replace(
        root / "src" / "llama-context.cpp",
        "        /*.hga_keep_last               =*/ 8,",
        "        /*.hga_keep_last               =*/ 7,",
    )
    maybe_replace(
        root / "src" / "llama-cparams.h",
        "    int32_t hga_keep_last = 8;",
        "    int32_t hga_keep_last = 7;",
    )
    maybe_replace(
        root / "common" / "common.h",
        "    int32_t hga_keep_last  = 8;",
        "    int32_t hga_keep_last  = 7;",
    )
    # Upgrade the previous production default as well as older upstream
    # experiments. The launcher passes the value explicitly, but keeping all
    # public/default parameter surfaces aligned prevents a direct llama-server
    # invocation from silently reverting to four local chunks.
    maybe_replace(
        root / "src" / "llama-context.cpp",
        "        /*.hga_keep_last               =*/ 4,",
        "        /*.hga_keep_last               =*/ 7,",
    )
    maybe_replace(
        root / "src" / "llama-cparams.h",
        "    int32_t hga_keep_last = 4;",
        "    int32_t hga_keep_last = 7;",
    )
    maybe_replace(
        root / "common" / "common.h",
        "    int32_t hga_keep_last  = 4;",
        "    int32_t hga_keep_last  = 7;",
    )

    # Prefill must not force every "norm" onto CUDA — that was the ~200 t/s
    # graph (q/k-norm stay with the host mmap weights). Decode pins those
    # ops explicitly via hga_pin_gpu_decode.
    ctx_cpp = root / "src" / "llama-context.cpp"
    ctx_txt = ctx_cpp.read_text(encoding="utf-8")
    pack_norm_if = """                if (hga_gpu_pack(cparams.hga_phase) ||
                    !( !cparams.offload_kqv && strcmp(name, \"norm\") == 0 )) {"""
    orig_norm_if = """                if (!( !cparams.offload_kqv && strcmp(name, \"norm\") == 0 && il < (int) model.hparams.n_layer() )) {"""
    if pack_norm_if in ctx_txt:
        ctx_cpp.write_text(ctx_txt.replace(pack_norm_if, orig_norm_if, 1), encoding="utf-8")
        print("  restored llama-context.cpp: prefill does not force RMSNorm onto CUDA")
        ctx_txt = ctx_cpp.read_text(encoding="utf-8")
    if orig_norm_if in ctx_txt or "When KV/QKV stay on the host, do not force RMSNorm" in ctx_txt:
        print("  already patched: llama-context.cpp graph_get_cb RMSNorm")
    else:
        replace(
            ctx_cpp,
            """            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
""",
            """            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                // When KV/QKV stay on the host, do not force RMSNorm onto the
                // layer GPU — that copies Q/K to CUDA just for q/k-norm, then
                // back (6 splits per dense layer). attn_norm / post_attn_norm
                // still run on CUDA because their weights and neighbors are GPU.
                if (!( !cparams.offload_kqv && strcmp(name, "norm") == 0 && il < (int) model.hparams.n_layer() )) {
                    const auto & dev_layer = model.dev_layer(il);
                    for (const auto & backend : backends) {
                        if (ggml_backend_get_device(backend.get()) == dev_layer) {
                            if (ggml_backend_supports_op(backend.get(), cur)) {
                                ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                            }
                        }
                    }
                }
            }
""",
        )

    patch_offload_rs(root)
    patch_hga_ubatch_pad(root)
    patch_cuda_vmm_shrink(root)
    patch_cuda_alloc_attribution(root)
    patch_cuda_alloc_ledger(root)
    patch_chunked_gpu_load(root)
    patch_async_eval_events(root)
    patch_cuda_i8_cast(root)
    patch_hga_skip_llama_attn_kv(root)
    patch_generate_k1(root)
    patch_spec_step_prof(root)
    patch_server_critical_path_prof(root)

    llama_h = root / "include" / "llama.h"
    if "llama_n_rs_seq" not in llama_h.read_text(encoding="utf-8"):
        once(
            llama_h,
            "    LLAMA_API uint32_t llama_n_seq_max  (const struct llama_context * ctx);\n",
            "    LLAMA_API uint32_t llama_n_rs_seq   (const struct llama_context * ctx);\n",
        )
    ctx_cpp_api = root / "src" / "llama-context.cpp"
    if "uint32_t llama_n_rs_seq" not in ctx_cpp_api.read_text(encoding="utf-8"):
        once(
            ctx_cpp_api,
            "uint32_t llama_n_seq_max(const llama_context * ctx) {\n    return ctx->n_seq_max();\n}\n",
            "\nuint32_t llama_n_rs_seq(const llama_context * ctx) {\n    return ctx->get_cparams().n_rs_seq;\n}\n",
        )

    # MTP draft context: upstream hard-codes n_rs_seq=0, so
    # common_context_can_seq_rm(ctx_dft) probes seq_rm(0,1,-1), which FAILS on
    # the hybrid recurrent memory -> type FULL -> speculative-simple sets
    # use_ckpt_dft=true and round-trips the draft context's sequence state
    # (llama_state_seq get/set PARTIAL_ONLY = all 48 GDN R/S planes, ~150 MiB
    # that graph_mtp never reads) on EVERY spec step. That was ~150-250 ms/step
    # (~43 % of spec wall time; the low-rate "background" PCIe RX in dmon).
    # Mirror the target's n_rs_seq: can_seq_rm then reports RS (bounded
    # rollback) and the example uses plain llama_memory_seq_rm instead.
    replace(
        root / "common" / "speculative.cpp",
        "    cparams.n_rs_seq  = 0;\n"
        "    cparams.ctx_other = ctx_tgt;",
        "    // HGA: mirror the target's bounded-rollback window. n_rs_seq=0 forces\n"
        "    // per-step llama_state_seq checkpoints of ~150 MiB of unused GDN state.\n"
        "    cparams.n_rs_seq  = ctx_tgt ? llama_n_rs_seq(ctx_tgt) : 0;\n"
        "    cparams.ctx_other = ctx_tgt;",
    )
    once(
        root / "examples" / "speculative-simple" / "speculative-simple.cpp",
        "    if (use_ckpt_tgt) {\n"
        "        LOG_INF(\"speculative decoding will use checkpoints (context does not support partial sequence removal)\\n\");\n"
        "    }\n",
        "    LOG_INF(\"speculative seq_rm ckpt tgt=%d dft=%d  n_rs_seq tgt=%u dft=%u\\n\",\n"
        "            (int) use_ckpt_tgt, (int) use_ckpt_dft,\n"
        "            llama_n_rs_seq(ctx_tgt), llama_n_rs_seq(ctx_dft));\n",
    )

    # One-shot MTP spec binary for scripts/run_hga.sh HGA_SPEC=K. EXAMPLES is
    # off in setup.sh (we do not want the whole examples/ tree).
    once(
        root / "CMakeLists.txt",
        "if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_EXAMPLES)\n    add_subdirectory(examples)\n    add_subdirectory(pocs)\nendif()\n",
        """
# HGA: MTP speculative one-shot used by scripts/run_hga.sh HGA_SPEC=K
if (LLAMA_BUILD_COMMON AND NOT LLAMA_BUILD_EXAMPLES)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/speculative-simple/CMakeLists.txt")
        find_package(Threads REQUIRED)
        add_subdirectory(examples/speculative-simple)
    endif()
endif()
        """,
    )

    patch_graph_recipe_cache(root)
    patch_lazy_prefix_checkpoints(root)

    print("HGA patch applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
