#!/usr/bin/env python3
"""Copy HGA sources into a llama.cpp checkout and apply the graph/CLI hooks."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def once(path: Path, needle: str, insert: str, *, after: bool = True) -> None:
    text = path.read_text(encoding="utf-8")
    if insert.strip() in text:
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


def replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text and old not in text:
        print(f"  already patched: {path.name}")
        return
    if old not in text:
        die(f"replace target not found in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"  patched {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("llama_cpp", type=Path, help="path to llama.cpp checkout")
    args = ap.parse_args()
    root: Path = args.llama_cpp.resolve()
    if not (root / "src" / "models" / "qwen35.cpp").is_file():
        die(f"{root} does not look like llama.cpp (missing src/models/qwen35.cpp)")

    src_hga_h = HERE / "cpp" / "include" / "hga.h"
    src_hga_c = HERE / "cpp" / "src" / "hga.cpp"
    glue_h = HERE / "llama.cpp-hga" / "llama-hga.h"
    glue_c = HERE / "llama.cpp-hga" / "llama-hga.cpp"
    for p in (src_hga_h, src_hga_c, glue_h, glue_c):
        if not p.is_file():
            die(f"missing {p}")

    shutil.copy2(src_hga_h, root / "src" / "hga.h")
    shutil.copy2(src_hga_c, root / "src" / "hga.cpp")
    shutil.copy2(glue_h, root / "src" / "llama-hga.h")
    shutil.copy2(glue_c, root / "src" / "llama-hga.cpp")
    print("copied hga.{h,cpp} llama-hga.{h,cpp}")

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
        "target_link_libraries(llama PUBLIC ggml)\n",
        """
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(llama PRIVATE OpenMP::OpenMP_CXX)
endif()
""",
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
    )

    once(
        root / "src" / "llama-context.cpp",
        "        /*.ctx_other                   =*/ nullptr,\n",
        """        /*.hga_enabled                 =*/ false,
        /*.hga_levels                  =*/ 1,
        /*.hga_chunk_size              =*/ 64,
        /*.hga_group_size              =*/ 16,
        /*.hga_keep_first              =*/ 2,
        /*.hga_keep_last               =*/ 8,
        /*.hga_frac_l1                 =*/ 0.08f,
        /*.hga_frac_l2                 =*/ 0.04f,
""",
    )

    once(
        root / "src" / "llama-cparams.h",
        "    llama_context * ctx_other;\n",
        """    bool hga_enabled = false;
    int32_t hga_levels = 1;
    int32_t hga_chunk_size = 64;
    int32_t hga_group_size = 16;
    int32_t hga_keep_first = 2;
    int32_t hga_keep_last = 8;
    float hga_frac_l1 = 0.08f;
    float hga_frac_l2 = 0.04f;
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
        root / "common" / "common.h",
        "    enum llama_flash_attn_type   flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_AUTO; // whether to use Flash Attention\n",
        """
    bool    hga_enabled    = false;
    int32_t hga_levels     = 1;
    int32_t hga_chunk_size = 64;
    int32_t hga_group_size = 16;
    int32_t hga_keep_first = 2;
    int32_t hga_keep_last  = 8;
    float   hga_frac_l1    = 0.08f;
    float   hga_frac_l2    = 0.04f;
""",
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
        "Hierarchical Global Attention on Qwen3.5/3.8 full-attention layers (CPU exact-token routing)",
        [](common_params & params, bool value) {
            params.hga_enabled = value;
        }
    ).set_env("LLAMA_ARG_HGA"));
    add_opt(common_arg(
        {"--hga-levels"}, "N",
        "HGA hierarchy depth: 1 = whole 64-token chunks (~8% tokens), 2 = groups of 16 (~4% tokens). Same chunk count.",
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
        string_format("1-level attended-token fraction (default: %.2f)", (double) params.hga_frac_l1),
        [](common_params & params, const std::string & value) { params.hga_frac_l1 = std::stof(value); }
    ));
    add_opt(common_arg(
        {"--hga-frac-l2"}, "F",
        string_format("2-level attended-token fraction (default: %.2f)", (double) params.hga_frac_l2),
        [](common_params & params, const std::string & value) { params.hga_frac_l2 = std::stof(value); }
    ));
""",
    )

    # INT8 KV + QK dots (re-runnable on an already-patched tree).
    once(root / "include" / "llama.h", "        float   hga_frac_l2;\n", "        bool    hga_i8;\n")
    once(
        root / "src" / "llama-context.cpp",
        "        /*.hga_frac_l2                 =*/ 0.04f,\n",
        "        /*.hga_i8                      =*/ true,\n",
    )
    once(root / "src" / "llama-cparams.h", "    float hga_frac_l2 = 0.04f;\n", "    bool hga_i8 = true;\n")
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
        string_format("2-level attended-token fraction (default: %.2f)", (double) params.hga_frac_l2),
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

    qwen = root / "src" / "models" / "qwen35.cpp"
    once(qwen, '#include "models.h"\n', '#include "llama-hga.h"\n')
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

    print("HGA patch applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
