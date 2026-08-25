#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace hga {

enum class Residency : uint8_t {
    HostMmap,
    GpuResident,
    StreamSlot,
    CpuState,
    GpuState,
};

struct ExchangePair {
    uint8_t layer_a;
    uint8_t layer_b;
    bool streamed_in_verify;
};

/* The loader supplies these values from GGUF metadata before any allocation.
 * Keeping this small makes an accidental near-Qwen checkpoint a hard error. */
struct ModelShape {
    uint32_t layer_count;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t mtp_layers;
};

enum class ProfileError : uint8_t {
    None,
    LayerCount,
    HeadCount,
    HeadDimension,
    RotaryDimension,
    MtpLayerCount,
};

/* Reference 16 GB packing. Runtime thread count follows the host
 * (scripts/env.sh); verify key tiles scale as 4 KV heads × tiles ≤ threads. */
class Vram16Qwen38Profile final {
public:
    static constexpr uint32_t kPrefillWidth = 768;
    static constexpr uint32_t kVerifyWidth = 3;
    static constexpr uint32_t kMtpK = 2;
    static constexpr uint32_t kChunkSize = 64;
    static constexpr uint32_t kGroupSize = 16;
    static constexpr uint32_t kSinkChunks = 2;
    static constexpr uint32_t kLocalChunks = 7;
    static constexpr uint32_t kHgaThreads = 12;
    static constexpr uint32_t kVerifyKTiles = 3;

    constexpr const std::array<uint8_t, 16> &full_attention_layers() const {
        return full_attention_layers_;
    }
    constexpr const std::array<ExchangePair, 8> &exchange_pairs() const {
        return exchange_pairs_;
    }
    constexpr bool cuda_graphs_enabled() const { return false; }
    constexpr ProfileError validate(const ModelShape &shape) const {
        return shape.layer_count != 64 ? ProfileError::LayerCount :
               shape.q_heads != 24 || shape.kv_heads != 4 ? ProfileError::HeadCount :
               shape.head_dim != 256 ? ProfileError::HeadDimension :
               shape.rotary_dim != 64 ? ProfileError::RotaryDimension :
               shape.mtp_layers != 1 ? ProfileError::MtpLayerCount : ProfileError::None;
    }

private:
    inline static constexpr std::array<uint8_t, 16> full_attention_layers_ = {
        3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63};
    /* Two pairs remain streamed after the prefill-to-VERIFY transition. The
     * other six are pinned then, rather than policy-switched. */
    inline static constexpr std::array<ExchangePair, 8> exchange_pairs_ = {{
        {0, 32, false}, {4, 36, false}, {8, 40, false}, {12, 44, false},
        {16, 48, true}, {20, 52, false}, {24, 56, false}, {28, 60, true},
    }};
};

using Turing1Qwen38Profile = Vram16Qwen38Profile;

} // namespace hga
