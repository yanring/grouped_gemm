/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

namespace groupedgemmformoe {
// Note: The shapes are in the format MxNxK. The K shape of the runtime config MUST match the K shape
//       in the kernel layout details when doing weight only quantization.
enum class CutlassTileConfig {
    // Signals that we should run heuristics do choose a config
    Undefined,

    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic,

    // SiMT config
    CtaShape128x128x8_WarpShape64x64x8,

    // TensorCore configs CTA_N = 128, CTA_K = 64
    // Warp configs for M=32
    CtaShape32x128x64_WarpShape32x32x64,

    // Warp configs for M=64
    CtaShape64x128x64_WarpShape32x64x64,
    CtaShape64x128x64_WarpShape64x32x64,

    // Warp configs for M=128
    CtaShape128x128x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape128x32x64
};

enum class SplitKStyle {
    NO_SPLIT_K,
    SPLIT_K_SERIAL,
    // SPLIT_K_PARALLEL // Not supported yet
};

struct CutlassGemmConfig {
    CutlassTileConfig tile_config    = CutlassTileConfig::ChooseWithHeuristic;
    SplitKStyle       split_k_style  = SplitKStyle::NO_SPLIT_K;
    int               split_k_factor = -1;
    int               stages         = -1;
};

}  // namespace groupedgemmformoe