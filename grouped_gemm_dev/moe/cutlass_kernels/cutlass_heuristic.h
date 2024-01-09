/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <string>
#include <vector>
#include "moe_gemm/ft_gemm_configs.h"

namespace groupedgemmformoe {

std::vector<CutlassGemmConfig> get_candidate_configs(int sm, const bool is_weight_only, const bool simt_configs_only);

CutlassGemmConfig estimate_best_config_from_occupancies(const std::vector<CutlassGemmConfig>& candidate_configs,
                                                        const std::vector<int>&               occupancies,
                                                        const int64_t                         m,
                                                        const int64_t                         n,
                                                        const int64_t                         k,
                                                        const int64_t                         num_experts,
                                                        const int                             split_k_limit,
                                                        const size_t                          workspace_bytes,
                                                        const int                             multi_processor_count,
                                                        const int                             is_weight_only);

}  // namespace groupedgemmformoe