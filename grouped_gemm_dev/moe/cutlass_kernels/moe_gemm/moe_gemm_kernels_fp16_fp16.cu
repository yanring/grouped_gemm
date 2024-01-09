/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "moe/cutlass_kernels/moe_gemm/moe_gemm_forward_template.h"

namespace groupedgemmformoe {
template class MoeGemmRunner<half, half>;
} // namespace groupedgemmformoe