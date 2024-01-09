/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "moe/cutlass_kernels/moe_gemm/moe_gemm_forward_template.h"

namespace groupedgemmformoe {
#ifdef ENABLE_BF16
template class MoeGemmRunner<__nv_bfloat16, __nv_bfloat16>;
#endif
} // namespace groupedgemmformoe