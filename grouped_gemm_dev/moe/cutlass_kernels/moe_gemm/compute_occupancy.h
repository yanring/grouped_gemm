/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <cuda_runtime_api.h>

#include "cutlass/device_kernel.h"
#include "moe/cutlass_kernels/th_utils.h"

namespace groupedgemmformoe {

template<typename GemmKernel>
inline int compute_occupancy_for_kernel()
{

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    if (smem_size > (48 << 10)) {
        cudaError_t status =
            cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        if (status == cudaError::cudaErrorInvalidValue) {
            // Clear the error bit since we can ignore this.
            // This should mean that smem_size > cudaDevAttrMaxSharedMemoryPerBlockOptin. In that case, we return an
            // occupancy of 0. This will cause the heuristic to ignore this configuration.
            status = cudaGetLastError();
            return 0;
        }
        check_cuda_error(status);
    }

    int max_active_blocks = -1;
    check_cuda_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, cutlass::Kernel<GemmKernel>, GemmKernel::kThreadCount, smem_size));

    return max_active_blocks;
}

}  // namespace groupedgemmformoe