# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set(CMAKE_CXX_STANDARD 17)

add_library(cutlass_heuristic STATIC cutlass_heuristic.cc)
set_property(TARGET cutlass_heuristic PROPERTY POSITION_INDEPENDENT_CODE ON)

set(moe_gemm_files "")
file(GLOB moe_gemm_files ${moe_gemm_files} moe_gemm/moe_gemm_kernels_*.cu)
add_library(moe_gemm_kernels STATIC ${moe_gemm_files})
set_property(TARGET moe_gemm_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET moe_gemm_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(moe_gemm_kernels PRIVATE cutlass_heuristic)
