/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Debug Helper
//
/////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void device_print(int *var, int count)
{
    printf("device ");
    for (int i = 0; i < count; i++)
    {
        printf("%d ", var[i]);
    }
    printf("\n");
}

__global__ void device_print_int64(int64_t *var, int count)
{
    printf("device ");
    for (int i = 0; i < count; i++)
    {
        printf("%ld ", var[i]);
    }
    printf("\n");
}

template <typename T>
__global__ void device_print_fp(T *var, int count)
{
    printf("device ");
    for (int i = 0; i < count; i++)
    {
        printf("%f ", float(var[i]));
    }
    printf("\n");
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Util Functions
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline T *get_ptr(torch::Tensor &t)
{
    return reinterpret_cast<T *>(t.data_ptr());
}

template <typename T>
__global__ void compute_rows_for_expert_kernel(
    const T *expert_for_rows,
    int *rows_per_expert,
    const int num_rows,
    const int num_experts)
{
    int tid = threadIdx.x;
    int tnum = blockDim.x;

    for (int expert_id = tid; expert_id < num_experts; expert_id += tnum)
    {
        rows_per_expert[expert_id] = 0;
    }

    __syncthreads();

    for (int row_id = tid; row_id < num_rows; row_id += tnum)
    {
        int expert_id = expert_for_rows[row_id];
        atomicAdd(&rows_per_expert[expert_id], 1);
    }
}

void compute_rows_for_expert(
    torch::Tensor expert_for_rows,
    int *rows_per_expert,
    const int num_rows,
    const int num_experts,
    cudaStream_t stream)
{
    if (expert_for_rows.is_cpu())
    {
        expert_for_rows = expert_for_rows.to(torch::kCUDA);
    }

    const at::ScalarType _st = expert_for_rows.scalar_type();
    switch (_st) {
        case at::ScalarType::Long: {
            using T = int64_t;

            const T *expert_for_rows_ptr = get_ptr<T>(expert_for_rows);
            compute_rows_for_expert_kernel<T><<<1, 1024, 0, stream>>>(
                expert_for_rows_ptr,
                rows_per_expert,
                num_rows,
                num_experts);
            break;
        }
        case at::ScalarType::Int: {
            using T = int;

            const T *expert_for_rows_ptr = get_ptr<T>(expert_for_rows);
            compute_rows_for_expert_kernel<T><<<1, 1024, 0, stream>>>(
                expert_for_rows_ptr,
                rows_per_expert,
                num_rows,
                num_experts);
            break;
        }
        default:
            throw std::runtime_error("Wrong expert_for_rows tensor type.");
    }
}