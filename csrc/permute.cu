#include "permute.h"

#include <torch/torch.h>
#include <cub/cub.cuh>
// #ifdef ENABLE_BF16
#include <cuda_bf16.h>
// #endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ATen/cuda/CUDAContext.h"

// #include "cutlass_kernels/th_utils.h"
// #include "cutlass_kernels/moe_gemm/moe_gemm_kernels.h"

// #include "cutlass_kernels/moe_gemm/moe_gemm_utils.h"
// #include "cutlass_kernels/moe_gemm/moe_gemm_backward_template.h"

using torch::Tensor;

namespace grouped_gemm {


template <typename T>
inline T *get_ptr(torch::Tensor &t)
{
    return reinterpret_cast<T *>(t.data_ptr());
}


template <typename T>
__global__ void moe_permute_kernel(const T *original_input,
                                   T *permuted_output,
                                   const int *map_dest_row_to_source_row,
                                   int *map_source_row_to_dest_row,
                                   const int num_rows,
                                   const int num_cols)
{
    // Reverse permutation map.
    // each block corresponds to one row
    const int dest_row = blockIdx.x;

    if (dest_row >= num_rows)
        return;

    int source_row = map_dest_row_to_source_row[dest_row];

    if (threadIdx.x == 0)
    {
        // write the map for the following unpermuting
        map_source_row_to_dest_row[source_row] = dest_row;
    }

    // permute activations rows based on experts
    const T *source_row_ptr = original_input + source_row * num_cols;
    T *dest_row_ptr = permuted_output + dest_row * num_cols;

    for (int tid = threadIdx.x; tid < num_cols; tid += blockDim.x)
    {
        dest_row_ptr[tid] = source_row_ptr[tid];
    }
}

template <typename T>
__global__ void moe_recover_kernel(const T *original_input,
                                   T *permuted_output,
                                   const int *map_dest_row_to_source_row,
                                   const int num_rows,
                                   const int num_cols)
{
    // Reverse permutation map.
    // each block corresponds to one row
    const int dest_row = blockIdx.x;

    if (dest_row >= num_rows)
        return;

    int source_row = map_dest_row_to_source_row[dest_row];

    // permute activations rows based on experts
    const T *source_row_ptr = original_input + source_row * num_cols;
    T *dest_row_ptr = permuted_output + dest_row * num_cols;

    for (int tid = threadIdx.x; tid < num_cols; tid += blockDim.x)
    {
        dest_row_ptr[tid] = source_row_ptr[tid];
    }
}

template <typename T>
void moe_permute_kernel_launcher(
    const T *original_input,
    T *permuted_output,
    const int *map_dest_row_to_source_row,
    int *map_source_row_to_dest_row,
    const int num_rows,
    const int num_cols,
    cudaStream_t stream)
{
    const int blocks = num_rows;
    const int threads = std::min(num_cols, 1024);

    if (map_source_row_to_dest_row != nullptr)
    {

        moe_permute_kernel<T><<<blocks, threads, 0, stream>>>(original_input,
                                                              permuted_output,
                                                              map_dest_row_to_source_row,
                                                              map_source_row_to_dest_row,
                                                              num_rows,
                                                              num_cols);
    }
    else
    {
        moe_recover_kernel<T><<<blocks, threads, 0, stream>>>(original_input,
                                                              permuted_output,
                                                              map_dest_row_to_source_row,
                                                              num_rows,
                                                              num_cols);
    }
}

std::tuple<torch::Tensor, torch::Tensor> moe_permute_op(
    Tensor original_input,
    Tensor expert_for_rows)
{
    const int num_rows = original_input.size(0);
    const int num_cols = original_input.size(1);

    if (original_input.is_cpu())
    {
        original_input = original_input.to(torch::kCUDA);
    }
    expert_for_rows = expert_for_rows.to(torch::kInt32);
    if (expert_for_rows.is_cpu())
    {
        expert_for_rows = expert_for_rows.to(torch::kCUDA);
    }

    auto options = torch::TensorOptions()
                       .dtype(torch::kInt32)
                       .device(torch::kCUDA)
                       .requires_grad(false);
    Tensor row_id = torch::range(0, num_rows - 1, 1, options);
    Tensor sorted_expert_for_rows = torch::empty(num_rows, options);
    Tensor dest_row_to_source_row = torch::empty(num_rows, options);

    int *expert_for_rows_ptr = get_ptr<int>(expert_for_rows);
    int *row_id_ptr = get_ptr<int>(row_id);
    int *sorted_expert_for_rows_ptr = get_ptr<int>(sorted_expert_for_rows);
    int *dest_row_to_source_row_ptr = get_ptr<int>(dest_row_to_source_row);

    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    expert_for_rows_ptr, sorted_expert_for_rows_ptr,
                                    row_id_ptr, dest_row_to_source_row_ptr, num_rows);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    expert_for_rows_ptr, sorted_expert_for_rows_ptr,
                                    row_id_ptr, dest_row_to_source_row_ptr, num_rows);

    // activations type
    const at::ScalarType _st = original_input.scalar_type();
    Tensor permuted_output =
        torch::empty({num_rows, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    int *&map_dest_row_to_source_row = dest_row_to_source_row_ptr;

    Tensor source_row_to_dest_row = torch::empty_like(sorted_expert_for_rows);
    int *map_source_row_to_dest_row = get_ptr<int>(source_row_to_dest_row);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;

        dType *original_input_ptr = get_ptr<dType>(original_input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_kernel_launcher<dType>(
            original_input_ptr,
            permuted_output_ptr,
            map_dest_row_to_source_row,
            map_source_row_to_dest_row,
            num_rows,
            num_cols,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = half;

        dType *original_input_ptr = get_ptr<dType>(original_input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_kernel_launcher<dType>(
            original_input_ptr,
            permuted_output_ptr,
            map_dest_row_to_source_row,
            map_source_row_to_dest_row,
            num_rows,
            num_cols,
            stream);

        break;
    }
// #ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        using dType = __nv_bfloat16;

        dType *original_input_ptr = get_ptr<dType>(original_input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_kernel_launcher<dType>(
            original_input_ptr,
            permuted_output_ptr,
            map_dest_row_to_source_row,
            map_source_row_to_dest_row,
            num_rows,
            num_cols,
            stream);

        break;
    }
// #endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    cudaStreamSynchronize(stream);

    return std::make_tuple(permuted_output, source_row_to_dest_row);
}

torch::Tensor moe_recover_op(
    Tensor permuted_input,
    Tensor source_row_to_dest_row)
{
    const int num_rows = permuted_input.size(0);
    const int num_cols = permuted_input.size(1);

    if (permuted_input.is_cpu())
    {
        permuted_input = permuted_input.to(torch::kCUDA);
    }
    if (source_row_to_dest_row.is_cpu())
    {
        source_row_to_dest_row = source_row_to_dest_row.to(torch::kCUDA);
    }

    // activations type
    const at::ScalarType _st = permuted_input.scalar_type();
    Tensor original_output =
        torch::empty({num_rows, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

    int *map_source_row_to_dest_row = get_ptr<int>(source_row_to_dest_row);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;

        dType *permuted_input_ptr = get_ptr<dType>(permuted_input);
        dType *original_output_ptr = get_ptr<dType>(original_output);

        moe_permute_kernel_launcher<dType>(
            permuted_input_ptr,
            original_output_ptr,
            map_source_row_to_dest_row,
            nullptr,
            num_rows,
            num_cols,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = half;

        dType *permuted_input_ptr = get_ptr<dType>(permuted_input);
        dType *original_output_ptr = get_ptr<dType>(original_output);

        moe_permute_kernel_launcher<dType>(
            permuted_input_ptr,
            original_output_ptr,
            map_source_row_to_dest_row,
            nullptr,
            num_rows,
            num_cols,
            stream);

        break;
    }
// #ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        using dType = __nv_bfloat16;

        dType *permuted_input_ptr = get_ptr<dType>(permuted_input);
        dType *original_output_ptr = get_ptr<dType>(original_output);

        moe_permute_kernel_launcher<dType>(
            permuted_input_ptr,
            original_output_ptr,
            map_source_row_to_dest_row,
            nullptr,
            num_rows,
            num_cols,
            stream);

        break;
    }
// #endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    cudaStreamSynchronize(stream);

    return original_output;
}


}  // namespace grouped_gemm
