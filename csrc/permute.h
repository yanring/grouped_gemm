#pragma once

#include <torch/extension.h>

namespace grouped_gemm {

std::tuple<torch::Tensor, torch::Tensor> moe_permute_op(torch::Tensor original_input,torch::Tensor expert_for_rows);

torch::Tensor moe_recover_op(torch::Tensor permuted_input, torch::Tensor source_row_to_dest_row);

}  // namespace grouped_gemm
