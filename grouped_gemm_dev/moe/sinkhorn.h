/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <torch/torch.h>

namespace torch_ext {

torch::Tensor sinkhorn(torch::Tensor cost, const double tol=0.0001);

}  // namespace torch_ext
