# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import unittest
import torch.cuda.nvtx as nvtx

try:
  from grouped_gemm import permute, unpermute, gmm
except ImportError:
  print("grouped-gemm toolkit is not installed. Fall back to local import.")
  # For local debug
  from moe.ops import permute, unpermute, gmm


class TestMoeOps(unittest.TestCase):

  def setUp(self) -> None:
    torch.manual_seed(734876213)

################################################################################################
##
## Test Helpers
##
################################################################################################

  def permute_ops_helper(self,
                         num_rows,
                         max_token_num,
                         num_cols,
                         num_experts,
                         dtype,
                         atol,
                         execution_times,
                         PRINT):

    # Prepare inputs
    expert_for_rows = torch.randint(size=(num_rows,),low=0,high=num_experts, dtype=torch.int32).cuda()
    # For debug test, this will get a source_row_to_dest_row of [7, 2, 0, 9, 5, 1, 6, 3, 8, 4]
    # expert_for_rows = torch.tensor([3, 1, 0, 4, 2, 0, 2, 1, 3, 1])
    unpermuted_inputs = torch.empty(size=(num_rows, num_cols), dtype=dtype).cuda()
    for i in range(num_rows):
        unpermuted_inputs[i] = i
    unpermuted_inputs.requires_grad_(True)
    original_inputs = unpermuted_inputs.detach()

    # Build network
    for _ in range(execution_times):
      # Forward
      # shape mismatch test
      # expert_for_rows = torch.nn.functional.pad(expert_for_rows, [0, 1])

      nvtx.range_push("permute op forward")
      permuted_inputs, source_row_to_dest_row = permute(unpermuted_inputs, expert_for_rows, max_token_num)
      nvtx.range_pop()

      # shape mismatch test
      # expert_for_rows = torch.nn.functional.pad(expert_for_rows, [0, 1])
      
      nvtx.range_push("unpermute op forward")
      unpermute_outputs = unpermute(permuted_inputs, expert_for_rows, source_row_to_dest_row, max_token_num)
      nvtx.range_pop()

      # Reset grad to avoid accumulation
      unpermuted_inputs.grad = torch.zeros_like(unpermuted_inputs)
      permuted_inputs.grad = torch.zeros_like(permuted_inputs)

      # Backward
      nvtx.range_push("permute & unpermute op backward")
      unpermute_outputs.backward(unpermute_outputs.detach())
      nvtx.range_pop()

    if PRINT:
      print("expert_for_rows: {}".format(expert_for_rows))
      print("unpermuted_inputs: {}".format(unpermuted_inputs))
      print("permuted_inputs: {}".format(permuted_inputs))
      print("unpermute_outputs: {}".format(unpermute_outputs))
      print("original_inputs: {}".format(original_inputs))
      print("backward: {}".format(unpermuted_inputs.grad))

    # Result check
    original_inputs = original_inputs.float().cpu().numpy().flatten()
    original_output = unpermute_outputs.float().cpu().detach().numpy().flatten()
    max_abs_error = abs(original_inputs - original_output).max()
    print(f"permute & unpermute forward max error: \t\t{max_abs_error:.3e} ({dtype})")
    assert (max_abs_error < atol), "test_moe_permute failed!"

    original_output = unpermuted_inputs.grad.float().cpu().numpy().flatten()
    max_abs_error = abs(original_inputs - original_output).max()
    print(f"permute & unpermute backward max error: \t{max_abs_error:.3e} ({dtype})")
    assert (max_abs_error < atol), "test_moe_permute failed!"

  def groupedgemm_ops_helper(self,
                             num_rows,
                             hidden_size,
                             inter_size,
                             num_experts,
                             transB,
                             dtype,
                             atol,
                             execution_times,
                             PRINT):
    # Prepare inputs
    rand_mean = 0
    rand_std = 0.02

    expert_for_rows = torch.randint(size=(num_rows,),low=0,high=num_experts, dtype=torch.int32).cuda()
    tokens_per_expert = torch.bincount(expert_for_rows, minlength=num_experts)
    tokens_per_expert = tokens_per_expert.to(torch.int32)

    permuted_inputs = torch.empty([num_rows, hidden_size], dtype=dtype, device="cuda").normal_(rand_mean, rand_std)
    if not transB:
      weights = torch.empty([num_experts, hidden_size, inter_size], dtype=dtype, device="cuda").normal_(rand_mean, rand_std)    
    else:
      weights = torch.empty([num_experts, inter_size, hidden_size], dtype=dtype, device="cuda").normal_(rand_mean, rand_std)    
    
    permuted_inputs.requires_grad_(True)
    weights.requires_grad_(True)

    # Build network
    for _ in range(execution_times):
      # Forward
      nvtx.range_push("grouped gemm op forward")
      
      # shape mismatch test
      # weights = torch.nn.functional.pad(weights, [0, 0, 0, 1])

      gemm_output = gmm(permuted_inputs, weights, tokens_per_expert, transB)
      nvtx.range_pop()

      # Reset grad to avoid accumulation
      permuted_inputs.grad = torch.zeros_like(permuted_inputs)
      weights.grad = torch.zeros_like(weights)

      # Backward
      nvtx.range_push("grouped gemm op backward")
      gemm_output.backward(gemm_output.detach())
      nvtx.range_pop()

    # Ref calculation
    gemm_output_ref_list = []
    weight_grad_ref_list = []
    activation_grad_ref_list = []

    rows_idx_for_expert = torch.cumsum(tokens_per_expert, dim=0)
    rows_idx_for_expert = torch.cat((torch.tensor([0]).cuda(), rows_idx_for_expert[:-1]))

    for expert_id in range(num_experts):
      row_start_id = rows_idx_for_expert[expert_id]
      row_end_id = row_start_id + tokens_per_expert[expert_id]

      activations_expert = permuted_inputs[row_start_id:row_end_id].detach()
      weights_expert = weights[expert_id].detach()
      if transB:
        weights_expert = weights_expert.T
      activations_expert.requires_grad_(True)
      weights_expert.requires_grad_(True)
      
      gemm_output_ref = torch.matmul(activations_expert, weights_expert)
      gemm_output_ref.backward(gemm_output_ref.detach())

      gemm_output_ref_list.append(gemm_output_ref)
      weights_expert_grad = weights_expert.grad
      if transB:
        weights_expert_grad = weights_expert_grad.T
      weight_grad_ref_list.append(weights_expert_grad.unsqueeze(0))
      activation_grad_ref_list.append(activations_expert.grad)

    gemm_output_ref = torch.cat(gemm_output_ref_list, dim=0)
    weight_grad_ref = torch.cat(weight_grad_ref_list, dim=0)
    activation_grad_ref = torch.cat(activation_grad_ref_list, dim=0)

    if PRINT:
      print(expert_for_rows)
      # Forward
      print(gemm_output)
      print(gemm_output_ref)
      # Backward
      print(permuted_inputs.grad)
      print(activation_grad_ref)
      print(weights.grad)
      print(weight_grad_ref)

    # Result check
    gemm_output = gemm_output.float().cpu().detach().numpy().flatten()
    gemm_output_ref = gemm_output_ref.float().cpu().detach().numpy().flatten()
    max_abs_error = abs(gemm_output - gemm_output_ref).max()
    print(f"group gemm forward max error: \t\t\t{max_abs_error:.3e} ({dtype})")
    assert (max_abs_error < atol), "test_moe_groupedgemm failed!"

    gemm_output = permuted_inputs.grad.float().cpu().detach().numpy()
    gemm_output_ref = activation_grad_ref.float().cpu().detach().numpy()
    max_abs_error = abs(gemm_output - gemm_output_ref).max()
    print(f"group gemm backward activation.grad max error: \t{max_abs_error:.3e} ({dtype})")
    assert (max_abs_error < atol), "test_moe_groupedgemm failed!"

    gemm_output = weights.grad.float().cpu().detach().numpy().flatten()
    gemm_output_ref = weight_grad_ref.float().cpu().detach().numpy().flatten()
    max_abs_error = abs(gemm_output - gemm_output_ref).max()
    print(f"group gemm backward weight.grad max error: \t{max_abs_error:.3e} ({dtype})")
    assert (max_abs_error < atol), "test_moe_groupedgemm failed!"

################################################################################################
##
## Test Cases
##
################################################################################################

  def test_moe_permute(self):
    num_rows =        4096 * 2
    max_token_num =   num_rows + 10
    num_cols =        2048
    num_experts =     8
    atol =            1e-5
    execution_times = 10
    PRINT =           False

    print()
    dtype = torch.float32
    self.permute_ops_helper(num_rows, max_token_num, num_cols, num_experts, dtype, atol, execution_times, PRINT)
    dtype = torch.float16
    self.permute_ops_helper(num_rows, max_token_num, num_cols, num_experts, dtype, atol, execution_times, PRINT)
    dtype = torch.bfloat16
    self.permute_ops_helper(num_rows, max_token_num, num_cols, num_experts, dtype, atol, execution_times, PRINT)

  def test_moe_groupedgemm(self):
    # Note that the test directly uses the forward result as the input for the backward process, 
    # so the max error of the backward result is the accumulation of errors from both the forward 
    # and backward processes.

    num_rows =        4096 * 2
    hidden_size =     2048
    inter_size =      hidden_size * 4
    num_experts =     8
    atol =            1e-2
    execution_times = 10
    PRINT =           False

    print()
    transB = False
    dtype = torch.float32
    self.groupedgemm_ops_helper(num_rows, hidden_size, inter_size, num_experts, transB, dtype, atol, execution_times, PRINT)
    dtype = torch.float16
    self.groupedgemm_ops_helper(num_rows, hidden_size, inter_size, num_experts, transB, dtype, atol, execution_times, PRINT)
    dtype = torch.bfloat16
    self.groupedgemm_ops_helper(num_rows, hidden_size, inter_size, num_experts, transB, dtype, atol, execution_times, PRINT)
    transB = True
    self.groupedgemm_ops_helper(num_rows, hidden_size, inter_size, num_experts, transB, dtype, atol, execution_times, PRINT)

def test_ops():
  loader = unittest.TestLoader()
  suite = loader.loadTestsFromTestCase(TestMoeOps)
  runner = unittest.TextTestRunner()
  runner.run(suite)


if __name__ == '__main__':
  test_ops()
