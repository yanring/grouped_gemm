# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import os

so_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/build'
torch.classes.load_library(so_dir + '/libmoe_unit_ops.so')

# TODO by Jiang Shao, add parameter `out` which can be optionally given to be used as output buffers.

################################################################################################
##
## PermuteMoE
##
################################################################################################

class PermuteMoE(torch.autograd.Function):
  
  workspace_fw=None
  workspace_bw=None
  dtype=None
  max_token_num=0

  @staticmethod
  def forward(ctx, 
              unpermuted_inputs: torch.Tensor,
              expert_for_rows: torch.Tensor,
              max_token_num: int):
    
    # Shape check
    if unpermuted_inputs.size(0) != expert_for_rows.size(0):
      raise RuntimeError(f"Error: permute op input expert_for_rows shape mismatch! \
Expect {unpermuted_inputs.size(0)}, but got {expert_for_rows.size(0)}.")

    if PermuteMoE.max_token_num < max_token_num:
      # print("Permute op workspace reset!")
      PermuteMoE.max_token_num = max_token_num
      PermuteMoE.workspace_fw = []
      PermuteMoE.workspace_bw = []

    if PermuteMoE.max_token_num < unpermuted_inputs.size(0) or PermuteMoE.dtype != unpermuted_inputs.dtype:
      # print("Permute op workspace reset!")
      PermuteMoE.max_token_num = unpermuted_inputs.size(0)
      PermuteMoE.dtype = unpermuted_inputs.dtype
      PermuteMoE.workspace_fw = []
      PermuteMoE.workspace_bw = []

    permuted_inputs, source_row_to_dest_row, PermuteMoE.workspace_fw = torch.ops.moe_unit_ops.moe_permute_op(
      unpermuted_inputs,
      expert_for_rows,
      PermuteMoE.workspace_fw,
      PermuteMoE.max_token_num)

    ctx.source_row_to_dest_row = source_row_to_dest_row

    return permuted_inputs, source_row_to_dest_row

  @staticmethod
  def backward(ctx, permuted_inputs_grad, _):
    permuted_inputs_grad = permuted_inputs_grad.contiguous()
    source_row_to_dest_row = ctx.source_row_to_dest_row

    original_output, PermuteMoE.workspace_bw = torch.ops.moe_unit_ops.moe_recover_op(
      permuted_inputs_grad,
      source_row_to_dest_row,
      PermuteMoE.workspace_bw,
      PermuteMoE.max_token_num)

    return original_output, None, None

################################################################################################
##
## UnpermuteMoE
##
################################################################################################

class UnpermuteMoE(torch.autograd.Function):

  workspace_fw=None
  workspace_bw=None
  dtype=None
  max_token_num=0
  
  @staticmethod
  def forward(ctx,
              permuted_inputs: torch.Tensor,
              expert_for_rows: torch.Tensor,
              source_row_to_dest_row: torch.Tensor,
              max_token_num: int):

    # Shape check
    if permuted_inputs.size(0) != expert_for_rows.size(0):
      raise RuntimeError(f"Error: unpermute op input expert_for_rows shape mismatch! \
Expect {permuted_inputs.size(0)}, but got {expert_for_rows.size(0)}.")

    if UnpermuteMoE.max_token_num < max_token_num:
      # print("Unpermute op workspace reset!")
      UnpermuteMoE.max_token_num = max_token_num
      UnpermuteMoE.workspace_fw = []
      UnpermuteMoE.workspace_bw = []

    if UnpermuteMoE.max_token_num < permuted_inputs.size(0) or UnpermuteMoE.dtype != permuted_inputs.dtype:
      # print("Unpermute op workspace reset!")
      UnpermuteMoE.max_token_num = permuted_inputs.size(0)
      UnpermuteMoE.dtype = permuted_inputs.dtype
      UnpermuteMoE.workspace_fw = []
      UnpermuteMoE.workspace_bw = []

    ctx.expert_for_rows = expert_for_rows

    original_output, UnpermuteMoE.workspace_bw = torch.ops.moe_unit_ops.moe_recover_op(
      permuted_inputs,
      source_row_to_dest_row,
      UnpermuteMoE.workspace_bw,
      UnpermuteMoE.max_token_num)
    
    return original_output
  
  @staticmethod
  def backward(ctx, unpermuted_inputs_grad):

    unpermuted_inputs_grad = unpermuted_inputs_grad.contiguous()
    expert_for_rows = ctx.expert_for_rows

    permuted_inputs, _, UnpermuteMoE.workspace_fw = torch.ops.moe_unit_ops.moe_permute_op(
      unpermuted_inputs_grad,
      expert_for_rows,
      UnpermuteMoE.workspace_fw,
      UnpermuteMoE.max_token_num)

    return permuted_inputs, None, None, None

################################################################################################
##
## GroupedGemmMoE
##
################################################################################################

class GroupedGemmMoE(torch.autograd.Function):

  @staticmethod
  def forward(ctx,
              permuted_inputs: torch.Tensor,
              weights: torch.Tensor,
              tokens_per_expert: torch.Tensor,
              transB: bool):

    # Shape check
    if not transB and permuted_inputs.size(1) != weights.size(1):
      raise RuntimeError(f"Error: groupedgemm op input weights shape mismatch! \
Expect {permuted_inputs.size(1)}, but got {weights.size(1)}.")
    if transB and permuted_inputs.size(1) != weights.size(2):
      raise RuntimeError(f"Error: groupedgemm op input weights shape mismatch! \
Expect {permuted_inputs.size(1)}, but got {weights.size(2)}.")

    output = torch.ops.moe_unit_ops.moe_group_gemm_op(
      permuted_inputs,
      weights,
      tokens_per_expert,
      transB)
    
    ctx.save_for_backward(permuted_inputs, tokens_per_expert, weights)
    ctx.transB = transB

    return output


  @staticmethod
  def backward(ctx, permuted_inputs_grad):
      
    permuted_inputs, tokens_per_expert, weights = ctx.saved_tensors
    transB = ctx.transB

    permuted_inputs_grad = permuted_inputs_grad.contiguous()

    weight_grad = None
    if ctx.needs_input_grad[0]:
      weight_grad = torch.ops.moe_unit_ops.moe_group_gemm_backward_op(
        permuted_inputs,
        permuted_inputs_grad,
        tokens_per_expert,
        transB)

    activation_grad = None
    if ctx.needs_input_grad[1]:
      activation_grad = torch.ops.moe_unit_ops.moe_group_gemm_op(
        permuted_inputs_grad,
        weights,
        tokens_per_expert,
        not transB)

    return activation_grad, weight_grad, None, None

################################################################################################
##
## Ops Wrapper
##
################################################################################################

def permute(unpermuted_inputs, expert_for_rows, max_token_num):
  return PermuteMoE.apply(unpermuted_inputs, expert_for_rows, max_token_num)

def unpermute(permuted_inputs, expert_for_rows, source_row_to_dest_row, max_token_num):
  return UnpermuteMoE.apply(permuted_inputs, expert_for_rows, source_row_to_dest_row, max_token_num)

def groupedgemm(permuted_inputs, weights, tokens_per_expert, transB):
  return GroupedGemmMoE.apply(permuted_inputs, weights, tokens_per_expert, transB)

def sinkhorn_kernel(cost, tol=0.0001):
    return torch.ops.moe_unit_ops.sinkhorn(cost, tol)
