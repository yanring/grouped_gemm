from grouped_gemm import backend
import torch


class GroupedGemm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, batch_sizes, trans_b):
        assert torch.count_nonzero(batch_sizes) != 0, "Input batch_sizes should not be all zeros!"
        ctx.save_for_backward(a, b, batch_sizes)
        ctx.trans_b = trans_b
        return backend.gmm(a, b, batch_sizes, trans_a=False, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        a, b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b

        agrad = None
        if ctx.needs_input_grad[0]:
            agrad = backend.gmm(
                grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)

        bgrad = None
        if ctx.needs_input_grad[1]:
            lhs, rhs = (grad, a) if trans_b else (a, grad)
            bgrad = backend.gmm(
                lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
        return agrad, bgrad, None, None

class Permutation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, indices):
        """ Permute input tokens according to the indices map so that tokens with
        the same ids are grouped together.

        Args:
            input (2D matrix): activation input of shape [num_tokens, hidden_size].
            indices (1D vector, int64): expert indices of each token with shape of [num_tokens].

        Returns:
            permuted_output: 2D matrix, permuted activations.
            source_row_to_dest_row: row map between the input and permuted_output
        """
        permuted_outputs, source_row_to_dest_row = backend.permute(input, indices)
        ctx.save_for_backward(source_row_to_dest_row)
        return permuted_outputs, source_row_to_dest_row

    @staticmethod
    def backward(ctx, grad, _):
        source_row_to_dest_row, = ctx.saved_tensors
        return backend.unpermute(grad, source_row_to_dest_row), None

class Unpermutation(torch.autograd.Function):
    """Reverse process of `Permutation` class which unpermutes permuted input
    into the original order to produce the final output.
    """

    @staticmethod
    def forward(ctx, permuted_input, indices, source_row_to_dest_row):
        ctx.save_for_backward(indices)
        permuted_output = backend.unpermute(permuted_input, source_row_to_dest_row)

        return permuted_output

    @staticmethod
    def backward(ctx, grad):
        indices, = ctx.saved_tensors
        unperted_output, _ = backend.permute(grad, indices)
        import pdb
        # pdb.set_trace()
        return unperted_output, None, None

def gmm(a, b, batch_sizes, trans_b=False):
    return GroupedGemm.apply(a, b, batch_sizes, trans_b)

def sinkhorn_kernel(cost, tol=0.0001):
    return backend.sinkhorn(cost, tol)

def permute(input, indices):
    return Permutation.apply(input, indices)

def unpermute(input, indices, source_row_to_dest_row):
    return Unpermutation.apply(input, indices, source_row_to_dest_row)

