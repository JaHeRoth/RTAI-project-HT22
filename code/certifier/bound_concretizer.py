import torch
from torch import Tensor

from .constants import Bounds


def cased_mul_w_bias(lhs_with_bias_column: Tensor, pos_rhs: Tensor, neg_rhs: Tensor):
    """:return: A sort of affine product between `lhs_with_bias_column` and `pos_rhs`
    and `neg_rhs`, where which of the latter is used for a specific scalar multiplication
    depends on the sign of the cell in `lhs_with_bias_column` which is involved.
    More specifically, cell [i, j] of output is `lhs_with_bias_column[i, 0]` plus the sum
    (over k) of the elementwise product between `lhs_with_bias_column[i, k+1]` and
    `pos_rhs[k, j]` if `lhs_with_bias_column[i, k] >= 0` else `neg_rhs[k, j]`."""
    bias_to_bias_row = torch.hstack([torch.ones(1, 1), torch.zeros(1, pos_rhs.shape[1] - 1)])
    pos_rhs_with_b2b_row = torch.vstack([bias_to_bias_row, pos_rhs])
    neg_rhs_with_b2b_row = torch.vstack([bias_to_bias_row, neg_rhs])
    return (lhs_with_bias_column * (lhs_with_bias_column >= 0) @ pos_rhs_with_b2b_row
            + lhs_with_bias_column * (lhs_with_bias_column < 0) @ neg_rhs_with_b2b_row)


def backtrack(direct_lb: Tensor, direct_ub: Tensor, past_bounds: Bounds):
    """
    Iteratively compute the affine coefficients used to formulate abstract bounds of
    some target-layer in terms of the input to the (sub)network.
    :param direct_lb: The abstract lower bound (affine coefficients) of the target
    layer in terms of our last layer (the one generating `past_bounds[-1]`), thus
    our starting point for computing the abstract lower bound of the target layer
    on the (sub)network input (layer).
    :param direct_ub: Same as `direct_lb`, but for upper bound.
    :param past_bounds: The abstract and concrete upper and lower bounds of all
    layers we should backtrack through (to update `direct_lb` and `direct_ub`).
    Note that these layers can be BasicBlocks, which require recursing.
    :return: Abstract upper and lower bounds of some layer in terms of some earlier
    layer. This earlier layer is either the input to the network or the input to the
    BasicBlock (subnetwork), rather than the last layer before the later layer.
    More specifically, two matrices, one for the lower bounds and one for the upper bounds,
    where the cell [i, j+1] gives the coefficient of the j-th node of the earlier layer
    used in the corresponding abstract bound of node i of the later layer while [i, 0]
    gives its intercept.
    """
    # Possible optimization: concretize every n layers to see if concrete lower is above 0 or concrete upper is below 0
    for past_bound in reversed(past_bounds):
        if type(past_bound) is tuple:
            abstract_lb, abstract_ub = past_bound[:2]
            direct_lb = cased_mul_w_bias(direct_lb, abstract_lb, abstract_ub)
            direct_ub = cased_mul_w_bias(direct_ub, abstract_ub, abstract_lb)
        else:
            # TODO: Ofc sanity check this (especially that backtracks to layer I'd expect)
            direct_a_lb, _ = backtrack(direct_lb, direct_lb, past_bound["a"])
            _, direct_a_ub = backtrack(direct_ub, direct_ub, past_bound["a"])
            direct_b_lb, _ = backtrack(direct_lb, direct_lb, past_bound["b"])
            _, direct_b_ub = backtrack(direct_ub, direct_ub, past_bound["b"])
            direct_lb, direct_ub = direct_a_lb + direct_b_lb, direct_a_ub + direct_b_ub
    return direct_lb, direct_ub


def concretize_bounds(abstract_lb: Tensor, abstract_ub: Tensor, past_bounds: Bounds, input_lb: Tensor, input_ub: Tensor):
    """:return: Concrete upper and lower bounds of a layer given its abstract lower (`abstract_lb`)
    and upper (`abstract_ub`) on the layer directly before it, the abstract and concrete upper and
    lower bounds of all layers before it and the concrete bounds of the network's input region we
    seek to verify."""
    # input_lb & input_ub are still src_lb & src_ub, so the original input region
    direct_lb, direct_ub = backtrack(abstract_lb, abstract_ub, past_bounds)
    input_lb, input_ub = input_lb.reshape(-1, 1), input_ub.reshape(-1, 1)
    concrete_lb = cased_mul_w_bias(direct_lb, input_lb, input_ub).flatten()
    concrete_ub = cased_mul_w_bias(direct_ub, input_ub, input_lb).flatten()
    return concrete_lb, concrete_ub
