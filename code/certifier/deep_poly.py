import re
from contextlib import nullcontext
from datetime import datetime
from itertools import product
from typing import Optional, Union, List, Dict

import torch
from torch import Tensor, Size
from torch.nn import Conv2d, BatchNorm2d, Linear, Sequential, ReLU

from .cache import ConvToAffineCache
from .networks.resnet import BasicBlock
from .logger import dprint
from .bound_concretizer import concretize_bounds
from .constants import Bounds, Alpha


def conv_to_affine(layer: Conv2d, in_height: int, in_width: int, bn_layer: BatchNorm2d = None):
    """
    :param layer: Convolutional layer.
    :param in_height: Height of the images passed to this layer (from the layer before it).
    :param in_width: Width of the images passed to this layer (from the layer before it).
    :param bn_layer: Batch normalization layer directly following convolutional layer (if applicable).
    :return: Coefficients such that an inner product between this and flattened input (prepended by 1 for bias)
    gives same result as flattening result of applying convolution layer (and possibly bn_layer) on input.
    """
    hpadding, wpadding = layer.padding
    hstride, wstride = layer.stride
    num_filters, depth, filter_height, filter_width = layer.weight.shape
    padded_height, padded_width = in_height + 2 * hpadding, in_width + 2 * wpadding
    num_hsteps, num_wsteps = 1 + (padded_height - filter_height) // hstride, 1 + (padded_width - filter_width) // wstride
    linear_coefficients_tensor = torch.empty(num_filters, num_hsteps, num_wsteps, depth, in_height, in_width)
    for f, r, c in product(range(num_filters), range(num_hsteps), range(num_wsteps)):
        cell_coefficients = torch.zeros((depth, padded_height, padded_width))
        start_row, start_column = r * hstride, c * wstride
        cell_coefficients[:, start_row : start_row + filter_height, start_column : start_column + filter_width] = layer.weight[f]
        if hpadding > 0:
            cell_coefficients = cell_coefficients[:, hpadding:-hpadding, :]
        if wpadding > 0:
            cell_coefficients = cell_coefficients[:, :, wpadding:-wpadding]
        linear_coefficients_tensor[f, r, c] = cell_coefficients
    filter_intercept = torch.zeros(num_filters) if layer.bias is None else layer.bias
    if bn_layer is not None:
        bn_bias, bn_weight, bn_mean, bn_var, bn_eps = bn_layer.bias.detach(), bn_layer.weight.detach(), bn_layer.running_mean.detach(), bn_layer.running_var.detach(), bn_layer.eps
        filter_intercept += bn_bias - bn_weight * bn_mean / torch.sqrt(bn_var + bn_eps)
        linear_coefficients_tensor *= (bn_weight / torch.sqrt(bn_var + bn_eps)).reshape(-1, 1, 1, 1, 1, 1)
    intercept = filter_intercept.repeat_interleave(num_hsteps * num_wsteps).reshape(-1, 1)
    linear_coefficients = linear_coefficients_tensor.reshape((num_filters * num_hsteps * num_wsteps, -1))
    return intercept.detach(), linear_coefficients.detach()


def affine_bounds(bias: Tensor, coefficients: Tensor, past_bounds: Bounds,
                  input_lb: Tensor, input_ub: Tensor, should_concretize: bool):
    """:return: Abstract and concrete upper and lower bounds of an affine layer given
    its intercept (`bias`) and linear coefficient (`coefficients`) on the layer directly
    before it, the abstract and concrete lower and upper bounds of all layers before it
    and the concrete upper and lower bounds of the region of input data that should be
    verified for this network."""
    # input_lb & input_ub are just the src_lb & src_ub of the first layer in the network
    # Concatinates the bias of a node with the weights of the edges coming into it
    abstract_lower = abstract_upper = torch.hstack([bias.reshape(-1, 1), coefficients])
    concrete_lower, concrete_upper = concretize_bounds(
        abstract_lower, abstract_upper, past_bounds, input_lb, input_ub) if should_concretize else (None, None)
    return abstract_lower, abstract_upper, concrete_lower, concrete_upper


def fc_bounds(layer: Linear, past_bounds: Bounds, input_lb: Tensor, input_ub: Tensor, should_concretize: bool):
    """:return: Abstract and concrete upper and lower bounds of the fully connected `layer`."""
    # input_lb & input_ub are just the src_lb & src_ub of the first layer in the network
    # Extract the bias and weights of the layer and hand them off to affine_bounds
    bias = layer.bias.detach() if layer.bias is not None else torch.zeros(layer.out_features)
    weights = layer.weight.detach()
    return affine_bounds(bias, weights, past_bounds, input_lb, input_ub, should_concretize)


def caching_conv_to_affine(layer: Conv2d, in_height: int, in_width: int,
                           bn_layer: Optional[BatchNorm2d], c2a_cache: ConvToAffineCache):
    if c2a_cache.get(layer, in_height, in_width, bn_layer) is None:
        c2a_cache.set(layer, in_height, in_width, bn_layer, conv_to_affine(layer, in_height, in_width, bn_layer))
    return c2a_cache.get(layer, in_height, in_width, bn_layer)


def conv_bounds(layer: Conv2d, past_bounds: Bounds, input_lb: Tensor, input_ub: Tensor, in_height: int, in_width: int,
                bn_layer: Optional[BatchNorm2d], c2a_cache: ConvToAffineCache, should_concretize: bool):
    """:return: Abstract and concrete upper and lower bounds of the convolutional layer `layer`
    and if applicable the batch normalization layer `bn_layer` directly following it, upon
    rewriting (and thus treating) both of these as a single fully connected layer."""
    st = datetime.now()
    intercept, coefficients = caching_conv_to_affine(layer, in_height, in_width, bn_layer, c2a_cache)
    dprint(f"Spent {(datetime.now()-st).total_seconds()} seconds on caching_conv_to_affine.")
    return affine_bounds(intercept, coefficients, past_bounds, input_lb, input_ub, should_concretize)


def generate_alpha(in_lb: Tensor, in_ub: Tensor, strategy: str):
    """:return: A tensor with the same shape as the input to the current ReLU layer,
    with all its values in [0,1] generated according to the provided `strategy`."""
    if strategy == "half":
        return torch.ones(in_lb.shape) / 2
    elif strategy == "rand":
        return torch.rand(in_lb.shape)
    elif strategy == "min":
        return (in_ub > -in_lb).float()
    elif strategy == "noisymin":
        # Like min strategy, but deviations from alpha=0.5 scaled by uniformly random numbers
        min_scale = 0.5
        random_scales = min_scale + torch.rand(in_lb.shape) * (1 - min_scale)
        return 1 / 2 + ((in_ub > -in_lb).float() - 1 / 2) * random_scales
    elif strategy == "smoothmin":
        # Like min, but with a smooth transition between 0 and 1, to reflect difference in area
        # being small when in_ub and -in_lb are almost the same
        return 1/2 + torch.atan(in_ub + in_lb) / torch.pi
    raise ValueError(f"{strategy} is an invalid alpha-generating strategy.")


def relu_bounds(past_bounds: Bounds, alpha: Union[Tensor, str]):
    """:return: Abstract (affine coefficients on the layer directly before it) and concrete
    upper and lower bounds of ReLU layer given its `alpha` and the abstract and concrete
    upper and lower bounds of all layers before it."""
    if type(past_bounds[-1]) is dict:
        prev_lb, prev_ub = past_bounds[-1]["lb"], past_bounds[-1]["ub"]
    else:
        prev_lb, prev_ub = past_bounds[-1][-2:]
    if type(alpha) == str:
        alpha = generate_alpha(prev_lb, prev_ub, strategy=alpha).requires_grad_()
    in_len = len(prev_lb)
    lb_bias, ub_bias, lb_scaling, ub_scaling = [torch.zeros(in_len) for _ in range(4)]
    lb_scaling[prev_lb >= 0], ub_scaling[prev_lb >= 0] = 1, 1

    # Upper and lower bounds of crossing ReLUs, secured against numeric
    # underflow in the alpha gradients through strategic detaching
    crossing_mask = (prev_lb < 0) & (prev_ub > 0)
    lb_scaling[crossing_mask] = alpha[crossing_mask]
    numerically_unstable = prev_lb.abs() < 10 ** -5
    stable_crossing = crossing_mask & ~numerically_unstable
    unstable_crossing = crossing_mask & numerically_unstable
    # I'm unsure why, but calling detach_() after these lines on the unstable
    # indices didn't hinder the backpropagation to use them for computing gradients
    # of past alphas, thus didn't help avoid the nan alpha gradients.
    # A possible explanation is that the computation graph creates shortcuts that bypass
    # these nodes, as they aren't leaf nodes anyway, thus that when we detach them we're
    # not actually cutting the connection between past alphas and ouptut caused by these
    # computations: https://youtu.be/MswxJw-8PvE?t=224
    if unstable_crossing.any():
        dprint(f"Encountered {unstable_crossing.sum()} dangerously small term(s) "
               f"(<e-5) in relu_bounds, thus detaching this/these to avoid under-/overflows.")
    stable_lb = prev_lb[stable_crossing]
    stable_ub = prev_ub[stable_crossing]
    unstable_lb = prev_lb[unstable_crossing].detach()
    unstable_ub = prev_ub[unstable_crossing].detach()
    upper_slope = torch.zeros(prev_lb.shape)
    upper_slope[stable_crossing] = (stable_ub / (stable_ub - stable_lb))
    ub_scaling[stable_crossing] = upper_slope[stable_crossing]
    ub_bias[stable_crossing] = (-upper_slope[stable_crossing] * stable_lb)
    upper_slope[unstable_crossing] = (unstable_ub / (unstable_ub - unstable_lb))
    ub_scaling[unstable_crossing] = upper_slope[unstable_crossing]
    ub_bias[unstable_crossing] = (-upper_slope[unstable_crossing] * unstable_lb)

    abstract_lb = torch.hstack([lb_bias.reshape(-1, 1), lb_scaling.diag()])
    abstract_ub = torch.hstack([ub_bias.reshape(-1, 1), ub_scaling.diag()])
    concrete_lb = abstract_lb[:, 0] + abstract_lb[:, 1:] @ prev_lb
    concrete_ub = abstract_ub[:, 0] + abstract_ub[:, 1:] @ prev_ub
    return (abstract_lb, abstract_ub, concrete_lb, concrete_ub), alpha


def infer_layer_input_shapes(layers: Sequential, input_lb: Tensor):
    """:return: A list of Size objects (shape tuples) that has at index i
    the shape of the input `layers[i]` will receive."""
    current = input_lb
    dims: List[Size] = []
    for layer in layers:
        dims.append(current.shape)
        current = layer(current)
    return dims


def extract_path_alphas(alpha: Alpha, block_layer_number: int, path_name: str):
    """Expand virtual key hierarchy of `alpha` (if it is a dict, else just return it as is)
    for the provided layer number and path name.
    :return: `alpha` if it is a string. Otherwise (if it is a dict), the entries belonging
    to path `path_name` of layer number `block_layer_number` with those localizations removed
    from the corresponding keys (leaving only the index of that sublayer within this path)."""
    if type(alpha) == str:
        return alpha
    return {int(re.findall(r'\d+', key)[-1]): value for key, value in alpha.items()
            if type(key) is str and key.startswith(f"{block_layer_number}{path_name}")}


def res_bounds(layer: BasicBlock, bounds: Bounds, input_lb: Tensor, input_ub: Tensor, k: int,
               alpha: Alpha, in_shape: Size, c2a_cache: ConvToAffineCache):
    """
    Call deep_poly for both paths of the BasicBlock `layer`, combining the resulting abstract bounds and
    computing the concrete output bounds using that combination.
    :param layer: The BasicBlock.
    :param bounds: The abstract and concrete bounds of all layers before the current.
    :param input_lb: The concrete lower bound of the input region to the full network we're verifying.
    :param input_ub: Same as `input_lb`, but upper instead of lower.
    :param k: The index of `layer`.
    :param alpha: See the docstring of `deep_poly(...)`.
    :param in_shape: The shape of the input `layer` will receive.
    :return: Dict containing bounds for all layers of both paths of the BasicBlock `layer` and the
    concrete upper and lower bound of the output of `layer`; the numerical alpha values used by the layers
    in this block, with key corresponding to the index of `layer`, the path they belong to, and
    the index of the corresponding layer in that path (so e.g. "2b1" for a layer at index 1 of path b
    if `layer` has index 2).
    """
    # TODO: Bugfix: concrete bounds blow up in the conv layers of path_b of the last basic block (k=8) of net10
    a_alphas, b_alphas = extract_path_alphas(alpha, k, "a"), extract_path_alphas(alpha, k, "b")
    _, out_a_alphas, a_bounds = deep_poly(layer.path_a, a_alphas, input_lb, input_ub, c2a_cache, bounds, in_shape)
    _, out_b_alphas, b_bounds = deep_poly(layer.path_b, b_alphas, input_lb, input_ub, c2a_cache, bounds, in_shape)
    block_bounds = {"a": a_bounds, "b": b_bounds}
    block_alphas = {**{f"{k}a{key}": value for key, value in out_a_alphas.items()},
                    **{f"{k}b{key}": value for key, value in out_b_alphas.items()}}
    out_len = len(b_bounds[-1][0])
    identity = torch.hstack([torch.zeros(out_len, 1), torch.eye(out_len)])
    bounds_with_block = [*bounds, block_bounds]
    with torch.no_grad():
        concrete_lb, concrete_ub = concretize_bounds(identity, identity, bounds_with_block, input_lb, input_ub)
    block_bounds["lb"], block_bounds["ub"] = concrete_lb, concrete_ub
    return block_bounds, block_alphas


def deep_poly(layers: Sequential, alpha: Alpha, src_lb: Tensor, src_ub: Tensor, c2a_cache: ConvToAffineCache, in_bounds=None, in_shape=None):
    """
    :param layers: The sequential network (list of layers) we intend to verify.
    :param alpha: The alpha values to use for all ReLU nodes (more specifically: a dictionary mapping
    layer indeces to 1d-tensors of alpha values for all nodes in the corresponding ReLU layer) or a
    string dictating the strategy with which to generate such numerical alphas for all ReLU nodes
    (for the first forward pass of a network).
    :param src_lb: The concrete lower bound of the region in which we wish to verify our full network
    (which is a strict superset of `layers` iff `in_bounds!=None`). Used for computing all concrete bounds.
    :param src_ub: Same as `src_lb`, just the upper instead of lower.
    :param in_bounds: Relevant in the case of deep_poly being called recursively (thus only for ResNets),
    containing the abstract and concrete bounds of all layers that came before the Sequential we're now
    running DeepPoly on.
    :param in_shape: The shape of the input to the first layer in `layers`. This equals `src_lb.shape`
    if `in_bounds=None`, so it has to be set iff `in_bounds!=None`.
    :return: Concrete upper bounds for the final layer in `layers`; the numerical alpha values used
    in this run (equals `alpha` if that already contained the numerical values, rather than just a
    strategy string); the abstract and concrete bounds of all layers in `layers`.
    """
    st = datetime.now()
    is_nested = in_bounds is not None
    in_shapes = infer_layer_input_shapes(layers, torch.zeros(in_shape) if is_nested else src_lb)
    dprint(f"Spent {(datetime.now()-st).total_seconds()} seconds on infer_layer_input_shapes.")
    st = datetime.now()
    src_lb, src_ub = src_lb.flatten(), src_ub.flatten()
    dprint(f"Spent {(datetime.now() - st).total_seconds()} seconds flattening concrete input bounds.")
    bounds: Bounds = in_bounds.copy() if is_nested else []
    out_alpha: Dict[Union[str, int], Tensor] = {}
    outer_st = datetime.now()
    for k, layer in enumerate(layers):
        st = datetime.now()
        if type(layer) == Linear:
            # Two last layers of our network are always linear, so concrete bounds of first of these aren't used
            # BasicBlocks have no Linear layers, so can assume is_nested=False
            should_concretize = k != len(layers) - 2
            needs_grad = k == len(layers) - 1
            with torch.no_grad() if not needs_grad else nullcontext():
                bounds.append(fc_bounds(layer, bounds, src_lb, src_ub, should_concretize))
        elif type(layer) == Conv2d:
            in_height, in_width = in_shapes[k][-2:]
            # Consecutive pair of Conv, BatchNorm layers is treated as a single affine layer
            bn_layer = layers[k+1] if len(layers) > k + 1 and type(layers[k+1]) == BatchNorm2d else None
            # Last layer of each path in BasicBlock is followed by an aggregation layer between these paths,
            # thus its concrete bounds aren't used, thus shouldn't be computed (to save time)
            should_concretize = not (is_nested and (
                    (bn_layer is None and k == len(layers) - 1) or (bn_layer is not None and k == len(layers) - 2)))
            with torch.no_grad():
                bounds.append(conv_bounds(
                    layer, bounds, src_lb, src_ub, in_height, in_width, bn_layer, c2a_cache, should_concretize))
        elif type(layer) == ReLU:
            bound, out_alpha[k] = relu_bounds(bounds, alpha[k] if type(alpha) == dict else alpha)
            bounds.append(bound)
        elif type(layer) == BasicBlock:
            block_bounds, block_alphas = res_bounds(layer, bounds, src_lb, src_ub, k, alpha, in_shapes[k], c2a_cache)
            out_alpha.update(block_alphas)
            bounds.append(block_bounds)
        dprint(f"Layer {k} of type {type(layer)} took {(datetime.now()-st).total_seconds()} seconds.")
    dprint(f"Spent {(datetime.now() - outer_st).total_seconds()} seconds passing through all layers.")
    output_ub = bounds[-1][3]
    # Return all the newly added abstract and concrete bounds as well
    added_bounds = bounds if in_bounds is None else bounds[len(in_bounds):]
    return output_ub, out_alpha, added_bounds
