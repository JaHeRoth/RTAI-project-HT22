import argparse
import csv
import itertools
import re
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from torch import nn, Tensor, Size
from torch.nn import Linear, ReLU, Conv2d, BatchNorm2d, Sequential

from networks import get_network, get_net_name, NormalizedResnet
from resnet import BasicBlock

DEVICE = 'cpu'
DTYPE = torch.float32
DEBUG = False


def transform_image(pixel_values, input_dim):
    '''
    Transform the input image to the correct, normalized format.
    In the case of the cifar10 dataset, additionally permute the dimensions.
    '''
    normalized_pixel_values = torch.tensor([float(p) / 255.0 for p in pixel_values])
    if len(input_dim) > 1:
        input_dim_in_hwc = (input_dim[1], input_dim[2], input_dim[0])
        image_in_hwc = normalized_pixel_values.view(input_dim_in_hwc)
        image_in_chw = image_in_hwc.permute(2, 0, 1)
        image = image_in_chw
    else:
        image = normalized_pixel_values

    assert (image >= 0).all()
    assert (image <= 1).all()
    return image


def get_spec(spec, dataset):
    input_dim = [1, 28, 28] if dataset == 'mnist' else [3, 32, 32]
    eps = float(spec[:-4].split('/')[-1].split('_')[-1])
    test_file = open(spec, "r")
    test_instances = csv.reader(test_file, delimiter=",")
    for i, (label, *pixel_values) in enumerate(test_instances):
        inputs = transform_image(pixel_values, input_dim)
        inputs = inputs.to(DEVICE).to(dtype=DTYPE)
        true_label = int(label)
    inputs = inputs.unsqueeze(0)
    return inputs, true_label, eps


def get_net(net, net_name):
    net = get_network(DEVICE, net)
    state_dict = torch.load('../nets/%s' % net_name, map_location=torch.device(DEVICE))
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
    net.load_state_dict(state_dict)
    net = net.to(dtype=DTYPE)
    net.eval()
    if 'resnet' in net_name:
        net = NormalizedResnet(DEVICE, net)
    return net


Bound = Tuple[Tensor, Tensor, Tensor, Tensor]
Bounds = List[Union[Bound, Dict[str, List[Bound]]]]
Alpha = Union[str, Dict[Union[str, int], Tensor]]


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
    direct_lb, direct_ub = backtrack(abstract_lb, abstract_ub, past_bounds)
    input_lb, input_ub = input_lb.reshape(-1, 1), input_ub.reshape(-1, 1)
    concrete_lb = cased_mul_w_bias(direct_lb, input_lb, input_ub).flatten()
    concrete_ub = cased_mul_w_bias(direct_ub, input_ub, input_lb).flatten()
    return concrete_lb, concrete_ub


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


def affine_bounds(bias: Tensor, coefficients: Tensor, past_bounds: Bounds, input_lb: Tensor, input_ub: Tensor):
    """:return: Abstract and concrete upper and lower bounds of an affine layer given
    its intercept (`bias`) and linear coefficient (`coefficients`) on the layer directly
    before it, the abstract and concrete lower and upper bounds of all layers before it
    and the concrete upper and lower bounds of the region of input data that should be
    verified for this network."""
    abstract_lower = abstract_upper = torch.hstack([bias.reshape(-1, 1), coefficients])
    concrete_lower, concrete_upper = concretize_bounds(abstract_lower, abstract_upper, past_bounds, input_lb, input_ub)
    return abstract_lower, abstract_upper, concrete_lower, concrete_upper


def fc_bounds(layer: Linear, past_bounds: Bounds, input_lb: Tensor, input_ub: Tensor):
    """:return: Abstract and concrete upper and lower bounds of the fully connected `layer`."""
    bias = layer.bias.detach() if layer.bias is not None else torch.zeros(layer.out_features)
    return affine_bounds(bias, layer.weight.detach(), past_bounds, input_lb, input_ub)


def conv_bounds(layer: Conv2d, past_bounds: Bounds, input_lb: Tensor, input_ub: Tensor, in_height: int, in_width: int, bn_layer: Optional[BatchNorm2d]):
    """:return: Abstract and concrete upper and lower bounds of the convolutional layer `layer`
    and if applicable the batch normalization layer `bn_layer` directly following it, uppon
    rewriting (and thus treating) both of these as a single fully connected layer."""
    intercept, coefficients = conv_to_affine(layer, in_height, in_width, bn_layer)
    return affine_bounds(intercept, coefficients, past_bounds, input_lb, input_ub)


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
        min_scale = 0.75
        random_scales = min_scale + torch.rand(in_lb.shape) * (1 - min_scale)
        return (in_ub > -in_lb) * random_scales
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


def res_bounds(layer: BasicBlock, bounds: Bounds, input_lb: Tensor, input_ub: Tensor, k: int, alpha: Alpha, in_shape: Size):
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
    _, out_a_alphas, a_bounds = deep_poly(layer.path_a, a_alphas, input_lb, input_ub, bounds, in_shape)
    _, out_b_alphas, b_bounds = deep_poly(layer.path_b, b_alphas, input_lb, input_ub, bounds, in_shape)
    block_bounds = {"a": a_bounds, "b": b_bounds}
    block_alphas = {**{f"{k}a{key}": value for key, value in out_a_alphas.items()},
                    **{f"{k}b{key}": value for key, value in out_b_alphas.items()}}
    out_len = len(b_bounds[-1][-1])
    identity = torch.hstack([torch.zeros(out_len, 1), torch.eye(out_len)])
    bounds_with_block = [*bounds, block_bounds]
    concrete_lb, concrete_ub = concretize_bounds(identity, identity, bounds_with_block, input_lb, input_ub)
    block_bounds["lb"], block_bounds["ub"] = concrete_lb, concrete_ub
    return block_bounds, block_alphas


def deep_poly(layers: Sequential, alpha: Alpha, src_lb: Tensor, src_ub: Tensor, in_bounds=None, in_shape=None):
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
    in_shapes = infer_layer_input_shapes(layers, src_lb if in_shape is None else torch.zeros(in_shape))
    src_lb, src_ub = src_lb.flatten(), src_ub.flatten()
    bounds: Bounds = in_bounds.copy() if in_bounds is not None else []
    out_alpha: Dict[Union[str, int], Tensor] = {}
    for k, layer in enumerate(layers):
        if type(layer) == Linear:
            bounds.append(fc_bounds(layer, bounds, src_lb, src_ub))
        elif type(layer) == Conv2d:
            in_height, in_width = in_shapes[k][-2:]
            bn_layer = layers[k+1] if len(layers) > k + 1 and type(layers[k+1]) == BatchNorm2d else None
            bounds.append(conv_bounds(layer, bounds, src_lb, src_ub, in_height, in_width, bn_layer))
        elif type(layer) == ReLU:
            bound, out_alpha[k] = relu_bounds(bounds, alpha[k] if type(alpha) == dict else alpha)
            bounds.append(bound)
        elif type(layer) == BasicBlock:
            block_bounds, block_alphas = res_bounds(layer, bounds, src_lb, src_ub, k, alpha, in_shapes[k])
            out_alpha.update(block_alphas)
            bounds.append(block_bounds)
    output_ub = bounds[-1][3]
    added_bounds = bounds if in_bounds is None else bounds[len(in_bounds):]
    return output_ub, out_alpha, added_bounds


def ensemble_poly(net_layers: Sequential, input_lb: Tensor, input_ub: Tensor, true_label: int):
    """
    Optimize multiple combinations of alphas simultaenously to allow DeepPoly to rule out all
    other categories than `true_label`, ruling out a category for good once any alpha value does so.

    :return: Whether we found for each category that wasn't `true_label` an alpha that ruled it out
    for all inputs (to `net_layers`) between `input_lb` and `input_ub`.
    """
    start_time = datetime.now()
    # Keep track of which categories we still need to rule out
    remaining_labels = Tensor([c for c in range(net_layers[-1].out_features) if c != true_label]).long()
    # Add an additional comparison layer to compare the ground truth label against all not yet proven labels
    # This can then be used to optimize the alphas for all remaining labels at once
    layers = with_comparison_layer(net_layers, true_label, adversarial_labels=remaining_labels)
    # The strategy with which we initialize the ensemble of our alphas
    alphas = ["min", "noisymin", "noisymin"]
    # The upper bounds we achieve with the three different strategies
    out_ubs: List[Optional[Tensor]] = [None for _ in alphas]

    # TODO: Arbitrary hyperparameters, further tuning is needed
    # Except when debugging there is no point in giving up early, since printing
    # "not verified" gives 0 points, just like timing out does
    max_iter = 10 if DEBUG else 10**9
    evolution_period = 5 if DEBUG else 10
    learning_rate = 10**0
    for epoch in range(max_iter):
        for i, (old_ub, alpha) in enumerate(zip(out_ubs, alphas)):
            # After the first epoch, we initialized the alphas for each strategy and do a Gradient Descent step
            if type(alpha) is not str:
                # TODO?: According to documentation, gradients do get aggregated over time, do we need to null them here for each epoch? 
                # Meaning alpha[k].grad.zero_() for k in alpha.keys()?
                # Or is this solved by our design that we always detach the alphas and then build a new graph in the next forward pass? Would investigate.

                # We always optimize/differentiate w.r.t. the first unbeaten competitive class comparison
                # The resulting gradient is stored in the alpha values, which are the leafes of the computational graph
                # TODO?: Just for myself, check that the index here does not need to be updated, i.e. that the first unbeaten class is always at index 0 because
                # we always remove the beaten classes. Should happen through changing the comparison layer and the execution of deep_poly, but just to be sure.
                old_ub[0].backward()
                alpha = {k: (ten - learning_rate * ten.grad).clamp(0, 1).detach().requires_grad_() for k, ten in alpha.items()}
            out_ub, out_alpha, _ = deep_poly(layers, alpha, input_lb, input_ub)
            # TODO: Are we satisfied with a tie? Or do we need to be strictly better? Could be an (unlikely) error source
            remaining_labels = remaining_labels[out_ub > 0]
            if len(remaining_labels) == 0:
                # We ruled out all other categories, so we're done
                dprint(f"Verified after {(datetime.now()-start_time).total_seconds()} seconds. "
                       f"[epoch: {epoch}; i: {i}; alpha: {alpha})]")
                return True
            # TODO: Again maybe needs a change if we change the tie-breaking strategy
            if out_ub.min() <= 0:
                # We found an alpha that ruled out at least one category, so we update the comparison layer
                dprint(f"{len(remaining_labels)} categories left to beat: {remaining_labels}.")
                # TODO?: Dumb idea, but is there anything in the comp graph of layers that we lose when we do this override?
                # Or is everything solved by the fact, that our next run of deep_poly will build a new comp graph?
                # TODO?: Also, maybe another stupid thought, but does this even help computation in any way? Since we'll optimize
                # the alphas only regarding the first to-be-beaten category anyways, so the edges that fall away here don't contribute anyways?
                # Of course it's very neat with the current implementation, but I don't know how big the overhead of rebuilding the network is.
                layers = with_comparison_layer(net_layers, true_label, adversarial_labels=remaining_labels)
            out_ubs[i], alphas[i] = out_ub, out_alpha
        dprint(f"out_ubs after epoch {epoch}: {[out_ub.detach().numpy() for out_ub in out_ubs]}")
        # Do a little bit of evolution, i.e. mutate the worst performing alpha after some epochs
        if epoch % evolution_period == 0 and epoch > 0:
            total_losses = Tensor([ReLU()(out_ub).sum() for out_ub in out_ubs])
            alphas[torch.argmax(total_losses)] = "noisymin"
    dprint(f"Failed to verify after {(datetime.now()-start_time).total_seconds()} seconds and {max_iter} epochs.")
    return False


def with_comparison_layer(net_layers: Sequential, true_label: int, adversarial_labels: Tensor):
    """
    :return: `net_layers` with an extra layer at the end that for each category in
    `adversarial_labels` has a node whose value is the output/probability for that category 
    minus the output/probability for the `true_label` category.

    Thus, when an output node is <= 0, that means the corresponding category is beaten.
    """
    num_categories = net_layers[-1].out_features
    weight = torch.eye(num_categories)[adversarial_labels, :]
    weight[:, true_label] = -1
    comparison_layer = Linear(in_features=num_categories, out_features=len(adversarial_labels), bias=False)
    comparison_layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    return Sequential(*net_layers, comparison_layer)


def print_if(msg: str, condition: bool):
    """Print `msg` if `condition` is True."""
    if condition:
        print(msg)


def dprint(msg: str):
    """Print `msg` if `DEBUG` is True."""
    print_if(msg, DEBUG)


def set_seed(seed: int):
    """Derandomize PyTorch and Numpy (between runs)."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def analyze(net, inputs, eps, true_label):
    set_seed(0)
    # We extract the normalization layer to work with the already normalized inputs, since our epsilons are also in the normalized space
    # TODO: Some networks also have a flattening layer which seems irrelevant for our purposes, could we extract that too? Possible performance gain?
    if type(net) == NormalizedResnet:
        normalizer = net.normalization
        # Flatten the nested ResNet by unfolding Sequential layers
        layers = Sequential(*itertools.chain.from_iterable([(layer if type(layer) is Sequential else [layer]) for layer in net.resnet]))
    else:
        normalizer = net.layers[0]
        layers = net.layers[1:]
    # We only have to account for valid input pixels, so we clamp back to [0, 1]
    input_lb, input_ub = (inputs - eps).clamp(0, 1), (inputs + eps).clamp(0, 1)
    normalized_lb, normalized_ub = normalizer(input_lb), normalizer(input_ub)
    # Ensure we crash (which counts as not verified) if gradients are nan or inf
    with torch.autograd.set_detect_anomaly(True):
        return ensemble_poly(layers, normalized_lb, normalized_ub, true_label)


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net', type=str, required=True, help='Neural network architecture to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    net_name = get_net_name(args.net)
    dataset = 'mnist' if 'mnist' in net_name else 'cifar10'
    
    inputs, true_label, eps = get_spec(args.spec, dataset)
    net = get_net(args.net, net_name)

    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    # We only verify the network on examples that it can correctly classify
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
