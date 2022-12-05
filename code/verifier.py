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
    bias_to_bias_row = torch.hstack([torch.ones(1, 1), torch.zeros(1, pos_rhs.shape[1] - 1)])
    pos_rhs_with_b2b_row = torch.vstack([bias_to_bias_row, pos_rhs])
    neg_rhs_with_b2b_row = torch.vstack([bias_to_bias_row, neg_rhs])
    return (lhs_with_bias_column * (lhs_with_bias_column >= 0) @ pos_rhs_with_b2b_row
            + lhs_with_bias_column * (lhs_with_bias_column < 0) @ neg_rhs_with_b2b_row)


def backtrack(direct_lb: Tensor, direct_ub: Tensor, past_bounds: Bounds):
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
    direct_lb, direct_ub = backtrack(abstract_lb, abstract_ub, past_bounds)
    input_lb, input_ub = input_lb.reshape(-1, 1), input_ub.reshape(-1, 1)
    concrete_lb = cased_mul_w_bias(direct_lb, input_lb, input_ub).flatten()
    concrete_ub = cased_mul_w_bias(direct_ub, input_ub, input_lb).flatten()
    return concrete_lb, concrete_ub


def conv_to_affine(layer: Conv2d, in_height: int, in_width: int, bn_layer: BatchNorm2d = None):
    """:return Coefficients such that an inner product between this and flattened input (prepended by 1 for bias)
    gives same result as flattening result of applying convolution layer (and possibly bn_layer) on input."""
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
    abstract_lower = abstract_upper = torch.hstack([bias.reshape(-1, 1), coefficients])
    concrete_lower, concrete_upper = concretize_bounds(abstract_lower, abstract_upper, past_bounds, input_lb, input_ub)
    return abstract_lower, abstract_upper, concrete_lower, concrete_upper


def fc_bounds(layer: Linear, past_bounds: Bounds, input_lb: Tensor, input_ub: Tensor):
    bias = layer.bias.detach() if layer.bias is not None else torch.zeros(layer.out_features)
    return affine_bounds(bias, layer.weight.detach(), past_bounds, input_lb, input_ub)


def conv_bounds(layer: Conv2d, past_bounds: Bounds, input_lb: Tensor, input_ub: Tensor, in_height: int, in_width: int, bn_layer: Optional[BatchNorm2d]):
    intercept, coefficients = conv_to_affine(layer, in_height, in_width, bn_layer)
    return affine_bounds(intercept, coefficients, past_bounds, input_lb, input_ub)


def generate_alpha(in_lb: Tensor, in_ub: Tensor, strategy: str):
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


def infer_layer_input_dimensions(layers: Sequential, input_lb: Tensor):
    current = input_lb
    dims: List[Size] = []
    for layer in layers:
        dims.append(current.shape)
        current = layer(current)
    return dims


def extract_path_alphas(alpha: Alpha, block_layer_number: int, path_name: str):
    if type(alpha) == str:
        return alpha
    return {int(re.findall(r'\d+', key)[-1]): value for key, value in alpha.items()
            if type(key) is str and key.startswith(f"{block_layer_number}{path_name}")}


def res_bounds(layer: BasicBlock, bounds: Bounds, input_lb: Tensor, input_ub: Tensor, k: int, alpha: Alpha, in_shape: Size):
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
    in_shapes = infer_layer_input_dimensions(layers, src_lb if in_shape is None else torch.zeros(in_shape))
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
    start_time = datetime.now()
    remaining_labels = Tensor([c for c in range(net_layers[-1].out_features) if c != true_label]).long()
    layers = with_comparison_layer(net_layers, true_label, adversarial_labels=remaining_labels)
    alphas = ["min", "noisymin", "noisymin"]
    out_ubs: List[Optional[Tensor]] = [None for _ in alphas]

    # TODO: Arbitrary hyperparameters, further tuning is needed
    # Except when debugging there is no point in giving up early, since printing
    # "not verified" gives 0 points, just like timing out does
    max_iter = 10 if DEBUG else 10**9
    evolution_period = 5 if DEBUG else 10
    lr = 10**0
    for epoch in range(max_iter):
        for i, (old_ub, alpha) in enumerate(zip(out_ubs, alphas)):
            if type(alpha) is not str:
                old_ub[0].backward()
                alpha = {k: (ten - lr * ten.grad).clamp(0, 1).detach().requires_grad_() for k, ten in alpha.items()}
            out_ub, out_alpha, _ = deep_poly(layers, alpha, input_lb, input_ub)
            remaining_labels = remaining_labels[out_ub > 0]
            if len(remaining_labels) == 0:
                dprint(f"Verified after {(datetime.now()-start_time).total_seconds()} seconds. "
                       f"[epoch: {epoch}; i: {i}; alpha: {alpha})]")
                return True
            if out_ub.min() <= 0:
                dprint(f"{len(remaining_labels)} categories left to beat: {remaining_labels}.")
                layers = with_comparison_layer(net_layers, true_label, adversarial_labels=remaining_labels)
            out_ubs[i], alphas[i] = out_ub, out_alpha
        dprint(f"out_ubs after epoch {epoch}: {[out_ub.detach().numpy() for out_ub in out_ubs]}")
        if epoch % evolution_period == 0 and epoch > 0:
            total_losses = Tensor([ReLU()(out_ub).sum() for out_ub in out_ubs])
            alphas[torch.argmax(total_losses)] = "noisymin"
    dprint(f"Failed to verify after {(datetime.now()-start_time).total_seconds()} seconds and {max_iter} epochs.")
    return False


def with_comparison_layer(net_layers: Sequential, true_label: int, adversarial_labels: Tensor):
    num_categories = net_layers[-1].out_features
    weight = torch.eye(num_categories)[adversarial_labels, :]
    weight[:, true_label] = -1
    comparison_layer = Linear(in_features=num_categories, out_features=len(adversarial_labels), bias=False)
    comparison_layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    return Sequential(*net_layers, comparison_layer)


def print_if(msg: str, condition: bool):
    if condition:
        print(msg)


def dprint(msg: str):
    print_if(msg, DEBUG)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def analyze(net, inputs, eps, true_label):
    set_seed(0)
    if type(net) == NormalizedResnet:
        normalizer = net.normalization
        layers = Sequential(*itertools.chain.from_iterable([(layer if type(layer) is Sequential else [layer]) for layer in net.resnet]))
    else:
        normalizer = net.layers[0]
        layers = net.layers[1:]
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
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
