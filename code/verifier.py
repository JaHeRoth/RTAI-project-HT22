import argparse
import csv
from enum import Enum
from itertools import product
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor, Size
from torch.nn import Linear, ReLU, Conv2d, BatchNorm2d, Sequential

from resnet import BasicBlock
from networks import get_network, get_net_name, NormalizedResnet, Normalization, FullyConnected, Conv

DEVICE = 'cpu'
DTYPE = torch.float32

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


Bounds = List[Tuple[Tensor, Tensor, Tensor, Tensor]]


def backtrack(bias: Tensor, coefficients: Tensor, past_bounds: Bounds, input_lb: Tensor, input_ub: Tensor):
    # Possible optimization: concretize every n layers to see if concrete lower is above 0 or concrete upper is below 0
    direct_lb, direct_ub = coefficients, coefficients
    for abstract_lb, abstract_ub, _, _ in reversed(past_bounds):
        # TODO: Fix crash here. Problem boils down to how to deal with bias
        direct_lb = direct_lb * (direct_lb > 0) @ abstract_lb + direct_lb * (direct_lb < 0) @ abstract_ub
        direct_ub = direct_ub * (direct_ub > 0) @ abstract_ub + direct_ub * (direct_ub < 0) @ abstract_lb
    concrete_lb = bias + direct_lb * (direct_lb > 0) @ input_lb + direct_lb * (direct_lb < 0) @ input_ub
    concrete_ub = bias + direct_ub * (direct_ub > 0) @ input_ub + direct_ub * (direct_ub < 0) @ input_lb
    return concrete_lb, concrete_ub


def conv_to_affine(layer: Conv2d, in_height: int, in_width: int, bn_layer: BatchNorm2d = None):
    """:return Coefficients such that an inner product between this and flattened input (prepended by 1 for bias)
    gives same result as flattening result of applying convolution layer (and possibly bn_layer) on input."""
    hpadding, wpadding = layer.kernel_size
    hstride, wstride = layer.stride
    num_filters, depth, filter_height, filter_width = layer.weight.shape
    padded_height, padded_width = in_height + 2 * hpadding, in_width + 2 * wpadding
    num_hsteps, num_wsteps = (padded_height - filter_height) // hstride + 1, (padded_width - filter_width) // wstride + 1
    linear_coefficients_tensor = torch.empty(num_filters, num_hsteps, num_wsteps, depth, in_height, in_width)
    for f, r, c in product(range(num_filters), range(num_hsteps), range(num_wsteps)):
        padded_coefficients = torch.zeros((depth, padded_height, padded_width))
        start_row, start_column = r * hstride, c * wstride
        padded_coefficients[:, start_row : start_row + filter_height, start_column : start_column + filter_width] = layer.weight[f]
        relevant_coefficients = padded_coefficients[:, hpadding:-hpadding, wpadding:-wpadding]
        linear_coefficients_tensor[f, r, c] = relevant_coefficients
    filter_intercept = torch.zeros(num_filters) if layer.bias is None else layer.bias
    if bn_layer is not None:
        filter_intercept += bn_layer.bias - bn_layer.weight * bn_layer.running_mean / torch.sqrt(bn_layer.running_var + bn_layer.eps)
        linear_coefficients_tensor *= (bn_layer.weight + 1 / torch.sqrt(bn_layer.running_var + bn_layer.eps)).reshape(-1, 1, 1, 1, 1, 1)
    intercept = filter_intercept.repeat_interleave(num_hsteps * num_wsteps).reshape(-1,1)
    linear_coefficients = linear_coefficients_tensor.reshape((num_filters * num_hsteps * num_wsteps, -1))
    return intercept, linear_coefficients


def affine_bounds(bias: Tensor, coefficients: Tensor, past_bounds: Bounds, input_lb: Tensor, input_ub: Tensor):
    abstract_lower = abstract_upper = torch.hstack([bias.reshape(-1, 1), coefficients])
    concrete_lower, concrete_upper = backtrack(bias, coefficients, past_bounds, input_lb, input_ub)
    return abstract_lower, abstract_upper, concrete_lower, concrete_upper


def fc_bounds(layer: Linear, past_bounds: Bounds, input_lb: Tensor, input_ub: Tensor):
    bias = layer.bias.detach() if layer.bias is not None else torch.zeros(layer.out_features)
    return affine_bounds(bias, layer.weight.detach(), past_bounds, input_lb, input_ub)


def conv_bounds(layer: Conv2d, past_bounds: Bounds, input_lb: Tensor, input_ub: Tensor, in_height: int, in_width: int, bn_layer: Optional[BatchNorm2d]):
    intercept, coefficients = conv_to_affine(layer.detach(), in_height, in_width, bn_layer.detach())
    return affine_bounds(intercept, coefficients, past_bounds, input_lb, input_ub)


def generate_alpha(in_lb: Tensor, in_ub: Tensor, strategy: str):
    if strategy == "half":
        return torch.ones(in_lb.shape) / 2
    elif strategy == "rand":
        return torch.rand(in_lb.shape)
    elif strategy == "min":
        return (in_ub > -in_lb).float()
    raise ValueError(f"{strategy} is an invalid alpha-generating strategy.")


def relu_bounds(past_bounds: Bounds, alpha: Union[Tensor, str]):
    prev_lb, prev_ub = past_bounds[-1][-2:]
    if type(alpha) == str:
        alpha = generate_alpha(prev_lb, prev_ub, strategy=alpha)
    upper_slope = prev_ub / (prev_ub - prev_lb)
    in_len = len(past_bounds[-1][0])
    lb_bias, ub_bias, lb_scaling, ub_scaling = [torch.zeros(in_len) for _ in range(4)]
    lb_scaling[prev_lb >= 0], ub_scaling[prev_lb >= 0] = 1, 1
    crossing_mask = (prev_lb < 0) & (prev_ub > 0)
    lb_scaling[crossing_mask] = alpha[crossing_mask]
    ub_scaling[crossing_mask] = upper_slope[crossing_mask]
    ub_bias[crossing_mask] = (-upper_slope * prev_lb)[crossing_mask]
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


def deep_poly(layers: Sequential, alpha: Union[str, Dict[int, Tensor]], input_lb: Tensor, input_ub: Tensor):
    in_dims = infer_layer_input_dimensions(layers, input_lb)
    input_lb, input_ub = input_lb.flatten(), input_ub.flatten()
    bounds: Bounds = []
    out_alpha: Dict[int, Tensor] = {}
    for k, layer in enumerate(layers):
        if type(layer) == Linear:
            bounds.append(fc_bounds(layer, bounds, input_lb, input_ub))
        elif type(layer) == Conv2d:
            in_height, in_width = in_dims[k][-2:]
            bn_layer = layers[k+1] if type(layers[k+1]) == BatchNorm2d else None
            bounds.append(conv_bounds(layer, bounds, input_lb, input_ub, in_height, in_width, bn_layer))
        elif type(layer) == ReLU:
            bound, out_alpha[k] = relu_bounds(bounds, alpha[k] if type(alpha) == Dict else alpha)
            bounds.append(bound)
        elif type(layer) == BasicBlock:
            raise NotImplementedError # TODO: Implement
    output_ub = bounds[-1][3]
    return output_ub, out_alpha


def make_loss_layers(layers, true_label):
    num_categories = layers[-1].out_features
    comparison_layer = Linear(in_features=num_categories, out_features=num_categories - 1, bias=False)
    comparison_layer.weight = torch.nn.Parameter(torch.Tensor(np.delete(np.eye(num_categories) - np.repeat(
        (np.arange(num_categories) == true_label).reshape((1, -1)), num_categories, axis=0
    ), true_label, axis=0)), requires_grad=False)
    sum_layer = Linear(in_features=num_categories - 1, out_features=1, bias=False)
    sum_layer.weight = torch.nn.Parameter(torch.ones((1, num_categories - 1)), requires_grad=False)
    return comparison_layer, nn.ReLU(), sum_layer


# TODO: alpha_optimizing_deep_poly

# TODO?: ensemble (like following pseudocode:)
# for c in categories:
#     for epoch in epochs:
#         for ens in ensebmles:
#             update(ens)


def analyze(net, inputs, eps, true_label):
    if type(net) == NormalizedResnet:
        normalizer = net.normalization
        layers = net.resnet
    else:
        normalizer = net.layers[0]
        # TODO: Flatten nested Sequentials (if doesn't change functioning of network)
        layers = net.layers[1:]
    input_lb, input_ub = (inputs - eps).clamp(0, 1), (inputs + eps).clamp(0, 1)
    normalized_lb, normalized_ub = normalizer(input_lb), normalizer(input_ub)
    layers = nn.Sequential(*layers, *make_loss_layers(layers, true_label))
    loss, _ = deep_poly(layers, "min", normalized_lb, normalized_ub)
    return loss == 0


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
