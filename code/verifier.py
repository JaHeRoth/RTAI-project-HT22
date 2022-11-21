import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, ReLU, Conv2d

from networks import get_network, get_net_name, NormalizedResnet


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


def backtrack(abstract_lower, abstract_upper, inputs, eps):
    inputs = torch.flatten(inputs)
    direct_lbs, direct_ubs = abstract_lower[-1], abstract_upper[-1]
    for prev_abs_lbs, prev_abs_ubs in reversed(list(zip(abstract_lower, abstract_upper))[:-1]):
        next_direct_lbs = torch.zeros((len(direct_lbs), prev_abs_lbs.shape[1]))
        next_direct_ubs = torch.zeros((len(direct_lbs), prev_abs_lbs.shape[1]))
        for i in range(len(direct_lbs)):
            for j in range(direct_lbs.shape[1]-1):
                next_direct_lbs[i,:] += direct_lbs[i,j+1] * (prev_abs_lbs if direct_lbs[i,j+1] > 0 else prev_abs_ubs)[j,:]
                next_direct_ubs[i,:] += direct_ubs[i,j+1] * (prev_abs_lbs if direct_ubs[i,j+1] < 0 else prev_abs_ubs)[j,:]
        next_direct_lbs[:,0] += direct_lbs[:,0]
        next_direct_ubs[:, 0] += direct_ubs[:, 0]
        direct_lbs, direct_ubs = next_direct_lbs, next_direct_ubs
    concrete_lbs = direct_lbs[:,0]
    concrete_ubs = direct_ubs[:, 0]
    for i in range(len(direct_lbs)):
        # Shows how we can get rid of inner loop above (using indicator functions)
        concrete_lbs[i] += direct_lbs[i,1:] @ ((inputs-eps) * (direct_lbs[i,1:] >= 0) + (inputs+eps) * ((direct_lbs[i,1:] < 0)))
        concrete_ubs[i] += direct_ubs[i, 1:] @ ((inputs + eps) * (direct_ubs[i, 1:] >= 0) + (inputs - eps) * ((direct_ubs[i, 1:] < 0)))
    return concrete_lbs, concrete_ubs


def conv_to_affine(layer: Conv2d, in_height: int, in_width: int):
    hpadding, wpadding = layer.kernel_size
    hstride, wstride = layer.stride
    num_filters, depth, filter_height, filter_width = layer.weight.shape
    padded_height, padded_width = in_height + 2 * hpadding, in_width + 2 * wpadding
    num_hsteps, num_wsteps = (padded_height - filter_height) // hstride + 1, (padded_width - filter_width) // wstride + 1
    linear_coefficients_tensor = torch.empty(num_filters, depth, padded_height, padded_width, in_height, in_width)
    for f in range(num_filters):
        for l in range(depth):
            for r in range(num_hsteps):
                for c in range(num_wsteps):
                    padded_coefficients = torch.zeros((padded_height, padded_width))
                    start_row, start_column = r * hstride, c * wstride
                    padded_coefficients[start_row : start_row + filter_height, start_column : start_column + filter_width] = layer.weight[f, l]
                    relevant_coefficients = padded_coefficients[hpadding:-hpadding, wpadding:-wpadding]
                    linear_coefficients_tensor[f, l, r, c] = relevant_coefficients
    linear_coefficients = linear_coefficients_tensor.flatten(start_dim=1)
    bias = torch.zeros(num_filters) if layer.bias is None else layer.bias
    affine_coefficients = torch.hstack([bias.reshape(-1,1), linear_coefficients])
    return affine_coefficients



def analyze(net, inputs, eps, true_label):
    from torchviz import make_dot
    make_dot(net(inputs), params=dict(net.named_parameters())).render("resnet_torchviz", format="png")
    import hiddenlayer as hl
    hl.build_graph(net, inputs).save('resnet_hiddenlayer', format='png')
    print(net)
    conv_to_affine(net.layers[1], *inputs.shape[-2:])
    # Maybe this can be replaced with direct operations on output. Or maybe current form is fine, but should rather be passed in
    num_categories = net.layers[-1].out_features
    comparison_layer = Linear(in_features=num_categories, out_features=num_categories-1, bias=False)
    comparison_layer.weight = torch.nn.Parameter(torch.Tensor(np.delete(np.eye(num_categories) - np.repeat(
        (np.arange(num_categories) == true_label).reshape((1, -1)), num_categories, axis=0
    ), true_label, axis=0)), requires_grad=False)
    sum_layer = Linear(in_features=num_categories-1, out_features=1, bias=False)
    sum_layer.weight = torch.nn.Parameter(torch.ones((1,num_categories-1)), requires_grad=False)
    net.layers = nn.Sequential(*net.layers, comparison_layer, nn.ReLU(), sum_layer)

    # Alpha should ideally be passed as function input, or the like. Maybe easiest to just have alphas for all layers
    # Seems like out_features should be replaced by out_channels for Conv2D layers
    alpha = [torch.rand(net.layers[k-1].out_features, requires_grad=False) for k, layer in enumerate(net.layers) if type(layer) == ReLU]
    alpha_i = 0

    abstract_lower = []
    abstract_upper = []
    for k, layer in enumerate(net.layers):
        if type(layer) == Linear:
            coefficients = torch.hstack(((layer.bias if layer.bias is not None else torch.zeros(layer.out_features)).detach().reshape(-1, 1), layer.weight.detach()))
            abstract_lower.append(coefficients)
            abstract_upper.append(coefficients)
        elif type(layer) == ReLU:
            prev_lbs, prev_ubs = backtrack(abstract_lower, abstract_upper, inputs, eps)
            curr_abstract_lower = []
            curr_abstract_upper = []
            # This should be parallelizable
            for i, (prev_lb, prev_ub) in enumerate(zip(prev_lbs, prev_ubs)):
                in_features = len(abstract_lower[-1])
                if prev_ub <= 0:
                    shared_coefficients = torch.zeros(in_features+1)
                    curr_abstract_lower.append(shared_coefficients)
                    curr_abstract_upper.append(shared_coefficients)
                elif prev_lb >= 0:
                    shared_coefficients = (torch.arange(in_features+1) == i+1)+0
                    curr_abstract_lower.append(shared_coefficients)
                    curr_abstract_upper.append(shared_coefficients)
                else:
                    upper_slope = prev_ub / (prev_ub - prev_lb)
                    relevant_input_mask = (torch.arange(in_features) == i)+0
                    curr_abstract_lower.append(torch.hstack((torch.zeros(1), relevant_input_mask*alpha[alpha_i][i])))
                    curr_abstract_upper.append(torch.hstack((-upper_slope * prev_lb, relevant_input_mask*upper_slope)))
            abstract_lower.append(torch.vstack(curr_abstract_lower))
            abstract_upper.append(torch.vstack(curr_abstract_upper))
            alpha_i += 1
    _, loss = backtrack(abstract_lower, abstract_upper, inputs, eps)
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
