import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, ReLU

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
    direct_lbs, direct_ubs = abstract_lower[-1], abstract_upper[-1]
    for prev_abs_lbs, prev_abs_ubs in reversed(zip(abstract_lower, abstract_upper)[:-1]):
        next_direct_lbs = np.zeros((len(direct_lbs), prev_abs_lbs.shape[1]))
        next_direct_ubs = np.zeros((len(direct_lbs), prev_abs_lbs.shape[1]))
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
        concrete_lbs += direct_lbs[:,1:] @ ((inputs-eps) * (direct_lbs[i,1:] >= 0) + (inputs+eps) * ((direct_lbs[i,1:] < 0)))
        concrete_ubs += direct_ubs[:, 1:] @ ((inputs + eps) * (direct_ubs[i, 1:] >= 0) + (inputs - eps) * ((direct_ubs[i, 1:] < 0)))
    return concrete_lbs,concrete_ubs


def analyze(net, inputs, eps, true_label):
    num_categories = net.layers[-1].out_features
    comparison_layer = Linear(num_categories, num_categories)
    comparison_layer.weight = np.delete(np.repeat(
        ((np.arange(num_categories) == true_label)+0).reshape((1, -1)), num_categories, axis=0
    ) - np.eye(num_categories), true_label, axis=0)
    net = nn.Sequential(net, comparison_layer)
    alpha = []
    abstract_lower = []
    abstract_upper = []
    for k, layer in enumerate(net.layers):
        alpha.append(torch.Tensor(np.rand(len(layer))))
        if type(layer) == Linear:
            coefficients = torch.hstack((layer.bias.detach().reshape(-1, 1), layer.weight.detach())).numpy()
            abstract_lower.append(coefficients)
            abstract_upper.append(coefficients)
        if type(layer) == ReLU:
            prev_lbs, prev_ubs = backtrack(abstract_lower, abstract_upper, inputs, eps)
            curr_abstract_lower = []
            curr_abstract_upper = []
            for i, prev_lb, prev_ub in enumerate(zip(prev_lbs, prev_ubs)):
                if prev_ub <= 0:
                    shared_coefficients = np.zeros(len(layer))
                    curr_abstract_lower.append(shared_coefficients)
                    curr_abstract_upper.append(shared_coefficients)
                elif prev_lb >= 0:
                    shared_coefficients = (np.arange(len(layer)+1) == i+1)+0
                    curr_abstract_lower.append(shared_coefficients)
                    curr_abstract_upper.append(shared_coefficients)
                else:
                    upper_slope = prev_ub / (prev_ub - prev_lb)
                    curr_abstract_lower.append(alpha[-1] * abstract_lower[-1])
                    curr_abstract_upper.append(upper_slope * (abstract_upper[-1] - prev_lb))
    final_lbs, _ = backtrack(abstract_lower, abstract_upper, inputs, eps)
    return all([final_lb >= 0 for final_lb in final_lbs])


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
