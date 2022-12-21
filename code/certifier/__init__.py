import itertools
import math
from datetime import datetime
from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Sequential, Linear

from .networks import NormalizedResnet
from .constants import DEBUG
from .deep_poly import deep_poly
from .logger import dprint


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


def choose_alpha(index: int):
    if index == 2:
        return "smoothmin"
    elif index % 2:
        return "gaussmin"
    return "noisymin"


def ensemble_poly(net_layers: Sequential, input_lb: Tensor, input_ub: Tensor, true_label: int):
    """
    Optimize multiple combinations of alphas simultaenously to allow DeepPoly to rule out all
    other categories than `true_label`, ruling out a category for good once any alpha value does so.

    :return: Whether we found for each category that wasn't `true_label` an alpha that ruled it out
    for all inputs (to `net_layers`) between `input_lb` and `input_ub`.
    """
    start_time = datetime.now()
    remaining_labels = Tensor([c for c in range(net_layers[-1].out_features) if c != true_label]).long()
    layers = with_comparison_layer(net_layers, true_label, adversarial_labels=remaining_labels)
    alphas = ["min"]
    out_ubs: List[Optional[Tensor]] = [None for _ in alphas]
    c2a_cache = {}

    # Except when debugging there is no point in giving up early, since printing
    # "not verified" gives 0 points, just like timing out does
    desired_iter = 15
    max_iter = (desired_iter + 1) if DEBUG else 10 ** 9
    no_grad = False
    learning_rate = 5
    seconds_per_epoch = 60 / desired_iter
    for epoch in range(max_iter):
        epoch_start = datetime.now()
        for i, (old_ub, alpha) in enumerate(zip(out_ubs, alphas)):
            # After the first epoch, we initialized the alphas for each strategy and do a Gradient Descent step
            if type(alpha) is not str:
                st = datetime.now()
                # We want each alpha to focus its optimization on the class it's the closest to beating
                old_ub[old_ub.argmin()].backward()
                # Gradient descent step
                alpha = {k: (ten - learning_rate / math.sqrt(epoch + 1) * ten.grad).clamp(0, 1).detach().requires_grad_() for k, ten in alpha.items()}
                dprint(f"Spent {(datetime.now() - st).total_seconds()} seconds on backprop and updating.")
            st = datetime.now()
            out_ub, out_alpha, _, c2a_cache = deep_poly(layers, alpha, input_lb, input_ub, c2a_cache, no_grad=no_grad)
            dprint(f"Spent {(datetime.now() - st).total_seconds()} seconds running DeepPoly (one forward pass).")
            out_ubs[i], alphas[i] = out_ub, out_alpha
            should_remain = out_ub >= 0
            remaining_labels = remaining_labels[should_remain]
            if len(remaining_labels) == 0:
                dprint(f"Verified after {(datetime.now() - start_time).total_seconds()} seconds. "
                       f"[epoch: {epoch}; i: {i}; alpha: {alpha})]")
                return True
            if out_ub.min() < 0:
                dprint(f"{len(remaining_labels)} categories left to beat: {remaining_labels}.")
                layers = with_comparison_layer(net_layers, true_label, adversarial_labels=remaining_labels)
                # We want out_ubs to only contain bounds for remaining classes, as only these are relevant
                for j in range(i+1):
                    out_ubs[j] = out_ubs[j][should_remain]
                # As fewer classes now remain, we re-init the alpha furthest from beating one of these (evolution step)
                min_losses = Tensor([ub.min() for ub in out_ubs])
                alphas[min_losses.argmax()] = "noisymin"
            # Estimates whether we can reach `desired_iter` number of epochs within the one minute we have
            # if we add another alpha, and if so adds it. epoch 0 is a bit special, so we wait till epoch 1
            if epoch == 1 and (datetime.now() - epoch_start).total_seconds() * (
                    len(alphas) + 1) / len(alphas) < seconds_per_epoch:
                alphas.append(choose_alpha(i))
                out_ubs.append(None)
        dprint(f"out_ubs after epoch {epoch}: {[out_ub.detach().numpy() for out_ub in out_ubs]}")
    dprint(f"Failed to verify after {(datetime.now() - start_time).total_seconds()} seconds and {max_iter} epochs.")
    return False


def certify(net, inputs, eps, true_label):
    if type(net) == NormalizedResnet:
        normalizer = net.normalization
        # Flatten the nested ResNet by unfolding Sequential layers (Keeps the BasicBlock layers intact)
        layers = Sequential(*itertools.chain.from_iterable(
            [(layer if type(layer) is Sequential else [layer]) for layer in net.resnet]))
    else:
        normalizer = net.layers[0]
        layers = net.layers[1:]
    input_lb, input_ub = (inputs - eps).clamp(0, 1), (inputs + eps).clamp(0, 1)
    normalized_lb, normalized_ub = normalizer(input_lb), normalizer(input_ub)
    # Ensure we crash (which counts as not verified) if gradients are nan or inf
    with torch.autograd.set_detect_anomaly(True):
        return ensemble_poly(layers, normalized_lb, normalized_ub, true_label)
