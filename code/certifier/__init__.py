import itertools
from datetime import datetime
from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Sequential, ReLU, Linear

from .cache import ConvToAffineCache
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
    alphas = ["min", "smoothmin", "noisymin"]
    # The upper bounds we achieve with the three different strategies
    out_ubs: List[Optional[Tensor]] = [None for _ in alphas]
    c2a_cache = ConvToAffineCache()

    # TODO: Arbitrary hyperparameters, further tuning is needed
    # Except when debugging there is no point in giving up early, since printing
    # "not verified" gives 0 points, just like timing out does
    max_iter = 10 if DEBUG else 10 ** 9
    evolution_period = 5 if DEBUG else 10
    learning_rate = 10**0
    for epoch in range(max_iter):
        # We iterate over our ensemble of 3 alpha strategies
        for i, (old_ub, alpha) in enumerate(zip(out_ubs, alphas)):
            # After the first epoch, we initialized the alphas for each strategy and do a Gradient Descent step
            if type(alpha) is not str:
                # TODO?: According to documentation, gradients do get aggregated over time, do we need to null them here for each epoch?
                # Meaning alpha[k].grad.zero_() for k in alpha.keys()?
                # Or is this solved by our design that we always detach the alphas and then build a new graph in the next forward pass? Would investigate. -> Done

                # We always optimize/differentiate w.r.t. the first unbeaten competitive class comparison
                # The resulting gradient is stored in the alpha values, which are the leafes of the computational graph
                # TODO?: Just for myself, check that the index here does not need to be updated, i.e. that the first unbeaten class is always at index 0 because
                # we always remove the beaten classes. Should happen through changing the comparison layer and the execution of deep_poly, but just to be sure.
                st = datetime.now()
                old_ub[0].backward()
                # Gradient descent step
                alpha = {k: (alpha[k] - learning_rate * alpha[k].grad).clamp(0, 1).detach().requires_grad_() for k in alpha.keys()}
                dprint(f"Spent {(datetime.now() - st).total_seconds()} seconds on backprop and updating.")
            st = datetime.now()
            out_ub, out_alpha, _ = deep_poly(layers, alpha, input_lb, input_ub, c2a_cache)
            dprint(f"Spent {(datetime.now() - st).total_seconds()} seconds running DeepPoly (one forward pass).")
            # TODO: Are we satisfied with a tie? Or do we need to be strictly better? Could be an (unlikely) error source
            remaining_labels = remaining_labels[out_ub >= 0]
            if len(remaining_labels) == 0:
                # We ruled out all other categories, so we're done
                dprint(f"Verified after {(datetime.now() - start_time).total_seconds()} seconds. "
                       f"[epoch: {epoch}; i: {i}; alpha: {alpha})]")
                return True
            # TODO: Again maybe needs a change if we change the tie-breaking strategy
            if out_ub.min() < 0:
                # We found an alpha that ruled out at least one category, so we update the comparison layer
                dprint(f"{len(remaining_labels)} categories left to beat: {remaining_labels}.")
                # TODO?: Dumb idea, but is there anything in the comp graph of layers that we lose when we do this override?
                # Or is everything solved by the fact, that our next run of deep_poly will build a new comp graph? -> yep
                # TODO?: Also, maybe another stupid thought, but does this even help computation in any way? Since we'll optimize
                # the alphas only regarding the first to-be-beaten category anyways, so the edges that fall away here don't contribute anyways?
                # Of course it's very neat with the current implementation, but I don't know how big the overhead of rebuilding the network is. -> Overhead is negligible
                layers = with_comparison_layer(net_layers, true_label, adversarial_labels=remaining_labels)
            # Save the upper bound and the alpha values for the right strategy
            out_ubs[i], alphas[i] = out_ub, out_alpha
        dprint(f"out_ubs after epoch {epoch}: {[out_ub.detach().numpy() for out_ub in out_ubs]}")
        # Do a little bit of evolution, i.e. mutate the worst performing alpha after some epochs
        if epoch % evolution_period == 0 and epoch > 0:
            # TODO?: Is this a typo or am I stupid, the extra brackets after ReLU()? -> Works
            total_losses = Tensor([ReLU()(out_ub).sum() for out_ub in out_ubs])
            # Worst performing strategy will be reinitialized with a noisy min strategy
            alphas[torch.argmax(total_losses)] = "noisymin"
    dprint(f"Failed to verify after {(datetime.now() - start_time).total_seconds()} seconds and {max_iter} epochs.")
    return False


def certify(net, inputs, eps, true_label):
    # We extract the normalization layer to work with the already normalized inputs,
    # since our epsilons are also in the normalized space
    if type(net) == NormalizedResnet:
        normalizer = net.normalization
        # Flatten the nested ResNet by unfolding Sequential layers (Keeps the BasicBlock layers intact)
        layers = Sequential(*itertools.chain.from_iterable(
            [(layer if type(layer) is Sequential else [layer]) for layer in net.resnet]))
    else:
        normalizer = net.layers[0]
        layers = net.layers[1:]
    # We only have to account for valid input pixels, so we clamp back to [0, 1]
    input_lb, input_ub = (inputs - eps).clamp(0, 1), (inputs + eps).clamp(0, 1)
    normalized_lb, normalized_ub = normalizer(input_lb), normalizer(input_ub)
    # Ensure we crash (which counts as not verified) if gradients are nan or inf
    with torch.autograd.set_detect_anomaly(True):
        return ensemble_poly(layers, normalized_lb, normalized_ub, true_label)
