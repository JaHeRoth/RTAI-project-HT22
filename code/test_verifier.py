import pytest
import torch
import itertools
from torch.nn import Sequential, Conv2d

from verifier import get_net
from certifier.deep_poly import conv_to_affine, deep_poly
from certifier.networks import Conv as ConvNet
from certifier.constants import SequentialCache

from dummy_networks import FullyConnected, Conv, ResNet, BasicBlock

"""
Test suite:

- conv_to_affine(): 
    test the conv_to_affine function for correctness
    - With bias, without batch_norm
    - Without bias, without batch_norm
    - With bias, with batch_norm
    - Without bias, with batch_norm

- deep_poly():
    test the forward pass for correctness
    - For a fully connected dummy network
    - For a convolutional dummy network
        - Without batch_norm
    - For a resnet dummy network

"""

def test_conv_to_affine_without_bias_without_batch_norm_net4():
    # Extract the layer from one of the official networks
    net = get_net('net4', 'net4_mnist_conv1.pt')
    layer = net.layers[1]
    layer.bias = None
    # Take a random input (shape of MNIST images)
    x = torch.randn(1, 1, 28, 28)
    # Run the convolutional layer
    y = layer(x)
    original_output = y.flatten()
    # Convert the convolutional layer to affine
    bias, affine = conv_to_affine(layer, 28, 28)
    # Run the affine layer
    z = affine @ x.flatten() + bias.flatten()
    # Check that the outputs are the same
    torch.testing.assert_close(original_output, z)


def test_conv_to_affine_with_bias_without_batch_norm_net5():
    # Extract the layer from one of the official networks
    net = get_net('net5', 'net5_mnist_conv2.pt')
    layer = net.layers[1]
    # Take a random input (shape of MNIST images)
    x = torch.randn(1, 1, 28, 28)
    # Run the convolutional layer
    y = layer(x)
    original_output = y.flatten()
    # Convert the convolutional layer to affine
    bias, affine = conv_to_affine(layer, 28, 28)
    # Run the affine layer
    z = affine @ x.flatten() + bias.flatten()
    # Check that the outputs are the same
    torch.testing.assert_close(original_output, z)


def test_conv_to_affine_without_bias_without_batch_norm_net6():
    # Extract the layer from one of the official networks
    net = get_net('net6', 'net6_cifar10_conv2.pt')
    layer = net.layers[1]
    layer.bias = None
    # Take a random input (shape of CIFAR-10 images)
    x = torch.randn(1, 3, 32, 32)
    # Run the convolutional layer
    y = layer(x)
    original_output = y.flatten()
    # Convert the convolutional layer to affine
    bias, affine = conv_to_affine(layer, 32, 32)
    # Run the affine layer
    z = affine @ x.flatten() + bias.flatten()
    # Check that the outputs are the same
    torch.testing.assert_close(original_output, z)


def test_conv_to_affine_with_bias_without_batch_norm_net6():
    # Extract the layer from one of the official networks
    net = get_net('net6', 'net6_cifar10_conv2.pt')
    layer = net.layers[1]
    # Take a random input (shape of CIFAR-10 images)
    x = torch.randn(1, 3, 32, 32)
    # Run the convolutional layer
    y = layer(x)
    original_output = y.flatten()
    # Convert the convolutional layer to affine
    bias, affine = conv_to_affine(layer, 32, 32)
    # Run the affine layer
    z = affine @ x.flatten() + bias.flatten()
    # Check that the outputs are the same
    torch.testing.assert_close(original_output, z)


# Sanity check for the matrix multiplication above
def test_matrix_multiplication():
    # Extract the layer from one of the official networks
    net = get_net('net6', 'net6_cifar10_conv2.pt')
    layer = net.layers[1]
    # Take a random input (shape of CIFAR-10 images)
    x = torch.randn(1, 3, 32, 32)
    # Convert the convolutional layer to affine
    bias, affine = conv_to_affine(layer, 32, 32)
    # Run the affine layer
    y = affine @ x.flatten() + bias.flatten()
    z = affine @ x.view(-1) + bias.view(-1)
    # Check that the outputs are the same
    assert torch.allclose(y, z)


# Sanity check 2 for the matrix multiplication above
def test_linear_layer_matrix_multiplication_equivalence():
    # Extract the layer from one of the official networks
    net = get_net('net6', 'net6_cifar10_conv2.pt')
    layer = net.layers[1]
    # Take a random input (shape of CIFAR-10 images)
    x = torch.randn(1, 3, 32, 32)
    a = torch.randn(3072)
    # Convert the convolutional layer to affine
    bias, affine = conv_to_affine(layer, 32, 32)
    # Build a linear layer
    linear = torch.nn.Linear(affine.shape[1], affine.shape[0])
    linear.weight = torch.nn.Parameter(affine)
    linear.bias = torch.nn.Parameter(bias.flatten())
    # Run the affine layer
    y = affine @ x.flatten() + bias.flatten()
    z = linear(x.flatten())
    # Check that the outputs are the same
    assert torch.allclose(y, z)


def test_conv_to_affine_without_bias_with_batch_norm_net9():
    # Build a simple convolutional network with just one layer and no bias but a batch normalization layer
    net = get_net('net9', 'net9_cifar10_resnet_2b2_bn.pt')
    # This convolutional layer has no bias
    conv = net.resnet[0]
    batch_norm = net.resnet[1]
    net = torch.nn.Sequential(conv, batch_norm)
    # Take a random input (shape of CIFAR-10 images)
    x = torch.randn(1, 3, 32, 32)
    # Run the convolutional layer
    y = net(x)
    original_output = y.flatten()
    # Convert the convolutional layer to affine
    bias, affine = conv_to_affine(conv, 32, 32, batch_norm)
    # Run the affine layer
    z = affine @ x.flatten() + bias.flatten()
    # Check that the outputs are the same
    torch.testing.assert_close(original_output, z)


def test_deep_poly_fc_forward_with_min_initialization_crossing():
    # Build a simple fully connected network that we can test by hand
    fc_net = FullyConnected(2, [2, 1], act='relu')
    # Set the weights and biases
    fc_net.layers[0].weight = torch.nn.Parameter(torch.tensor([[1., 1.], [1., -1.]]))
    fc_net.layers[0].bias = torch.nn.Parameter(torch.tensor([0., 0.]))
    # Layer 1 is ReLU, so we don't need to set weights and biases
    fc_net.layers[2].weight = torch.nn.Parameter(torch.tensor([[1., -1.]]))
    fc_net.layers[2].bias = torch.nn.Parameter(torch.tensor([-0.5]))
    # Needed for deep_poly implementation
    layer = fc_net.layers
    # Input "image"
    x = torch.tensor([0., 1.])
    # Input region
    epsilon = 1.
    lb = x - epsilon
    ub = x + epsilon
    # We initialize the alphas for the ReLU layers
    alphas = 'min'

    # Check that the forward pass is correct
    # Check that added_bounds[:, -2:] is correct for all layers of the network (the concrete upper & lower bounds in added_bounds)
    # MVP check that output bound is correct at least
    c2a_cache = {}
    output_ub, out_alpha, added_bounds, c2a_cache = deep_poly(layer, alphas, lb, ub, c2a_cache)
    # Build the correct bounds, it's always [lb_node1, lb_node2], [ub_node1, ub_node2], etc.
    correct_bounds_per_layer = [(torch.tensor([-1., -3.]), torch.tensor([3., 1.])), (torch.tensor([-1., 0.]), torch.tensor([3., 1.])), (torch.tensor([-2.]), torch.tensor([2.5]))]
    # Check that the concrete bounds are correct for all layers
    for bounds, correct_bounds in zip(added_bounds, correct_bounds_per_layer):
        bounds = bounds[-2:]
        lb, ub, correct_lb, correct_ub = bounds[0], bounds[1], correct_bounds[0], correct_bounds[1]
        assert torch.allclose(lb, correct_lb)
        assert torch.allclose(ub, correct_ub)
    # Check that the initialized alphas are correct as well
    correct_alphas = torch.tensor([1., 0.])
    assert torch.allclose(out_alpha.get(1), correct_alphas)
    

def test_deep_poly_fc_forward_with_fixed_alphas_crossing():
    # Build a simple fully connected network that we can test by hand
    fc_net = FullyConnected(2, [2, 1], act='relu')
    # Set the weights and biases
    fc_net.layers[0].weight = torch.nn.Parameter(torch.tensor([[1., 1.], [1., -1.]]))
    fc_net.layers[0].bias = torch.nn.Parameter(torch.tensor([0., 0.]))
    # Layer 1 is ReLU, so we don't need to set weights and biases
    fc_net.layers[2].weight = torch.nn.Parameter(torch.tensor([[1., -1.]]))
    fc_net.layers[2].bias = torch.nn.Parameter(torch.tensor([-0.5]))
    # Needed for deep_poly implementation
    layer = fc_net.layers
    # Input "image"
    x = torch.tensor([0., 1.])
    # Input region
    epsilon = 1.
    lb = x - epsilon
    ub = x + epsilon
    # We initialize the alphas for the ReLU layers
    alphas = {1: torch.tensor([0.5, 0.5])}

    # Check that the forward pass is correct
    # Check that added_bounds[:, -2:] is correct for all layers of the network (the concrete upper & lower bounds in added_bounds)
    # MVP check that output bound is correct at least
    c2a_cache = {}
    output_ub, out_alpha, added_bounds, c2a_cache = deep_poly(layer, alphas, lb, ub, c2a_cache)
    # Build the correct bounds, it's always [lb_node1, lb_node2], [ub_node1, ub_node2], etc.
    correct_bounds_per_layer = [(torch.tensor([-1., -3.]), torch.tensor([3., 1.])), (torch.tensor([-0.5, -1.5]), torch.tensor([3., 1.])), (torch.tensor([-1.5]), torch.tensor([3.]))]
    # Check that the concrete bounds are correct for all layers
    for bounds, correct_bounds in zip(added_bounds, correct_bounds_per_layer):
        bounds = bounds[-2:]
        lb, ub, correct_lb, correct_ub = bounds[0], bounds[1], correct_bounds[0], correct_bounds[1]
        assert torch.allclose(lb, correct_lb)
        assert torch.allclose(ub, correct_ub)


def test_deep_poly_fc_backward_gradient_crossing():
    # Build a simple fully connected network that we can test by hand
    fc_net = FullyConnected(2, [2, 1], act='relu')
    # Set the weights and biases
    fc_net.layers[0].weight = torch.nn.Parameter(torch.tensor([[1., 1.], [1., -1.]]))
    fc_net.layers[0].bias = torch.nn.Parameter(torch.tensor([0., 0.]))
    # Layer 1 is ReLU, so we don't need to set weights and biases
    fc_net.layers[2].weight = torch.nn.Parameter(torch.tensor([[1., -1.]]))
    fc_net.layers[2].bias = torch.nn.Parameter(torch.tensor([-0.5]))
    # Needed for deep_poly implementation
    layer = fc_net.layers
    # Input "image"
    x = torch.tensor([0., 1.])
    # Input region
    epsilon = 1.
    lb = x - epsilon
    ub = x + epsilon
    # We initialize the alphas for the ReLU layers
    alphas = {1: torch.tensor([0.2, 0.7]).requires_grad_()}

    # Check that the forward pass is correct
    # Check that added_bounds[:, -2:] is correct for all layers of the network (the concrete upper & lower bounds in added_bounds)
    # MVP check that output bound is correct at least
    c2a_cache = {}
    output_ub, out_alpha, added_bounds, c2a_cache = deep_poly(layer, alphas, lb, ub, c2a_cache)
    
    # Calculate the gradients of the upper bound w.r.t. the alphas
    output_ub.backward()
    # Check that the gradients are correct (should be -(x1 - x2) for alpha2 and 0 for alpha1)
    correct_grads = torch.tensor([0., 1.])
    assert torch.allclose(alphas.get(1).grad, correct_grads)


def test_deep_poly_fc_forward_with_min_initialization_not_crossing_positive():
    # Build a simple fully connected network that we can test by hand
    fc_net = FullyConnected(2, [2, 1], act='relu')
    # Set the weights and biases
    fc_net.layers[0].weight = torch.nn.Parameter(torch.tensor([[1., 1.], [1., -1.]]))
    fc_net.layers[0].bias = torch.nn.Parameter(torch.tensor([0., 0.]))
    # Layer 1 is ReLU, so we don't need to set weights and biases
    fc_net.layers[2].weight = torch.nn.Parameter(torch.tensor([[1., -1.]]))
    fc_net.layers[2].bias = torch.nn.Parameter(torch.tensor([-0.5]))
    # Needed for deep_poly implementation
    layer = fc_net.layers
    # Input "image" for which we have no ReLU crossing
    x = torch.tensor([2., 3.])
    # Input region
    epsilon = 1.
    lb = x - epsilon
    ub = x + epsilon
    # We initialize the alphas for the ReLU layers
    alphas = 'min'

    # Check that the forward pass is correct
    # Check that added_bounds[:, -2:] is correct for all layers of the network (the concrete upper & lower bounds in added_bounds)
    # MVP check that output bound is correct at least
    c2a_cache = {}
    output_ub, out_alpha, added_bounds, c2a_cache = deep_poly(layer, alphas, lb, ub, c2a_cache)
    # Build the correct bounds, it's always [lb_node1, lb_node2], [ub_node1, ub_node2], etc.
    correct_bounds_per_layer = [(torch.tensor([3., -3.]), torch.tensor([7., 1.])), (torch.tensor([3., 0.]), torch.tensor([7., 1.])), (torch.tensor([2.]), torch.tensor([6.5]))]
    # Check that the concrete bounds are correct for all layers
    for bounds, correct_bounds in zip(added_bounds, correct_bounds_per_layer):
        bounds = bounds[-2:]
        lb, ub, correct_lb, correct_ub = bounds[0], bounds[1], correct_bounds[0], correct_bounds[1]
        assert torch.allclose(lb, correct_lb)
        assert torch.allclose(ub, correct_ub)
    # Check that the initialized alphas are correct as well
    # Currently, alpha values get initialized even when the ReLU is not crossed, with the typical rule
    correct_alphas = torch.tensor([1., 0.])
    assert torch.allclose(out_alpha.get(1), correct_alphas)


def test_deep_poly_fc_backward_gradient_not_crossing_positive():
    # Build a simple fully connected network that we can test by hand
    fc_net = FullyConnected(2, [2, 1], act='relu')
    # Set the weights and biases
    fc_net.layers[0].weight = torch.nn.Parameter(torch.tensor([[1., 1.], [1., -1.]]))
    fc_net.layers[0].bias = torch.nn.Parameter(torch.tensor([0., 0.]))
    # Layer 1 is ReLU, so we don't need to set weights and biases
    fc_net.layers[2].weight = torch.nn.Parameter(torch.tensor([[1., -1.]]))
    fc_net.layers[2].bias = torch.nn.Parameter(torch.tensor([-0.5]))
    # Needed for deep_poly implementation
    layer = fc_net.layers
    # Input "image"
    x = torch.tensor([2., 3.])
    # Input region
    epsilon = 1.
    lb = x - epsilon
    ub = x + epsilon
    # We initialize the alphas for the ReLU layers
    alphas = 'min'

    # Check that the forward pass is correct
    # Check that added_bounds[:, -2:] is correct for all layers of the network (the concrete upper & lower bounds in added_bounds)
    # MVP check that output bound is correct at least
    c2a_cache = {}
    output_ub, out_alpha, added_bounds, c2a_cache = deep_poly(layer, alphas, lb, ub, c2a_cache)
    
    # Calculate the gradients of the upper bound w.r.t. the alphas
    output_ub.backward()
    # Check that the gradients are correct (should be -(x1 - x2) for alpha2 and 0 for alpha1, since it doesn't impact the output)
    correct_grads = torch.tensor([0., 1.])
    assert torch.allclose(out_alpha.get(1).grad, correct_grads)


def test_deep_poly_fc_forward_with_min_initialization_not_crossing_negative():
    # Build a simple fully connected network that we can test by hand
    fc_net = FullyConnected(2, [2, 1], act='relu')
    # Set the weights and biases
    fc_net.layers[0].weight = torch.nn.Parameter(torch.tensor([[1., 1.], [1., -1.]]))
    fc_net.layers[0].bias = torch.nn.Parameter(torch.tensor([0., 0.]))
    # Layer 1 is ReLU, so we don't need to set weights and biases
    fc_net.layers[2].weight = torch.nn.Parameter(torch.tensor([[1., -1.]]))
    fc_net.layers[2].bias = torch.nn.Parameter(torch.tensor([-0.5]))
    # Needed for deep_poly implementation
    layer = fc_net.layers
    # Input "image" for which we have no ReLU crossing
    x = torch.tensor([-2., 3.])
    # Input region
    epsilon = 1.
    lb = x - epsilon
    ub = x + epsilon
    # We initialize the alphas for the ReLU layers
    alphas = 'min'

    # Check that the forward pass is correct
    # Check that added_bounds[:, -2:] is correct for all layers of the network (the concrete upper & lower bounds in added_bounds)
    # MVP check that output bound is correct at least
    c2a_cache = {}
    output_ub, out_alpha, added_bounds, c2a_cache = deep_poly(layer, alphas, lb, ub, c2a_cache)
    # Build the correct bounds, it's always [lb_node1, lb_node2], [ub_node1, ub_node2], etc.
    correct_bounds_per_layer = [(torch.tensor([-1., -7.]), torch.tensor([3., -3.])), (torch.tensor([-1., 0.]), torch.tensor([3., 0.])), (torch.tensor([-1.5]), torch.tensor([2.5]))]
    # Check that the concrete bounds are correct for all layers
    for bounds, correct_bounds in zip(added_bounds, correct_bounds_per_layer):
        bounds = bounds[-2:]
        lb, ub, correct_lb, correct_ub = bounds[0], bounds[1], correct_bounds[0], correct_bounds[1]
        assert torch.allclose(lb, correct_lb)
        assert torch.allclose(ub, correct_ub)
    # Check that the initialized alphas are correct as well
    # Currently, alpha values get initialized even when the ReLU is not crossed, with the typical rule
    correct_alphas = torch.tensor([1., 0.])
    assert torch.allclose(out_alpha.get(1), correct_alphas)


def test_deep_poly_fc_backward_gradient_not_crossing_negative():
    # Build a simple fully connected network that we can test by hand
    fc_net = FullyConnected(2, [2, 1], act='relu')
    # Set the weights and biases
    fc_net.layers[0].weight = torch.nn.Parameter(torch.tensor([[1., 1.], [1., -1.]]))
    fc_net.layers[0].bias = torch.nn.Parameter(torch.tensor([0., 0.]))
    # Layer 1 is ReLU, so we don't need to set weights and biases
    fc_net.layers[2].weight = torch.nn.Parameter(torch.tensor([[1., -1.]]))
    fc_net.layers[2].bias = torch.nn.Parameter(torch.tensor([-0.5]))
    # Needed for deep_poly implementation
    layer = fc_net.layers
    # Input "image"
    x = torch.tensor([-2., 3.])
    # Input region
    epsilon = 1.
    lb = x - epsilon
    ub = x + epsilon
    # We initialize the alphas for the ReLU layers
    alphas = 'min'

    # Check that the forward pass is correct
    # Check that added_bounds[:, -2:] is correct for all layers of the network (the concrete upper & lower bounds in added_bounds)
    # MVP check that output bound is correct at least
    c2a_cache = {}
    output_ub, out_alpha, added_bounds, c2a_cache = deep_poly(layer, alphas, lb, ub, c2a_cache)
    
    # Calculate the gradients of the upper bound w.r.t. the alphas
    output_ub.backward()
    # Check that the gradients are correct (should be zero for both)
    correct_grads = torch.tensor([0., 0.])
    assert torch.allclose(out_alpha.get(1).grad, correct_grads)


def test_deep_poly_conv_without_batch_norm():
    # For now, use stride 1, since there may be some issues with stride > 1 and the current implementation
    # Include a single conv layer, out_channel = 1, kernel_size = 2, stride = 1, padding = 1
    # After that a fc layer to 2 neurons, a relu layer and a single output layer
    # Padding is by default 0
    conv_net = Conv(input_size=2, input_channels=1, conv_layers=[(1, 3, 1, 1)], fc_layers=[2, 1], n_class=10)
    # Set the weights and biases
    conv_net.layers[0].weight = torch.nn.Parameter(torch.tensor([[[[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]]]))
    conv_net.layers[0].bias = torch.nn.Parameter(torch.tensor([0.]))
    conv_net.layers[3].weight = torch.nn.Parameter(torch.tensor([[1., -1., 0., 0.], [0., 0., -1., 1.]]))
    conv_net.layers[3].bias = torch.nn.Parameter(torch.tensor([-0.5, 0.]))
    conv_net.layers[5].weight = torch.nn.Parameter(torch.tensor([[-1., 1.]]))
    conv_net.layers[5].bias = torch.nn.Parameter(torch.tensor([0.5]))
    layer = conv_net.layers
    # Input "image"
    x = torch.tensor([[[[-4., 0.], 
                        [ 1., 4.]]]])
    # Input region
    epsilon = 1.
    lb = x - epsilon
    ub = x + epsilon
    # We initialize the alphas for the ReLU layers
    alphas = {1: torch.tensor([0.5, 0.5, 0.5, 0.5]).requires_grad_(), 4: torch.tensor([0.5, 0.5]).requires_grad_()}

    # Check that the forward pass is correct
    # Check that added_bounds[:, -2:] is correct for all layers of the network (the concrete upper & lower bounds in added_bounds)
    # MVP check that output bound is correct at least
    c2a_cache = {}
    output_ub, out_alpha, added_bounds, c2a_cache = deep_poly(layer, alphas, lb, ub, c2a_cache)
    # Build the correct bounds, it's always [lb_node1, lb_node2], [ub_node1, ub_node2], etc.
    correct_bounds_per_layer = [(torch.tensor([-6., -3., -2., 2.]), torch.tensor([0., 3., 4., 8.])), 
                                (torch.tensor([0., -1.5, -1., 2.]), torch.tensor([0., 3., 4., 8.])), 
                                (torch.tensor([-7/2, 2/3]), torch.tensor([1., 7.])),
                                (torch.tensor([-7/4., 2/3.]), torch.tensor([1., 7.])),
                                (torch.tensor([7/18.]), torch.tensor([35/4.]))]
    # Check that the concrete bounds are correct for all layers
    for bounds, correct_bounds in zip(added_bounds, correct_bounds_per_layer):
        bounds = bounds[-2:]
        lb, ub, correct_lb, correct_ub = bounds[0], bounds[1], correct_bounds[0], correct_bounds[1]
        assert torch.allclose(lb, correct_lb)
        assert torch.allclose(ub, correct_ub)
    
    # Calculate the gradients of the upper bound w.r.t. the alphas
    output_ub.backward()
    # Check that the gradients are correct
    correct_grads_1 = torch.tensor([0., 0., -2., 0.])
    correct_grads_4 = torch.tensor([2.5, 0.])
    assert torch.allclose(out_alpha.get(1).grad, correct_grads_1)
    assert torch.allclose(out_alpha.get(4).grad, correct_grads_4)
    


def test_deep_poly_resnet_cancellation_between_and_within_paths():
    net = ResNet(BasicBlock, in_dim=2, num_stages=1, in_ch=1, num_blocks=1, num_classes=1, in_planes=1, bn=False, stride=[1, 1], last_layer="dense")
    flat_net = Sequential(*itertools.chain.from_iterable([(layer if type(layer) is Sequential else [layer]) for layer in net]))
    
    # Set the weights and biases
    flat_net[0].weight = torch.nn.Parameter(torch.tensor([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]]]))
    flat_net[0].bias = torch.nn.Parameter(torch.tensor([0.]))
    # Basic block
    flat_net[2].path_b[0].weight = torch.nn.Parameter(torch.tensor([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]]]))
    flat_net[2].path_b[0].bias = torch.nn.Parameter(torch.tensor([0.]))
    flat_net[2].path_b[2].weight = torch.nn.Parameter(torch.tensor([[[[0., 1., 0.], [1., -1., 1.], [0., 1., 0.]]]]))
    flat_net[2].path_b[2].bias = torch.nn.Parameter(torch.tensor([0.]))

    flat_net[5].weight = torch.nn.Parameter(torch.tensor([[1., -1., 0., 0.], [0., 0., -1., 1.]]))
    flat_net[5].bias = torch.nn.Parameter(torch.tensor([0., 0.]))
    flat_net[7].weight = torch.nn.Parameter(torch.tensor([[-1., 1.]]))
    flat_net[7].bias = torch.nn.Parameter(torch.tensor([0.]))
    
    # Input "image"
    x = torch.tensor([[[[0., 0.], 
                        [0., 0.]]]])
    # Input region
    epsilon = 1.
    lb = x - epsilon
    ub = x + epsilon
    # We initialize the alphas for the ReLU layers
    alphas = {1: torch.tensor([0.25, 0.25, 0.25, 0.25]).requires_grad_(), 
              '2b1': torch.tensor([0.5, 0.5, 0.5, 0.5]).requires_grad_(),
              3: torch.tensor([0.75, 0.75, 0.75, 0.75]).requires_grad_(),
              6: torch.tensor([0.9, 0.9]).requires_grad_()}
    # alphas = 'min'

    # Check that the forward pass is correct
    # Check that added_bounds[:, -2:] is correct for all layers of the network (the concrete upper & lower bounds in added_bounds)
    # MVP check that output bound is correct at least
    c2a_cache = {}
    output_ub, out_alpha, added_bounds, c2a_cache = deep_poly(flat_net, alphas, lb, ub, c2a_cache)
    # Build the correct bounds, it's always [lb_node1, lb_node2], [ub_node1, ub_node2], etc.
    correct_bounds_per_layer = [# Conv layer ---------------------------------------------------------------------------------
                                (torch.tensor([-1., -1., -1., -1.]), torch.tensor([1., 1., 1., 1.])),
                                # ReLU layer ---------------------------------------------------------------------------------
                                (torch.tensor([-1/4., -1/4., -1/4., -1/4.]), torch.tensor([1., 1., 1., 1.])),
                                # Basic block --------------------------------------------------------------------------------
                                # Path b of resnet ---------------------------------------------------------------------------
                                (torch.tensor([-1/4., -1/4., -1/4., -1/4.]), torch.tensor([1., 1., 1., 1.])),
                                (torch.tensor([-1/8., -1/8., -1/8., -1/8.]), torch.tensor([1., 1., 1., 1.])),
                                # (torch.tensor([-10/8., -10/8., -10/8., -10/8.]), torch.tensor([17/8., 17/8., 17/8., 17/8.])), Discrete bound not calculated anymore for speedup reasons
                                # --------------------------------------------------------------------------------------------
                                # Addition within resnet ---------------------------------------------------------------------
                                (torch.tensor([-1/2., -1/2., -1/2., -1/2.]), torch.tensor([5/2., 5/2., 5/2., 5/2.])),
                                # --------------------------------------------------------------------------------------------
                                # ReLU ---------------------------------------------------------------------------------------
                                (torch.tensor([-3/8., -3/8., -3/8., -3/8.]), torch.tensor([5/2., 5/2., 5/2., 5/2.])),
                                # --------------------------------------------------------------------------------------------
                                # FC layer -----------------------------------------------------------------------------------
                                (torch.tensor([-71/32., -71/32.]), torch.tensor([71/32., 71/32.]))]
    # Unroll the bounds given in added_bounds
    unrolled_bounds = []
    for bounds in added_bounds:
        if type(bounds) is dict:
            unrolled_bounds += [bound for bound in bounds['b'] if bound[-1] is not None]
            unrolled_bounds.append((bounds['lb'], bounds['ub']))
        else:
            unrolled_bounds.append(bounds)
    
    # Check that the concrete bounds are correct for all layers
    for bounds, correct_bounds in zip(unrolled_bounds, correct_bounds_per_layer):
        bounds = bounds[-2:]
        lb, ub, correct_lb, correct_ub = bounds[0], bounds[1], correct_bounds[0], correct_bounds[1]
        assert torch.allclose(lb, correct_lb)
        assert torch.allclose(ub, correct_ub)


def test_deep_poly_resnet_identity_path():
    net = ResNet(BasicBlock, in_dim=2, num_stages=1, in_ch=1, num_blocks=1, num_classes=1, in_planes=1, bn=False, stride=[1, 1], last_layer="dense")
    flat_net = Sequential(*itertools.chain.from_iterable([(layer if type(layer) is Sequential else [layer]) for layer in net]))
    
    # Set the weights and biases
    flat_net[0].weight = torch.nn.Parameter(torch.tensor([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]]]))
    flat_net[0].bias = torch.nn.Parameter(torch.tensor([0.]))
    # Basic block
    flat_net[2].path_b[0].weight = torch.nn.Parameter(torch.tensor([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]]]))
    flat_net[2].path_b[0].bias = torch.nn.Parameter(torch.tensor([-5.]))
    flat_net[2].path_b[2].weight = torch.nn.Parameter(torch.tensor([[[[0., 1., 0.], [1., -1., 1.], [0., 1., 0.]]]]))
    flat_net[2].path_b[2].bias = torch.nn.Parameter(torch.tensor([0.]))

    flat_net[5].weight = torch.nn.Parameter(torch.tensor([[1., -1., 0., 0.], [0., 0., -1., 1.]]))
    flat_net[5].bias = torch.nn.Parameter(torch.tensor([0., 0.]))
    flat_net[7].weight = torch.nn.Parameter(torch.tensor([[-1., 1.]]))
    flat_net[7].bias = torch.nn.Parameter(torch.tensor([0.]))
    
    # Input "image"
    x = torch.tensor([[[[0., 0.], 
                        [0., 0.]]]])
    # Input region
    epsilon = 1.
    lb = x - epsilon
    ub = x + epsilon
    # We initialize the alphas for the ReLU layers
    alphas = {1: torch.tensor([0.25, 0.25, 0.25, 0.25]).requires_grad_(), 
              '2b1': torch.tensor([0.5, 0.5, 0.5, 0.5]).requires_grad_(),
              3: torch.tensor([0.75, 0.75, 0.75, 0.75]).requires_grad_(),
              6: torch.tensor([0.9, 0.9]).requires_grad_()}
    # alphas = 'min'

    # Check that the forward pass is correct
    # Check that added_bounds[:, -2:] is correct for all layers of the network (the concrete upper & lower bounds in added_bounds)
    # MVP check that output bound is correct at least
    c2a_cache = {}
    output_ub, out_alpha, added_bounds, c2a_cache = deep_poly(flat_net, alphas, lb, ub, c2a_cache)
    # Build the correct bounds, it's always [lb_node1, lb_node2], [ub_node1, ub_node2], etc.
    correct_bounds_per_layer = [# Conv layer ---------------------------------------------------------------------------------
                                (torch.tensor([-1., -1., -1., -1.]), torch.tensor([1., 1., 1., 1.])),
                                # ReLU layer ---------------------------------------------------------------------------------
                                (torch.tensor([-1/4., -1/4., -1/4., -1/4.]), torch.tensor([1., 1., 1., 1.])),
                                # Basic block --------------------------------------------------------------------------------
                                # Path b of resnet ---------------------------------------------------------------------------
                                (torch.tensor([-21/4., -21/4., -21/4., -21/4.]), torch.tensor([-4., -4., -4., -4.])),
                                (torch.tensor([0., 0., 0., 0.]), torch.tensor([0., 0., 0., 0.])),
                                # (torch.tensor([0., 0., 0., 0.]), torch.tensor([0., 0., 0., 0.])),
                                # --------------------------------------------------------------------------------------------
                                # Addition within resnet ---------------------------------------------------------------------
                                (torch.tensor([-1/4., -1/4., -1/4., -1/4.]), torch.tensor([1., 1., 1., 1.])),
                                # --------------------------------------------------------------------------------------------
                                # ReLU ---------------------------------------------------------------------------------------
                                (torch.tensor([-3/16., -3/16., -3/16., -3/16.]), torch.tensor([1., 1., 1., 1.])),
                                # --------------------------------------------------------------------------------------------
                                # FC layer -----------------------------------------------------------------------------------
                                (torch.tensor([-19/16., -19/16.]), torch.tensor([19/16., 19/16.])),
                                (torch.tensor([-171/160., -171/160.]), torch.tensor([19/16., 19/16.])),
                                (torch.tensor([-361/160.]), torch.tensor([361/160.]))] 
    # Unroll the bounds given in added_bounds
    unrolled_bounds = []
    for bounds in added_bounds:
        if type(bounds) is dict:
            unrolled_bounds += [bound for bound in bounds['b'] if bound[-1] is not None]
            unrolled_bounds.append((bounds['lb'], bounds['ub']))
        else:
            unrolled_bounds.append(bounds)
    
    # Check that the concrete bounds are correct for all layers
    for bounds, correct_bounds in zip(unrolled_bounds, correct_bounds_per_layer):
        bounds = bounds[-2:]
        lb, ub, correct_lb, correct_ub = bounds[0], bounds[1], correct_bounds[0], correct_bounds[1]
        assert torch.allclose(lb, correct_lb)
        assert torch.allclose(ub, correct_ub)


def test_deep_poly_resnet_identity_path_identity_after():
    net = ResNet(BasicBlock, in_dim=2, num_stages=1, in_ch=1, num_blocks=1, num_classes=1, in_planes=1, bn=False, stride=[1, 1], last_layer="dense")
    flat_net = Sequential(*itertools.chain.from_iterable([(layer if type(layer) is Sequential else [layer]) for layer in net]))
    
    # Set the weights and biases
    flat_net[0].weight = torch.nn.Parameter(torch.tensor([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]]]))
    flat_net[0].bias = torch.nn.Parameter(torch.tensor([0.]))
    # Basic block
    flat_net[2].path_b[0].weight = torch.nn.Parameter(torch.tensor([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]]]))
    flat_net[2].path_b[0].bias = torch.nn.Parameter(torch.tensor([-5.]))
    flat_net[2].path_b[2].weight = torch.nn.Parameter(torch.tensor([[[[0., 1., 0.], [1., -1., 1.], [0., 1., 0.]]]]))
    flat_net[2].path_b[2].bias = torch.nn.Parameter(torch.tensor([0.]))

    flat_net[5].weight = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.], [0., 0., 0., 1.]]))
    flat_net[5].bias = torch.nn.Parameter(torch.tensor([0., 0.]))
    flat_net[7].weight = torch.nn.Parameter(torch.tensor([[-1., 1.]]))
    flat_net[7].bias = torch.nn.Parameter(torch.tensor([0.]))
    
    # Input "image"
    x = torch.tensor([[[[0., 0.], 
                        [0., 0.]]]])
    # Input region
    epsilon = 1.
    lb = x - epsilon
    ub = x + epsilon
    # We initialize the alphas for the ReLU layers
    alphas = {1: torch.tensor([0.25, 0.25, 0.25, 0.25]).requires_grad_(), 
              '2b1': torch.tensor([0.5, 0.5, 0.5, 0.5]).requires_grad_(),
              3: torch.tensor([0.75, 0.75, 0.75, 0.75]).requires_grad_(),
              6: torch.tensor([0.9, 0.9]).requires_grad_()}
    # alphas = 'min'

    # Check that the forward pass is correct
    # Check that added_bounds[:, -2:] is correct for all layers of the network (the concrete upper & lower bounds in added_bounds)
    # MVP check that output bound is correct at least
    c2a_cache = {}
    output_ub, out_alpha, added_bounds, c2a_cache = deep_poly(flat_net, alphas, lb, ub, c2a_cache)
    # Build the correct bounds, it's always [lb_node1, lb_node2], [ub_node1, ub_node2], etc.
    correct_bounds_per_layer = [# Conv layer ---------------------------------------------------------------------------------
                                (torch.tensor([-1., -1., -1., -1.]), torch.tensor([1., 1., 1., 1.])),
                                # ReLU layer ---------------------------------------------------------------------------------
                                (torch.tensor([-1/4., -1/4., -1/4., -1/4.]), torch.tensor([1., 1., 1., 1.])),
                                # Basic block --------------------------------------------------------------------------------
                                # Path b of resnet ---------------------------------------------------------------------------
                                (torch.tensor([-21/4., -21/4., -21/4., -21/4.]), torch.tensor([-4., -4., -4., -4.])),
                                (torch.tensor([0., 0., 0., 0.]), torch.tensor([0., 0., 0., 0.])),
                                # (torch.tensor([0., 0., 0., 0.]), torch.tensor([0., 0., 0., 0.])),
                                # --------------------------------------------------------------------------------------------
                                # Addition within resnet ---------------------------------------------------------------------
                                (torch.tensor([-1/4., -1/4., -1/4., -1/4.]), torch.tensor([1., 1., 1., 1.])),
                                # --------------------------------------------------------------------------------------------
                                # ReLU ---------------------------------------------------------------------------------------
                                (torch.tensor([-3/16., -3/16., -3/16., -3/16.]), torch.tensor([1., 1., 1., 1.])),
                                # --------------------------------------------------------------------------------------------
                                # FC layer -----------------------------------------------------------------------------------
                                (torch.tensor([-3/16., -3/16.]), torch.tensor([1., 1.])),
                                (torch.tensor([-27/160., -27/160.]), torch.tensor([1., 1.])),
                                (torch.tensor([-187/160.]), torch.tensor([187/160.]))] 
    # Unroll the bounds given in added_bounds
    unrolled_bounds = []
    for bounds in added_bounds:
        if type(bounds) is dict:
            unrolled_bounds += [bound for bound in bounds['b'] if bound[-1] is not None]
            unrolled_bounds.append((bounds['lb'], bounds['ub']))
        else:
            unrolled_bounds.append(bounds)
    
    # Check that the concrete bounds are correct for all layers
    for bounds, correct_bounds in zip(unrolled_bounds, correct_bounds_per_layer):
        bounds = bounds[-2:]
        lb, ub, correct_lb, correct_ub = bounds[0], bounds[1], correct_bounds[0], correct_bounds[1]
        assert torch.allclose(lb, correct_lb)
        assert torch.allclose(ub, correct_ub)


def test_resnet_flattening():
    # Load one of the resnets
    networks = ['net8', 'net9', 'net10']
    network_names = ['net8_cifar10_resnet_2b.pt', 'net9_cifar10_resnet_2b2_bn.pt', 'net10_cifar10_resnet_4b.pt']

    # Pick a random input image (CIFAR10)
    x = torch.rand(1, 3, 32, 32)

    for network, network_name in zip(networks, network_names):
        net = get_net(network, network_name)
        flat_net = Sequential(*itertools.chain.from_iterable([(layer if type(layer) is Sequential else [layer]) for layer in net.resnet]))
        assert torch.allclose(net.resnet(x), flat_net(x))
        

# def test_conv_caching_conflicts():
#     # load the different networks, run them on a random input and check the length of the cache
#     # should equal the number of conv layers in the network
#     networks = ['net1', 'net2', 'net3', 'net4', 'net5', 'net6', 'net7', 'net8', 'net9', 'net10']
#     network_names = ['net1_mnist_fc1.pt',
#                      'net2_mnist_fc2.pt',
#                      'net3_cifar10_fc3.pt',
#                      'net4_mnist_conv1.pt',
#                      'net5_mnist_conv2.pt',
#                      'net6_cifar10_conv2.pt',
#                      'net7_mnist_conv3.pt',
#                      'net8_cifar10_resnet_2b.pt',
#                      'net9_cifar10_resnet_2b2_bn.pt',
#                      'net10_cifar10_resnet_4b.pt']
#     for network, network_name in zip(networks, network_names):
#         net = get_net(network, network_name)
#         if network in ['net8', 'net9', 'net10']:
#             layers = Sequential(*itertools.chain.from_iterable([(layer if type(layer) is Sequential else [layer]) for layer in net.resnet]))
#         else:
#             layers = net.layers
#         c2a_cache = {}
#         if 'mnist' in network_name:
#             x = torch.rand(1, 1, 28, 28)
#         else:
#             x = torch.rand(1, 3, 32, 32)
#         epsilon = 0.
#         lb = x - epsilon
#         ub = x + epsilon
#         deep_poly(layers, 'min', lb, ub, c2a_cache)
#         num_conv_layer = len([layer for layer in net.modules() if type(layer) is Conv2d])
#         assert len(c2a_cache._cache) == num_conv_layer
    

def test_no_grad():
    # Load one of the networks
    # Change deep_poly to also take in no_grad True or false
    # Check that the gradients computed after deep_poly stay the same with no_grad True

    # load the different networks, run them on a random input and check the length of the cache
    # should equal the number of conv layers in the network
    networks = ['net1', 'net2', 'net3', 'net4', 'net5', 'net6', 'net7', 'net8', 'net9', 'net10']
    network_names = ['net1_mnist_fc1.pt',
                        'net2_mnist_fc2.pt',
                        'net3_cifar10_fc3.pt',
                        'net4_mnist_conv1.pt',
                        'net5_mnist_conv2.pt',
                        'net6_cifar10_conv2.pt',
                        'net7_mnist_conv3.pt',
                        'net8_cifar10_resnet_2b.pt',
                        'net9_cifar10_resnet_2b2_bn.pt',
                        'net10_cifar10_resnet_4b.pt']
    for network, network_name in zip(networks, network_names):
        net = get_net(network, network_name)
        if network in ['net8', 'net9', 'net10']:
            layers = Sequential(*itertools.chain.from_iterable([(layer if type(layer) is Sequential else [layer]) for layer in net.resnet]))
        else:
            layers = net.layers
        c2a_cache = {}
        if 'mnist' in network_name:
            x = torch.rand(1, 1, 28, 28)
        else:
            x = torch.rand(1, 3, 32, 32)
        epsilon = 0.
        lb = x - epsilon
        ub = x + epsilon
        output_ub_1, out_alpha_1, _, _ = deep_poly(layers, 'min', lb, ub, c2a_cache, no_grad=False)
        output_ub_2, out_alpha_2, _, _ = deep_poly(layers, 'min', lb, ub, c2a_cache, no_grad=True)
        # Somehow fails atm
        output_ub_1.backward()
        output_ub_2.backward()
        # Check that all the alpha gradients are the same
        
    assert False

# To enable debugging the test cases
def main():
    test_no_grad()

if __name__ == '__main__':
    main()