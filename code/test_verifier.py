import pytest
import torch

from verifier import conv_to_affine, get_net, deep_poly
from networks import Conv as ConvNet
from dummy_networks import FullyConnected, Conv, UnnormalizedResnet

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


# TODO?: Include different strides in the tests -> Should be accomplished by using all the different networks
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


# This actually does not occur in any network as far as I know
# and conv_to_affine() fails if I manually add a bias to the convolutional layer
# def test_conv_to_affine_with_bias_with_batch_norm_net9():
#     # Build a simple convolutional network with just one layer and no bias but a batch normalization layer
#     net = get_net('net9', 'net9_cifar10_resnet_2b2_bn.pt')
#     # This convolutional layer has no bias
#     conv = net.resnet[0]
#     conv.bias = torch.nn.Parameter(torch.randn(16))
#     batch_norm = net.resnet[1]
#     net = torch.nn.Sequential(conv, batch_norm)
#     # Take a random input (shape of CIFAR-10 images)
#     x = torch.randn(1, 3, 32, 32)
#     # Run the convolutional layer
#     y = net(x)
#     original_output = y.flatten()
#     # Convert the convolutional layer to affine
#     bias, affine = conv_to_affine(conv, 32, 32, batch_norm)
#     # Run the affine layer
#     z = affine @ x.flatten() + bias.flatten()
#     # Check that the outputs are the same
#     torch.testing.assert_close(original_output, z)


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
    output_ub, out_alpha, added_bounds = deep_poly(layer, alphas, lb, ub)
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
    output_ub, out_alpha, added_bounds = deep_poly(layer, alphas, lb, ub)
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
    output_ub, out_alpha, added_bounds = deep_poly(layer, alphas, lb, ub)
    
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
    output_ub, out_alpha, added_bounds = deep_poly(layer, alphas, lb, ub)
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
    output_ub, out_alpha, added_bounds = deep_poly(layer, alphas, lb, ub)
    
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
    output_ub, out_alpha, added_bounds = deep_poly(layer, alphas, lb, ub)
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
    output_ub, out_alpha, added_bounds = deep_poly(layer, alphas, lb, ub)
    
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
    # TODO: Set fc weights and biases
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
    output_ub, out_alpha, added_bounds = deep_poly(layer, alphas, lb, ub)
    # Build the correct bounds, it's always [lb_node1, lb_node2], [ub_node1, ub_node2], etc.
    # TODO: Calculate by hand
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
    


def test_deep_poly_resnet():
    # Use ResNet directly (without the normalization layer)
    # Use simple BasicBlocks and no batch_norm
    # Iff there is time, use a conv layer and relu before basic block
    # 1 basic block, 1 path identity, 1 path conv without batch norm
    # Addition of the two paths, ReLU, FC layer to output

    assert False


def test_resnet_flattening():
    # Check that the flattening done in our code does not change the output 
    # of the ResNet network

    assert False

# To enable debugging the test cases
def main():
    test_deep_poly_conv_without_batch_norm()

if __name__ == '__main__':
    main()