import pytest
import torch

from verifier import conv_to_affine, get_net
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


# TODO: Include different strides in the tests
def test_conv_to_affine_without_bias_without_batch_norm():
    # Build a simple convolutional network with just one layer and no bias
    # Extract the layer from one of the official networks
    net = get_net('net6', 'net6_cifar10_conv2.pt')
    # net = ConvNet('cpu', 'cifar10', 32, 3, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10)
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
    assert torch.allclose(original_output, z)


def test_conv_to_affine_with_bias_without_batch_norm():
    # Build a simple convolutional network with just one layer
    net = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
    # Take a random input (shape of CIFAR-10 images)
    x = torch.randn(1, 3, 32, 32)
    # Run the convolutional layer
    y = net(x)
    # Convert the convolutional layer to affine
    bias, affine = conv_to_affine(net, 32, 32)
    # Run the affine layer
    z = affine @ x.view(-1) + bias
    # Check that the outputs are the same
    assert torch.allclose(y, z.view(y.shape))


def test_conv_to_affine_without_bias_with_batch_norm():
    # Build a simple convolutional network with just one layer and no bias but a batch normalization layer
    conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False)
    batch_norm = torch.nn.BatchNorm2d(3)
    net = torch.nn.Sequential(conv, batch_norm)
    # Take a random input (shape of CIFAR-10 images)
    x = torch.randn(1, 3, 32, 32)
    # Run the convolutional layer
    y = net(x)
    # Convert the convolutional layer to affine
    bias, affine = conv_to_affine(conv, 32, 32, batch_norm)
    # Run the affine layer
    z = affine @ x.view(-1) + bias
    # Check that the outputs are the same
    assert torch.allclose(y, z.view(y.shape))


def test_conv_to_affine_with_bias_with_batch_norm():
    # Build a simple convolutional network with just one layer
    conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
    batch_norm = torch.nn.BatchNorm2d(3)
    net = torch.nn.Sequential(conv, batch_norm)
    # Take a random input (shape of CIFAR-10 images)
    x = torch.randn(1, 3, 32, 32)
    # Run the convolutional layer
    y = net(x)
    # Convert the convolutional layer to affine
    bias, affine = conv_to_affine(conv, 32, 32, batch_norm)
    # Run the affine layer
    z = affine @ x.view(-1) + bias
    # Check that the outputs are the same
    assert torch.allclose(y, z.view(y.shape))


def test_deep_poly_fc_forward():
    # Build a simple fully connected network that we can test by hand
    fc_net = FullyConnected(2, [2, 1], act='relu')
    # Set the weights and biases
    fc_net.layers[0].weight = torch.nn.Parameter(torch.tensor([[1., 1.], [1., -1.]]))
    fc_net.layers[0].bias = torch.nn.Parameter(torch.tensor([0., 0.]))
    # Layer 1 is ReLU, so we don't need to set weights and biases
    fc_net.layers[2].weight = torch.nn.Parameter(torch.tensor([[1., 1.]]))
    fc_net.layers[2].bias = torch.nn.Parameter(torch.tensor([-0.5]))

    # Check that the forward pass is correct
    # Check that added_bounds[:, -2:] is correct for all layers (the concrete upper & lower bounds in added_bounds)
    # MVP check that output bound is correct
    alphas = 'min'
    

    # Check that the gradient after backprop is correct
    # 

    # TODO: Check what the actual expected output of deep_poly is

# Check that backprop behaves how we expected it to behave
def test_deep_poly_fc_backward_gradient():
    # Build a simple fully connected network that we can test by hand
    fc_net = FullyConnected(2, [2, 1], act='relu')
    # Set the weights and biases
    fc_net.layers[0].weight = torch.nn.Parameter(torch.tensor([[1., 1.], [1., -1.]]))
    fc_net.layers[0].bias = torch.nn.Parameter(torch.tensor([0., 0.]))
    # Layer 1 is ReLU, so we don't need to set weights and biases
    fc_net.layers[2].weight = torch.nn.Parameter(torch.tensor([[1., 1.]]))
    fc_net.layers[2].bias = torch.nn.Parameter(torch.tensor([-0.5]))

    # Check that the forward pass is correct
    # Check that added_bounds[:, -2:] is correct for all layers (the concrete upper & lower bounds in added_bounds)
    # MVP check that output bound is correct
    alphas = 'min'
    
    # Check that the gradient after backprop is correct
    # alphas.backward()
    # compare to calculation by hand




def test_deep_poly_conv_without_batch_norm():
    # For now, use stride 1, since there may be some issues with stride > 1 and the current implementation
    # Include a single conv layer, out_channel = 1, kernel_size = 2, stride = 1, padding = 1
    # TODO: Sanity check that this is actually a valid network with the given parameters
    # After that a fc layer to 2 neurons, a relu layer and a single output layer
    # Padding is by default 0
    conv_net = Conv(input_size=2, input_channels=1, conv_layers=[(1, 2, 1, 1)], fc_layers=[2, 1], n_class=10)
    # TODO: Set the weights and biases

    # Test the forward pass with deep_poly()
    
    assert False


def test_deep_poly_resnet():
    # Use ResNet directly (without the normalization layer)
    # Use simple BasicBlocks and no batch_norm
    # Iff there is time, use a conv layer and relu before basic block
    # 1 basic block, 1 path identity, 1 path conv without batch norm
    # Addition of the two paths, ReLU, FC layer to output

    assert False


def main():
    test_conv_to_affine_without_bias_without_batch_norm()

if __name__ == '__main__':
    main()