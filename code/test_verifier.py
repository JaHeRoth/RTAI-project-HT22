import pytest
import torch

from verifier import conv_to_affine
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
        - With batch_norm
        - Without batch_norm
    - For a resnet dummy network

"""


def test_conv_to_affine_without_bias_without_batch_norm():
    # Build a simple convolutional network with just one layer and no bias
    net = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False)
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


def test_deep_poly_fc():
    # Build a simple fully connected network that we can test by hand
    fc_net = FullyConnected(2, [2, 1], act='relu')
    # Set the weights and biases
    fc_net.layers[0].weight = torch.nn.Parameter(torch.tensor([[1., 1.], [1., -1.]]))
    fc_net.layers[0].bias = torch.nn.Parameter(torch.tensor([0., 0.]))
    # Layer 1 is ReLU, so we don't need to set weights and biases
    fc_net.layers[2].weight = torch.nn.Parameter(torch.tensor([[1., 1.]]))
    fc_net.layers[2].bias = torch.nn.Parameter(torch.tensor([-0.5]))

    # TODO: Check what the actual expected output of deep_poly is


def test_deep_poly_conv_with_batch_norm():
    # Is this even caluculatable by hand?
    assert False


def test_deep_poly_conv_without_batch_norm():
    # For now, use stride 1, since there may be some issues with stride > 1 and the current implementation
    # Include a single conv layer, out_channel = 1, kernel_size = 2, stride = 1, padding = 1
    # TODO: Sanity check that this is actually a valid network with the given parameters
    # After that a fc layer to 2 neurons, a relu layer and a single output layer
    conv_net = Conv(input_size=2, input_channels=1, conv_layers=[(1, 2, 1, 1)], fc_layers=[2, 1], n_class=10)
    # TODO: Set the weights and biases
    
    assert False


def test_deep_poly_resnet():
    # Also with fully connected network?
    # I mean we should use BasicBlocks to imitate the real networks
    assert False