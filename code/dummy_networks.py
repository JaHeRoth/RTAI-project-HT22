import torch
import torch.nn as nn
from resnet import ResNet, BasicBlock


class FullyConnected(nn.Module):

    def __init__(self, input_size, fc_layers, act='relu'):
        super(FullyConnected, self).__init__()

        layers = []
        prev_fc_size = input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                if act == 'relu':
                    layers += [nn.ReLU()]
                else:
                    assert False
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Conv(nn.Module):

    def __init__(self, input_size, input_channels, conv_layers, fc_layers, n_class=10):
        super(Conv, self).__init__()

        self.input_size = input_size
        self.n_class = n_class

        layers = []
        prev_channels = input_channels
        img_dim = input_size

        for n_channels, kernel_size, stride, padding in conv_layers:
            layers += [
                nn.Conv2d(prev_channels, n_channels, kernel_size, stride=stride, padding=padding),
                nn.ReLU(),
            ]
            prev_channels = n_channels
            # TODO: This might actually need changing for the dummy network
            img_dim = img_dim // stride
        layers += [nn.Flatten()]

        prev_fc_size = prev_channels * img_dim * img_dim
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [nn.ReLU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# Not sure if this was even used for the normal networks
class UnnormalizedResnet(nn.Module):

    def __init__(self, device, resnet):
        super(UnnormalizedResnet, self).__init__()
        
        self.resnet = resnet

    def forward(self, x):
        x = self.resnet(x)
        return x


    

