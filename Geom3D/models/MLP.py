from collections import *
from torch import nn


class MLP(nn.Module):
    def __init__(self, ECFP_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.ECFP_dim = ECFP_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        layer_dim = [self.ECFP_dim] + self.hidden_dim

        layers = OrderedDict()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(layer_dim[:-1], layer_dim[1:])):
            layers['fc layer {}'.format(layer_idx)] = nn.Linear(in_dim, out_dim)
            layers['relu {}'.format(layer_idx)] = nn.ReLU()
        self.represent_layers = nn.Sequential(layers)
        self.fc_layers = nn.Linear(layer_dim[-1], self.output_dim)
        return

    def represent(self, x):
        x = self.represent_layers(x)
        return x

    def forward(self, x):
        x = self.represent(x)
        x = self.fc_layers(x)
        return x