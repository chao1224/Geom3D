from collections import namedtuple
from itertools import product

import torch
import torch.nn.functional as F
from torch import einsum, nn

from .fibers import Fiber
from .TFN_utils import GConvSE3, GNormSE3
from .utils import get_basis


class TFN(nn.Module):
    def __init__(
        self,
        num_layers,
        atom_feature_size,
        num_channels,
        num_nlayers=1,
        num_degrees=4,
        edge_dim=4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.num_channels_out = num_channels * num_degrees
        self.edge_dim = edge_dim

        self.fibers = {
            "in": Fiber(1, atom_feature_size),
            "mid": Fiber(num_degrees, self.num_channels),
            "out": Fiber(1, self.num_channels_out),
        }

        block0 = []
        fin = self.fibers["in"]
        for i in range(self.num_layers - 1):
            block0.append(
                GConvSE3(
                    fin,
                    self.fibers["mid"],
                    self_interaction=True,
                    edge_dim=self.edge_dim,
                )
            )
            block0.append(GNormSE3(self.fibers["mid"], num_layers=self.num_nlayers))
            fin = self.fibers["mid"]
        block0.append(
            GConvSE3(
                self.fibers["mid"],
                self.fibers["out"],
                self_interaction=True,
                edge_dim=self.edge_dim,
            )
        )
        self.block0 = nn.ModuleList(block0)

        return

    def forward(self, x, positions, edge_index, edge_feat=None):
        # Compute equivariant weight basis from relative positions
        row, col = edge_index
        positions_diff = positions[row] - positions[col]
        radial = torch.sqrt(torch.sum((positions_diff) ** 2, 1).unsqueeze(1))
        basis = get_basis(cloned_d=positions_diff, max_degree=self.num_degrees - 1)

        x = x.unsqueeze(2)  # (N, one_hot dim+1, 1)
        h = {"0": x}
        for layer in self.block0:
            h = layer(
                h, edge_index=edge_index, edge_feat=edge_feat, r=radial, basis=basis
            )
        h = h["0"].squeeze()

        return h
