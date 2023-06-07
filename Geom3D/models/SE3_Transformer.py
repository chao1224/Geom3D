import torch
import torch.nn.functional as F
from torch import einsum, nn

from .fibers import Fiber
from .SE3_Transformer_utils import GSE3Res
from .TFN_utils import GConvSE3, GNormSE3
from .utils import get_basis


class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""

    def __init__(
        self,
        num_layers,
        atom_feature_size,
        num_channels,
        num_nlayers=1,
        num_degrees=4,
        edge_dim=4,
        div=4,
        n_heads=1,
    ):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.n_heads = n_heads

        self.fibers = {
            "in": Fiber(1, atom_feature_size),
            "mid": Fiber(num_degrees, self.num_channels),
            "out": Fiber(1, num_degrees * self.num_channels),
        }

        # Equivariant layers
        layers = []
        fin = self.fibers["in"]
        for i in range(self.num_layers):
            layers.append(
                GSE3Res(
                    fin,
                    self.fibers["mid"],
                    edge_dim=self.edge_dim,
                    div=self.div,
                    n_heads=self.n_heads,
                )
            )
            layers.append(GNormSE3(self.fibers["mid"]))
            fin = self.fibers["mid"]
        layers.append(
            GConvSE3(
                self.fibers["mid"],
                self.fibers["out"],
                self_interaction=True,
                edge_dim=self.edge_dim,
            )
        )
        self.layers = nn.ModuleList(layers)
        return

    def forward(self, x, positions, edge_index, edge_feat=None):
        # Compute equivariant weight basis from relative positions
        row, col = edge_index
        positions_diff = positions[row] - positions[col]
        radial = torch.sqrt(torch.sum((positions_diff) ** 2, 1).unsqueeze(1))
        basis = get_basis(cloned_d=positions_diff, max_degree=self.num_degrees - 1)

        x = x.unsqueeze(2)  # (N, one_hot dim+1, 1)
        h = {"0": x}
        for layer in self.layers:
            h = layer(
                h,
                edge_index=edge_index,
                edge_feat=edge_feat,
                positions=positions,
                r=radial,
                basis=basis,
            )

        h = h["0"].squeeze()

        return h
