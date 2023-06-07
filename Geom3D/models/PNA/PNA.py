'''
credit to https://github.com/lukecavabarrett/PNA/blob/master/models/pytorch_geometric/PNA.py
and https://github.com/wdimmy/GNN_Molecule_Retrieval/tree/main/PNA
'''
from typing import Optional, List, Dict

import torch
from torch import Tensor
from typing import Optional
import torch.nn as nn
from torch_geometric.typing import OptTensor
from typing import Dict
from torch_geometric.nn.conv import MessagePassing
from .aggregators import AGGREGATORS
from .scalers import SCALERS
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn import BatchNorm, global_mean_pool
from torch_geometric.utils import degree
from torch.nn import Sequential, ModuleList, Linear, ReLU
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class PNAConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 aggregators: List[str], scalers: List[str], deg: Tensor,
                 edge_dim: Optional[int] = None, towers: int = 1,
                 pre_layers: int = 1, post_layers: int = 1,
                 divide_input: bool = False, **kwargs):
        super(PNAConv, self).__init__(aggr=None, node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scale] for scale in scalers]
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        deg = deg.to(torch.float)
        total_no_vertices = deg.sum()
        bin_degrees = torch.arange(len(deg))
        self.avg_deg: Dict[str, float] = {
            'lin': ((bin_degrees * deg).sum() / total_no_vertices).item(),
            'log': (((bin_degrees + 1).log() * deg).sum() / total_no_vertices).item(),
            'exp': ((bin_degrees.exp() * deg).sum() / total_no_vertices).item(),
        }

        if self.edge_dim is not None:
            self.bond_encoder = BondEncoder(self.F_in)

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules))

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [Linear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for nn in self.pre_nns:
            reset(nn)
        for nn in self.post_nns:
            reset(nn)
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:

        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr: OptTensor) -> Tensor:

        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_attr = self.bond_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        outs = [aggr(inputs, index, dim_size) for aggr in self.aggregators]
        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype).view(-1, 1, 1)
        outs = [scaler(out, deg, self.avg_deg) for scaler in self.scalers]
        return torch.cat(outs, dim=-1)


class PNA(nn.Module):
    def __init__(self, num_layer, emb_dim, dropout_ratio, deg):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout_ratio = dropout_ratio
        self.edge_dim = emb_dim
        for _ in range(num_layer):
            conv = PNAConv(
                in_channels=emb_dim, out_channels=emb_dim, aggregators=aggregators, scalers=scalers,
                deg=deg, edge_dim=self.edge_dim)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(emb_dim))
        return

    def get_graph_representation(self, batch):
        x = self.node_emb(batch.x)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            h = F.relu(batch_norm(conv(x, batch.edge_index, None)))
            x = h + x # residual
            x = F.dropout(x, self.dropout_ratio, training=self.training)

        h_graph = global_mean_pool(x, batch.batch)
        return h_graph

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.atom_encoder(x)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            if self.edge_dim:
                h = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            else:
                h = F.relu(batch_norm(conv(x, edge_index, None)))
            x = h + x # residual
            x = F.dropout(x, self.dropout_ratio, training=self.training)
        return x
