'''
credit to https://github.com/chao1224/BioChemGNN_Dense/blob/master/src/models/DMPNN.py
credit to https://github.com/chao1224/BioChemGNN/blob/main/BioChemGNN/models/DMPNN.py
'''
from collections import *
from re import L
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


def get_revert_edge_index(num_edge):
    """
    Corresponding to this line: https://github.com/chao1224/3D_Benchmark_dev/blob/main/Geom3D/datasets/datasets_utils.py#L90-L92
    """
    l = []
    for i in range(int(num_edge / 2)):
        l.extend([i*2+1, i*2])
    return l


class DMPNN(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(DMPNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)
        
        self.W_input = nn.Linear(emb_dim*2, emb_dim, bias=False)
        self.W_hidden = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_output = nn.Linear(emb_dim*2, emb_dim)

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)

        b_from_a = out_node_index = edge_index[0]
        b_to_a = in_node_index = edge_index[1]
        message = torch.cat([x[b_to_a], edge_attr], dim=-1)
        message = self.W_input(message)

        num_nodes = len(x)
        num_edges = len(b_from_a)

        reverse_edge_index = torch.LongTensor(get_revert_edge_index(num_edges))

        for i in range(self.num_layer - 1):
            node_message = scatter_add(message, in_node_index, dim=0, dim_size=num_nodes)
            rev_edge_message = message[reverse_edge_index]
            message = node_message[b_from_a] - rev_edge_message
            message = self.W_hidden(message)
            message = self.batch_norms[i](message)
            message = F.dropout(F.relu(message), self.drop_ratio, training=self.training)

        node_message = scatter_add(message, in_node_index, dim=0, dim_size=num_nodes)
        node_representation = torch.cat([x, node_message], dim=1)
        node_representation = F.relu(self.W_output(node_representation))

        return node_representation
