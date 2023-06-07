'''
credit to https://github.com/chao1224/BioChemGNN_Dense/blob/master/src/models/enn.py
'''
from collections import *
import torch
from torch import nn
from torch.nn import functional as F
from .molecule_gnn_model import GINConv
from torch_scatter import scatter_add, scatter_max
from ogb.graphproppred.mol_encoder import AtomEncoder


class GraphSoftmax(nn.Module):
    eps = 1e-10

    def forward(self, batch, input):
        batch_size = torch.max(batch).item() + 1
        x = input - scatter_max(input, batch, dim=0, dim_size=batch_size)[0][batch]
        x = x.exp()
        normalizer = scatter_add(x, batch, dim=0, dim_size=batch_size)[batch]
        return x / (normalizer + self.eps)



class Set2Set(nn.Module):
    def __init__(self, input_dim, processing_steps, num_layers):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim * 2
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.output_dim, self.input_dim, self.num_layers)
        self.softmax = GraphSoftmax()
        return

    def forward(self, x, batch):
        batch_size = torch.max(batch).item() + 1

        h = (torch.zeros(self.num_layers, batch_size, self.input_dim).to(device=x.device),
             torch.zeros(self.num_layers, batch_size, self.input_dim).to(device=x.device))
        q_star = torch.zeros(batch_size, self.output_dim).to(device=x.device)

        for _ in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.squeeze(0)
            
            product = torch.einsum("bd, bd -> b", q[batch], x)
            attention = self.softmax(batch, product)
            attenteion_output = scatter_add(attention.unsqueeze(-1) * x, batch, dim=0, dim_size=batch_size)  # n_graph, dim
            q_star = torch.cat([q, attenteion_output], dim=-1)

        q_star = q_star.squeeze(0)
        return q_star


class ENN_S2S(nn.Module):
    def __init__(
            self, hidden_dim, gru_layer_num, enn_layer_num,
            set2set_processing_steps, set2set_num_layers, output_dim
    ):
        super(ENN_S2S, self).__init__()

        self.hidden_dim = hidden_dim
        self.gru_layer_num = gru_layer_num
        self.enn_layer_num = enn_layer_num
        self.output_dim = output_dim

        self.atom_encoder = AtomEncoder(hidden_dim)

        self.enn_layer = GINConv(hidden_dim)
        self.gru_layer = nn.GRU(self.hidden_dim, self.hidden_dim, self.gru_layer_num)

        self.set2set = Set2Set(input_dim=self.hidden_dim, processing_steps=set2set_processing_steps, num_layers=set2set_num_layers)
        self.fc_layer = nn.Linear(self.hidden_dim*2, self.output_dim)
        return

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        x = self.atom_encoder(x)
        hx = x.repeat(self.gru_layer_num, 1, 1)

        for _ in range(self.enn_layer_num):
            x = self.enn_layer(x, edge_index, edge_attr)
            x, hx = self.gru_layer(x.unsqueeze(0), hx)
            x = x.squeeze(0)

        x = self.set2set(x, batch)
        x = self.fc_layer(x)
        return x
