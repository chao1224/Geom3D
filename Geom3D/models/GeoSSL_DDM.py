import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

        self.reset_parameters()

    def reset_parameters(self):
        for i, layer in enumerate(self.layers):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, input):
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


class GeoSSL_DDM(torch.nn.Module):
    def __init__(self, emb_dim, sigma_begin, sigma_end, num_noise_level, noise_type, anneal_power):
        super(GeoSSL_DDM, self).__init__()

        self.anneal_power = anneal_power

        self.noise_type = noise_type
        self.input_distance_mlp = MultiLayerPerceptron(1, [emb_dim, 1], activation="relu")
        self.output_mlp = MultiLayerPerceptron(1+emb_dim, [emb_dim, emb_dim// 2, 1])

        sigmas = torch.tensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), num_noise_level)), dtype=torch.float32)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False) # (num_noise_level)

        return
    
    def forward(self, data, node_feature, distance):
        self.device = self.sigmas.device

        node2graph = data.batch
        edge2graph = node2graph[data.super_edge_index[0]]

        # sample noise level
        noise_level = torch.randint(0, self.sigmas.size(0), (data.num_graphs,), device=self.device) # (num_graph)
        used_sigmas = self.sigmas[noise_level] # (num_graph)
        used_sigmas = used_sigmas[edge2graph].unsqueeze(-1) # (num_edge, 1)

        distance_noise = torch.randn_like(distance)

        perturbed_distance = distance + distance_noise * used_sigmas
        distance_emb = self.input_distance_mlp(perturbed_distance) # (num_edge, hidden)

        target = -1 / (used_sigmas ** 2) * (perturbed_distance - distance) # (num_edge, 1)

        h_row, h_col = node_feature[data.super_edge_index[0]], node_feature[data.super_edge_index[1]] # (num_edge, hidden)

        distance_feature = torch.cat([h_row + h_col, distance_emb], dim=-1) # (num_edge, 2*hidden)
        scores = self.output_mlp(distance_feature) # (num_edge, 1)
        scores = scores * (1. / used_sigmas) # f_theta_sigma(x) =  f_theta(x) / sigma, (num_edge, 1)
 
        target = target.view(-1) # (num_edge)
        scores = scores.view(-1) # (num_edge)
        loss =  0.5 * ((scores - target) ** 2) * (used_sigmas.squeeze(-1) ** self.anneal_power) # (num_edge)
        loss = scatter_add(loss, edge2graph) # (num_graph)

        loss = loss.mean()
        return loss
