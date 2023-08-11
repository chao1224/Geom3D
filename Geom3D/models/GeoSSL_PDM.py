"""
credit to https://github.com/jiaor17/3D-EMGP/blob/main/mgp/models/denoise_prednoise.py
We modify the pipeline to better fit the NCSN pipeline
"""
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max, scatter_min, scatter_mean
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch.autograd import grad


loss_func = {
    "L1" : nn.L1Loss(reduction='none'),
    "L2" : nn.MSELoss(reduction='none'),
    "Cosine" : nn.CosineSimilarity(dim=-1, eps=1e-08),
    "CrossEntropy" : nn.CrossEntropyLoss(reduction='none')
}


class GeoSSL_PDM(torch.nn.Module):
    def __init__(self, emb_dim, sigma_begin, sigma_end, num_noise_level, noise_type, anneal_power):
        super(GeoSSL_PDM, self).__init__()
        self.emb_dim = emb_dim
        self.noise_type = noise_type
        self.anneal_power = anneal_power

        self.noise_pred = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim))

        sigmas = torch.tensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), num_noise_level)), dtype=torch.float32)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False) # (num_noise_level)

    @staticmethod
    @torch.no_grad()
    def get_score_target(pos_perturbed, pos_target, node2graph, noise_type):
        # s = - (pos_perturbed @ (pos_perturbed.T @ pos_perturbed) - pos_target @ (pos_target.T @ pos_perturbed)) / (torch.norm(pos_perturbed.T @ pos_perturbed) + torch.norm(pos_target.T @ pos_perturbed))
        if noise_type == 'riemann':
            v = pos_target.shape[-1]
            center = scatter_mean(pos_target, node2graph, dim = -2) # B * 3
            perturbed_center = scatter_mean(pos_perturbed, node2graph, dim = -2) # B * 3
            pos_c = pos_target - center[node2graph]
            pos_perturbed_c = pos_perturbed - perturbed_center[node2graph]
            pos_perturbed_c_left = pos_perturbed_c.repeat_interleave(v,dim=-1)
            pos_perturbed_c_right = pos_perturbed_c.repeat([1,v])
            pos_c_left = pos_c.repeat_interleave(v,dim=-1)
            ptp = scatter_add(pos_perturbed_c_left * pos_perturbed_c_right, node2graph, dim = -2).reshape(-1,v,v) # B * 3 * 3     
            otp = scatter_add(pos_c_left * pos_perturbed_c_right, node2graph, dim = -2).reshape(-1,v,v) # B * 3 * 3     
            ptp = ptp[node2graph]
            otp = otp[node2graph]
            tar_force = - 2 * (pos_perturbed_c.unsqueeze(1) @ ptp - pos_c.unsqueeze(1) @ otp).squeeze(1) / (torch.norm(ptp,dim=(1,2)) + torch.norm(otp,dim=(1,2))).unsqueeze(-1).repeat([1,3])
            return tar_force
        else:
            return pos_target - pos_perturbed

    def forward(self, data, energy, molecule_repr, pos_noise_pred, pos_perturbed, pos_target, debug=False):
        self.device = self.sigmas.device

        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]
        
        noise_level = torch.randint(0, self.sigmas.size(0), (data.num_graphs,), device=self.device) # (num_graph)
        used_sigmas = self.sigmas[noise_level] # (num_graph)

        used_sigmas = used_sigmas[node2graph].unsqueeze(-1) # (num_nodes, 1)
        
        score_target = self.get_score_target(pos_perturbed, pos_target, node2graph, self.noise_type) / used_sigmas

        ##### node-level: score or force #####
        pred_noise = (pos_noise_pred - pos_perturbed) * (1. / used_sigmas)
        pos_denoise = loss_func['L2'](pred_noise, score_target)
        pos_denoise = torch.sum(pos_denoise, dim = -1)
        pos_denoise = scatter_add(pos_denoise, node2graph)
        pos_denoise = pos_denoise.mean()

        ##### graph-level: noise scale #####
        pred_scale = self.noise_pred(molecule_repr)
        loss_pred_noise = loss_func['CrossEntropy'](pred_scale, noise_level)
        pred_scale_ = pred_scale.argmax(dim=1)

        return pos_denoise, loss_pred_noise.mean()
