import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import global_mean_pool

from Geom3D.models.GPS_layer import GPSLayer


class SANGraphHead(nn.Module):
    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()
        self.pooling_fun = global_mean_pool
        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch)
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = F.relu(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label


class GPSModel(torch.nn.Module):
    def __init__(
        self, dim_in, num_tasks,
        gt_layers=5, gt_dim_hidden=300, gt_n_heads=4, gt_dropout=0, gt_attn_dropout=0.5,
        gt_layer_norm=False, gt_batch_norm=True
    ):
        super().__init__()
        self.atom_encoder = AtomEncoder(dim_in)
        self.bond_encoder = BondEncoder(dim_in)

        local_gnn_type = 'GENConv'
        global_model_type = 'Transformer'

        layers = []
        for _ in range(gt_layers):
            layers.append(GPSLayer(
                dim_h=gt_dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=gt_n_heads,
                equivstable_pe=False,
                dropout=gt_dropout,
                attn_dropout=gt_attn_dropout,
                layer_norm=gt_layer_norm,
                batch_norm=gt_batch_norm,
            ))
        self.layers = torch.nn.Sequential(*layers)

        self.post_mp = SANGraphHead(dim_in=dim_in, dim_out=num_tasks)
        return

    def forward(self, batch):
        batch.x = self.atom_encoder(batch.x)
        batch.edge_attr = self.bond_encoder(batch.edge_attr)
        batch = self.layers(batch)
        batch = self.post_mp(batch)
        return batch


if __name__ == "__main__":
    model = GPSModel(dim_in=300, num_tasks=1)
    print(model)