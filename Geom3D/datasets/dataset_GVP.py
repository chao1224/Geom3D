
'''
This is a protein Dataset specifically for GVP.
'''
import numpy as np
import math
import os.path as osp
import torch
from torch_geometric.data import Data, InMemoryDataset
import torch.nn.functional as F
import torch_cluster
from tqdm import tqdm


class DatasetGVP(InMemoryDataset):
    def __init__(
        self, root, dataset, transform=None, pre_transform=None, pre_filter=None,
        split='train', num_positional_embeddings=16, top_k=30, num_rbf=16
    ):
        self.split = split
        self.root = root
        self.preprocessed_dataset = dataset
        self.num_positional_embeddings = num_positional_embeddings
        self.top_k = top_k
        self.num_rbf = num_rbf
 
        super(DatasetGVP, self).__init__(root, transform, pre_transform, pre_filter)
        
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        return osp.join(self.root, "processed_GVP", self.split)

    @property
    def processed_file_names(self):
        return 'data.pt'

    def positional_embeddings_GVP(self, edge_index, 
                               num_embeddings=None,
                               period_range=[2, 1000], device=None):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _normalize(self, tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def _rbf(self, D, D_min=0., D_max=20., D_count=16, device=None):
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1).to(device)

        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def dihedrals_GVP(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self._normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = self._normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = self._normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def orientations_GVP(self, X):
        forward = self._normalize(X[1:] - X[:-1])
        backward = self._normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def sidechains_GVP(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = self._normalize(c - origin), self._normalize(n - origin)
        bisector = self._normalize(c + n)
        perp = self._normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec      

    def process(self):  
        print('Beginning Processing ...')
        data_list = []
        device = 'cpu'

        with torch.no_grad():
            for data in tqdm(self.preprocessed_dataset):
                coords = []
                for i in range(len(data.coords_n)):
                    coords.append([list(data.coords_n[i]), list(data.coords_ca[i]), list(data.coords_c[i])])
                
                coords = torch.tensor(coords, dtype=torch.float32)

                mask = torch.isfinite(coords.sum(dim=(1,2)))
                coords[~mask] = np.inf
                
                X_ca = coords[:, 1]
                edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k).to(device)
                
                pos_embeddings = self.positional_embeddings_GVP(edge_index, self.num_positional_embeddings, device=device)
                E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
                rbf = self._rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=device)

                dihedrals = self.dihedrals_GVP(coords)
                orientations = self.orientations_GVP(X_ca)
                sidechains = self.sidechains_GVP(coords)

                node_s = dihedrals
                node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
                edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
                edge_v = self._normalize(E_vectors).unsqueeze(-2)
                
                node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))

                data.edge_index = edge_index
                data.node_s = node_s
                data.node_v = node_v
                data.edge_s = edge_s
                data.edge_v = edge_v
                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return
