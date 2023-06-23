"""
credit to https://github.com/chao1224/n_gram_graph/blob/master/n_gram_graph/embedding/graph_embedding.py
"""
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


def qm9_random_customized_01(dataset, null_value=None, seed=0):
    num_mols = len(dataset)
    np.random.seed(seed)
    all_idx = np.random.permutation(num_mols)

    Nmols = 133885 - 3054
    Ntrain = 110000
    Nvalid = 10000
    Ntest = Nmols - (Ntrain + Nvalid)

    train_idx = all_idx[:Ntrain]
    valid_idx = all_idx[Ntrain : Ntrain + Nvalid]
    test_idx = all_idx[Ntrain + Nvalid :]

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_idx)
    return train_dataset, valid_dataset, test_dataset


def qm9_random_customized_02(dataset, null_value=None, seed=0):
    num_mols = len(dataset)
    np.random.seed(seed)
    all_idx = np.random.permutation(num_mols)

    Nmols = 133885 - 3054
    Ntrain = 100000
    Ntest = int(0.1 * Nmols)
    Nvalid = Nmols - (Ntrain + Ntest)

    train_idx = all_idx[:Ntrain]
    valid_idx = all_idx[Ntrain : Ntrain + Nvalid]
    test_idx = all_idx[Ntrain + Nvalid :]

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

    train_dataset = Subset(dataset, torch.tensor(train_idx))
    valid_dataset = Subset(dataset, torch.tensor(valid_idx))
    test_dataset = Subset(dataset, torch.tensor(test_idx))
    return train_dataset, valid_dataset, test_dataset


def split(args, dataset, data_root):
    if args.split == "customized_01" and "qm9" in args.dataset:
        train_dataset, valid_dataset, test_dataset = qm9_random_customized_01(
            dataset, null_value=0, seed=args.seed
        )
        print("customized random (01) on QM9")
    elif args.split == "customized_02" and "qm9" in args.dataset:
        train_dataset, valid_dataset, test_dataset = qm9_random_customized_02(
            dataset, null_value=0, seed=args.seed
        )
        print("customized random (02) on QM9")
    else:
        raise ValueError("Invalid split option on {}.".format(args.dataset))
    print(len(train_dataset), "\t", len(valid_dataset), "\t", len(test_dataset))
    return train_dataset, valid_dataset, test_dataset


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        # print("=== x", x.size())
        for i in range(x.shape[-1]):
            # temp = x[..., i:i+1]
            # print(i, "temp", temp.size())
            # print(self.atom_embedding_list[i])
            x_embedding += self.atom_embedding_list[i](x[..., i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[-1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[..., i])

        return bond_embedding   


def extract_data_dict(data_loader_list, emb_dim, use_bond=False, use_3D_coordinates=False, normalize=False):
    data_dict = {
        "node_attr_matrix_list": [],
        "adj_matrix_list": [],
        "adj_attr_matrix_list": [],
        "node_num_list": [],
        "label_list": [],

        "ngram_node_matrix_list": [],
        "ngram_graph_matrix_list": [],
    }
    atom_encoder = AtomEncoder(emb_dim)
    bond_encoder = BondEncoder(emb_dim)

    for data_loader in data_loader_list:
        for batch in tqdm(data_loader):
            node_attr_matrix, adj_matrix, adj_attr_matrix, node_num, label = batch
            data_dict["node_attr_matrix_list"].append(node_attr_matrix)
            data_dict["adj_matrix_list"].append(adj_matrix)
            data_dict["adj_attr_matrix_list"].append(adj_attr_matrix)
            data_dict["node_num_list"].append(node_num)
            data_dict["label_list"].append(label)

            adj_matrix = adj_matrix.float()
            tilde_node_attr_matrix = atom_encoder(node_attr_matrix)

            for data_id, node_num_ in enumerate((node_num)):
                for j in range(node_num_):
                    adj_matrix[data_id, j, j] = 1

            if use_bond:
                adj_attr_matrix = bond_encoder(adj_attr_matrix)
                adj_mask = adj_matrix.unsqueeze(dim=3)
                adj_attr_matrix = adj_attr_matrix * adj_mask

                for data_id, node_num_ in enumerate((node_num)):
                    for j in range(node_num_):
                        adj_attr_matrix[data_id, j, j, :] = 1
            else:
                adj_attr_matrix = 0


            node_mask = node_attr_matrix.sum(dim=2, keepdim=True)
            tilde_node_attr_matrix = tilde_node_attr_matrix * node_mask

            walk = tilde_node_attr_matrix  # (N, max_node_num, atom_attr)
            v1 = torch.sum(walk, dim=1)  # (N, atom_attr)
            
            def message_passing_along_walk(adj_matrix, walk, tilde_node_attr_matrix, adj_attr_matrix, use_bond):
                # equivalent to torch.bmm(adj_matrix, walk) * tilde_node_attr_matrix
                adj_matrix = adj_matrix.unsqueeze(dim=3)  # (N, max_node_num, max_node_num, 1)
                walk = walk.unsqueeze(dim=2)  # (N, max_node_num, 1, atom_attr)

                tilde_node_attr_matrix = tilde_node_attr_matrix.unsqueeze(dim=1)  # (N, max_node_num, 1, atom_attr)
                intermediate = adj_matrix * walk * tilde_node_attr_matrix # (N, max_node_num, max_node_num, atom_attr)

                if use_bond:
                    intermediate = intermediate * adj_attr_matrix

                walk = intermediate.sum(dim=1)
                return walk

            # walk = (torch.bmm(adj_matrix, walk)) * tilde_node_attr_matrix
            # v2 = torch.sum(walk, dim=1)
            # print("v2", v2)

            walk = message_passing_along_walk(adj_matrix=adj_matrix, walk=walk, tilde_node_attr_matrix=tilde_node_attr_matrix, adj_attr_matrix=adj_attr_matrix, use_bond=use_bond)
            v2 = torch.sum(walk, dim=1)
            if normalize:
                v2 = F.normalize(v2) * emb_dim

            walk = message_passing_along_walk(adj_matrix=adj_matrix, walk=walk, tilde_node_attr_matrix=tilde_node_attr_matrix, adj_attr_matrix=adj_attr_matrix, use_bond=use_bond)
            v3 = torch.sum(walk, dim=1)
            if normalize:
                v3 = F.normalize(v3) * emb_dim

            walk = message_passing_along_walk(adj_matrix=adj_matrix, walk=walk, tilde_node_attr_matrix=tilde_node_attr_matrix, adj_attr_matrix=adj_attr_matrix, use_bond=use_bond)
            v4 = torch.sum(walk, dim=1)
            if normalize:
                v4 = F.normalize(v4) * emb_dim

            walk = message_passing_along_walk(adj_matrix=adj_matrix, walk=walk, tilde_node_attr_matrix=tilde_node_attr_matrix, adj_attr_matrix=adj_attr_matrix, use_bond=use_bond)
            v5 = torch.sum(walk, dim=1)
            if normalize:
                v5 = F.normalize(v5) * emb_dim

            walk = message_passing_along_walk(adj_matrix=adj_matrix, walk=walk, tilde_node_attr_matrix=tilde_node_attr_matrix, adj_attr_matrix=adj_attr_matrix, use_bond=use_bond)
            v6 = torch.sum(walk, dim=1)
            if normalize:
                v6 = F.normalize(v6) * emb_dim

            embedded_graph_matrix = torch.cat([v1, v2, v3, v4, v5, v6], dim=1)
            data_dict["ngram_node_matrix_list"].append(tilde_node_attr_matrix.detach())
            data_dict["ngram_graph_matrix_list"].append(embedded_graph_matrix.detach())
    
    data_dict["node_attr_matrix_list"] = torch.cat(data_dict["node_attr_matrix_list"], dim=0)  # (N, max_node_num, atom_attr)
    data_dict["adj_matrix_list"] = torch.cat(data_dict["adj_matrix_list"], dim=0)  # (N, max_node_num, max_node_num)
    data_dict["adj_attr_matrix_list"] = torch.cat(data_dict["adj_attr_matrix_list"], dim=0)  # (N, max_node_num, max_node_num, bond_attr)
    data_dict["node_num_list"] = torch.cat(data_dict["node_num_list"], dim=0)  # (N,)
    data_dict["label_list"] = torch.cat(data_dict["label_list"], dim=0)  # (N, 12/13)
    data_dict["ngram_node_matrix_list"] = torch.cat(data_dict["ngram_node_matrix_list"], dim=0)  # (N, 12/13)
    data_dict["ngram_graph_matrix_list"] = torch.cat(data_dict["ngram_graph_matrix_list"], dim=0)  # (N, 12/13)

    for k, v in data_dict.items():
        data_dict[k] = v.numpy()
    return data_dict
