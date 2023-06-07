import torch
import torch.nn as nn
import torch.nn.functional as F


class AWARE(nn.Module):
    def __init__(self, emb_dim, r_prime, max_walk_len, num_layers, out_dim, use_bond=False):
        super(AWARE, self).__init__()
        self.r_prime = r_prime
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.max_walk_len = max_walk_len
        self.activation = nn.SiLU()
        self.softmax = nn.Softmax(dim=1)
        self.use_bond = use_bond
        ##
        self.Wv = nn.Linear(emb_dim, r_prime)
        self.Ww = nn.Linear(r_prime, r_prime)
        self.Wg = nn.Linear(r_prime, r_prime)
        ##
        self.target_model = []
        for i in range(self.num_layers + 1):
            if i == 0:
                self.target_model.append(nn.Linear(r_prime * max_walk_len, r_prime * max_walk_len))
                self.target_model.append(nn.SiLU())
            elif i != num_layers:
                self.target_model.append(nn.Linear(self.target_model[2 * i - 2].out_features, self.target_model[2 * i - 2].out_features // (2 ** (1 - i % 2))))
                self.target_model.append(nn.SiLU())
            else:
                self.target_model.append(nn.Linear(self.target_model[2 * i - 2].out_features, out_dim))
        self.target_model = nn.Sequential(*self.target_model)

        if self.use_bond:
            self.bond_mlp = nn.Sequential(
                nn.Linear(self.emb_dim, self.r_prime),
                nn.SiLU(),
                nn.Linear(self.r_prime, 1),
            )
        print('MODEL:\n', self)
        print()
        return

    def forward(self, node_attribute_matrix, adjacent_matrix, adj_attr_matrix):
        F_1 = self.activation(self.Wv(node_attribute_matrix))  # (B, max_node, dim)

        F_n = F_1
        f_1 = torch.sum(self.activation(self.Wg(F_n)), dim=1)  # (B, dim)
        f_T = [f_1]

        for n in range(self.max_walk_len - 1):
            S = torch.bmm(self.Ww(F_n), torch.transpose(F_n, 1, 2))  # (B, max_node, max_node)
            masked_S = S.masked_fill(adjacent_matrix == 0, -1e8)
            A_S = self.softmax(masked_S)  # (B, max_node, max_node)
            if self.use_bond:
                bond_S = self.bond_mlp(adj_attr_matrix).squeeze(dim=3)  # (B, max_node, max_node)
                A_S = A_S + bond_S
            # Add self-loop
            F_n = (F_n + torch.bmm(A_S, F_n)) * F_1  # (B, max_node, dim)
            f_n = torch.sum(self.activation(self.Wg(F_n)), dim=1)  # (B, dim)
            f_T.append(f_n)
        f_T = F.normalize(torch.cat(f_T, dim=1))

        return self.target_model(f_T)
