import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GlobalAttention, MessagePassing, Set2Set,
                                global_add_pool, global_max_pool,
                                global_mean_pool)
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_scatter import scatter_add
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)    
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
    
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out


class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__(node_dim=0)
        self.aggr = aggr
        self.heads = heads
        self.emb_dim = emb_dim
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(emb_dim, heads * emb_dim)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = nn.Parameter(torch.Tensor(emb_dim))
        self.bond_encoder = BondEncoder(emb_dim=heads * emb_dim)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)

        x = self.weight_linear(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

    def message(self, edge_index, x_i, x_j, edge_attr):
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)
        
    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out += self.bias
        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        self.aggr = aggr


    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "GIN":
                self.gnns.append(GINConv(emb_dim))
            elif gnn_type == "GCN":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "GAT":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "GraphSAGE":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.atom_encoder(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GNN_graphpred(nn.Module):
    def __init__(self, args, num_tasks, molecule_model=None):
        super(GNN_graphpred, self).__init__()
        self.num_layer = args.num_layer
        self.emb_dim = args.emb_dim
        self.JK = args.JK
        graph_pooling = args.graph_pooling
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.molecule_model = molecule_model

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = nn.Linear(
                self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks
            )
        else:
            self.graph_pred_linear = nn.Linear(
                self.mult * self.emb_dim, self.num_tasks
            )
        return

    def from_pretrained(self, model_file):
        self.molecule_model.load_state_dict(torch.load(model_file))
        return

    def get_graph_representation(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)

        graph_representation = self.pool(node_representation, batch)
        pred = self.graph_pred_linear(graph_representation)

        return graph_representation, pred

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)

        output = self.graph_pred_linear(graph_representation)

        return output
