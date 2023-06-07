import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.nn import BatchNorm
from e3nn.o3 import Irreps, spherical_harmonics
from torch_geometric.nn import (MessagePassing, global_add_pool,
                                global_mean_pool)
from torch_geometric.data import Data

from .balanced_irreps import BalancedIrreps, WeightBalancedIrreps
from .instance_norm import InstanceNorm
from .node_attribute_network import NodeAttributeNetwork
from .o3_building_blocks import O3TensorProduct, O3TensorProductSwishGate


class SEGNNModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_features, N, norm, lmax_h, lmax_pos=None, pool="avg", edge_inference=False):
        super(SEGNNModel, self).__init__()
        self.num_classes = input_features

        if lmax_pos == None:
            lmax_pos = lmax_h
        self.pool = pool

        # Irreps for the node features (scalar type)
        node_in_irreps_scalar = Irreps("{0}x0e".format(input_features))         # This is the type of the input
        node_hidden_irreps_scalar = Irreps("{0}x0e".format(hidden_features))    # For the output layers
        node_out_irreps_scalar = Irreps("{0}x0e".format(output_features))       # This is the type on the output

        # Irreps for the edge and node attributes
        attr_irreps = Irreps.spherical_harmonics(lmax_pos)
        self.attr_irreps = attr_irreps

        # Irreps for the hidden activations (s.t. the nr of weights in the TP approx that of a standard linear layer)
        node_hidden_irreps = WeightBalancedIrreps(
            node_hidden_irreps_scalar, attr_irreps, True, lmax=lmax_h)  # True: copies of sh
        self.node_hidden_irreps = node_hidden_irreps
        # Network for computing the node attributes
        self.node_attribute_net = NodeAttributeNetwork()

        # The embedding layer (acts point-wise, no orientation information so only use trivial/scalar irreps)
        self.embedding_layer_1 = O3TensorProductSwishGate(node_in_irreps_scalar,  # in
                                                          node_hidden_irreps,     # out
                                                          attr_irreps)            # steerable attribute
        self.embedding_layer_2 = O3TensorProduct(node_hidden_irreps,  # in
                                                 node_hidden_irreps,  # out
                                                 attr_irreps)         # steerable attribute

        # The main layers
        self.layers = []
        for i in range(N):
            self.layers.append(SEGNN(node_hidden_irreps,  # in
                                     node_hidden_irreps,  # hidden
                                     node_hidden_irreps,  # out
                                     attr_irreps,         # steerable attribute
                                     norm=norm,
                                     edge_inference=edge_inference))
        self.layers = nn.ModuleList(self.layers)

        # The output network (again via point-wise operation via scalar irreps)
        self.head_pre_pool_layer_1 = O3TensorProductSwishGate(node_hidden_irreps,           # in
                                                              node_hidden_irreps_scalar,    # out
                                                              attr_irreps)                  # steerable attribute
        self.head_pre_pool_layer_2 = O3TensorProduct(node_hidden_irreps_scalar,
                                                     node_hidden_irreps_scalar)
        self.head_post_pool_layer_1 = O3TensorProductSwishGate(node_hidden_irreps_scalar,
                                                               node_hidden_irreps_scalar)
        self.head_post_pool_layer_2 = O3TensorProduct(node_hidden_irreps_scalar,
                                                      node_out_irreps_scalar)

    def forward(self, *argv):
        if len(argv) == 4:
            x, pos, edge_index, batch = argv[0], argv[1], argv[2], argv[3]
            graph = Data(x=x, pos=pos, radius_edge_index=edge_index, batch=batch)
        elif len(argv) == 1:
            graph = argv[0]
            x, pos, edge_index, batch = graph.x, graph.positions, graph.radius_edge_index, graph.batch

        if x.dim() > 1:
            x = x[:, 0]
        x = F.one_hot(x, num_classes=self.num_classes).float()

        # construct the node and edge attributes
        rel_pos = pos[edge_index[0]] - pos[edge_index[1]]  # pos_j - pos_i (note in edge_index stores tuples like (j,i))
        edge_dist = rel_pos.pow(2).sum(-1, keepdims=True)
        edge_attr = spherical_harmonics(self.attr_irreps, rel_pos, normalize=True, normalization='component')
        node_attr = self.node_attribute_net(edge_index, edge_attr)

        # A fix for isolated nodes (which are set to zero)
        graph.edge_index = graph.radius_edge_index
        edge_index_max = edge_index.max().item()
        num_nodes = graph.num_nodes
        assert num_nodes == x.size()[0]
        if (graph.has_isolated_nodes() and edge_index_max + 1 != num_nodes):
            nr_add_attr = num_nodes - (edge_index_max + 1)
            add_attr = node_attr.new_tensor(np.zeros((nr_add_attr, node_attr.shape[-1])))
            node_attr = torch.cat((node_attr, add_attr), -2)

        # Trivial irrep value should always be 1 (is automatically so for connected nodes, but isolated nodes are now 0)
        node_attr[:, 0] = 1.

        x = self.embedding_layer_1(x, node_attr)
        x = self.embedding_layer_2(x, node_attr)

        # The main layers
        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_dist, edge_attr, node_attr, batch)

        # Output head
        x = self.head_pre_pool_layer_1(x, node_attr)
        x = self.head_pre_pool_layer_2(x)

        # Pool over nodes
        if self.pool == "avg":
            x = global_mean_pool(x, batch)
        elif self.pool == "sum":
            x = global_add_pool(x, batch)

        x = self.head_post_pool_layer_1(x)
        x = self.head_post_pool_layer_2(x)
        return x

    def forward_with_gathered_index(self, x, pos, edge_index, batch, periodic_index_mapping, graph):
        
        if x.dim() > 1:
            x = x[:, 0]
        x = F.one_hot(x, num_classes=self.num_classes).float()

        row, col = edge_index
        gathered_row = periodic_index_mapping[row]
        gathered_col = periodic_index_mapping[col]
        edge_index = torch.stack([gathered_row, gathered_col])

        # construct the node and edge attributes
        rel_pos = pos[edge_index[0]] - pos[edge_index[1]]  # pos_j - pos_i (note in edge_index stores tuples like (j,i))
        edge_dist = rel_pos.pow(2).sum(-1, keepdims=True)
        edge_attr = spherical_harmonics(self.attr_irreps, rel_pos, normalize=True, normalization='component')
        node_attr = self.node_attribute_net(edge_index, edge_attr)

        # A fix for isolated nodes (which are set to zero)
        edge_index_max = edge_index.max().item()
        num_nodes = pos.size()[0]
        if (graph.has_isolated_nodes() and edge_index_max + 1 != num_nodes):
            nr_add_attr = num_nodes - (edge_index_max + 1)
            add_attr = node_attr.new_tensor(np.zeros((nr_add_attr, node_attr.shape[-1])))
            node_attr = torch.cat((node_attr, add_attr), -2)

        # Trivial irrep value should always be 1 (is automatically so for connected nodes, but isolated nodes are now 0)
        node_attr[:, 0] = 1.

        x = self.embedding_layer_1(x, node_attr)
        x = self.embedding_layer_2(x, node_attr)

        # The main layers
        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_dist, edge_attr, node_attr, batch)

        # Output head
        x = self.head_pre_pool_layer_1(x, node_attr)
        x = self.head_pre_pool_layer_2(x)

        # Pool over nodes
        if self.pool == "avg":
            x = global_mean_pool(x, batch)
        elif self.pool == "sum":
            x = global_add_pool(x, batch)

        x = self.head_post_pool_layer_1(x)
        x = self.head_post_pool_layer_2(x)
        return x


class SEGNN(MessagePassing):
    """
        E(3) equivariant message passing layer.
    """

    def __init__(self, node_in_irreps, node_hidden_irreps, node_out_irreps, attr_irreps, norm, edge_inference):
        super(SEGNN, self).__init__(node_dim=-2, aggr="add")

        self.norm = norm
        self.edge_inference = edge_inference

        # The message network layers
        irreps_message_in = (node_in_irreps + node_in_irreps + Irreps("1x0e")).simplify()
        self.message_layer_1 = O3TensorProductSwishGate(irreps_message_in,
                                                        node_hidden_irreps,
                                                        attr_irreps)
        self.message_layer_2 = O3TensorProductSwishGate(node_hidden_irreps,
                                                        node_hidden_irreps,
                                                        attr_irreps)

        # The node update layers
        irreps_update_in = (node_in_irreps + node_hidden_irreps).simplify()
        self.update_layer_1 = O3TensorProductSwishGate(irreps_update_in,
                                                       node_hidden_irreps,
                                                       attr_irreps)
        self.update_layer_2 = O3TensorProduct(node_hidden_irreps,
                                              node_out_irreps,
                                              attr_irreps)

        self.feature_norm = None
        self.message_norm = None

        if norm == "batch":
            self.feature_norm = BatchNorm(node_hidden_irreps)
            self.message_norm = BatchNorm(node_hidden_irreps)
        elif norm == "instance":
            self.feature_norm = InstanceNorm(node_hidden_irreps)

        if self.edge_inference:
            self.inference_layer = O3TensorProduct(node_hidden_irreps, Irreps("1x0e"), attr_irreps)

    def forward(self, x, pos, edge_index, edge_dist, edge_attr, node_attr, batch):
        """ Propagate messages along edges """
        x, pos = self.propagate(edge_index, x=x, pos=pos, edge_dist=edge_dist,
                                node_attr=node_attr, edge_attr=edge_attr)

        # Normalise features
        if self.feature_norm:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)

        return x, pos

    def message(self, x_i, x_j, edge_dist, edge_attr):
        """ Create messages """
        message = self.message_layer_1(torch.cat((x_i, x_j, edge_dist), dim=-1), edge_attr)
        message = self.message_layer_2(message, edge_attr)

        if self.message_norm:
            message = self.message_norm(message)
        if self.edge_inference:
            attention = torch.sigmoid(self.inference_layer(message, edge_attr))
            message = message*attention
        return message

    def update(self, message, x, pos, node_attr):
        """ Update note features """
        update = self.update_layer_1(torch.cat((x, message), dim=-1), node_attr)
        update = self.update_layer_2(update, node_attr)
        x += update  # Residual connection
        return x, pos
