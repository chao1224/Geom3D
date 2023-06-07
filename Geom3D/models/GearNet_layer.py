import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add
from torch_geometric.data import Data


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden


class IEConvLayer(nn.Module):
    eps = 1e-6

    def __init__(self, input_dim, hidden_dim, output_dim, edge_input_dim, kernel_hidden_dim=32,
                dropout=0.05, dropout_before_conv=0.2, activation="relu", aggregate_func="sum"):
        super(IEConvLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.kernel_hidden_dim = kernel_hidden_dim
        self.aggregate_func = aggregate_func

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.kernel = MultiLayerPerceptron(edge_input_dim, [kernel_hidden_dim, (hidden_dim + 1) * hidden_dim])
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.input_batch_norm = nn.BatchNorm1d(input_dim)
        self.message_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.update_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.output_batch_norm = nn.BatchNorm1d(output_dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout_before_conv = nn.Dropout(dropout_before_conv)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def message(self, graph, input, edge_input):
        node_in = graph.edge_list[:, 0]
        message = self.linear1(input[node_in])
        message = self.message_batch_norm(message)
        message = self.dropout_before_conv(self.activation(message))
        kernel = self.kernel(edge_input).view(-1, self.hidden_dim + 1, self.hidden_dim)
        message = torch.einsum('ijk, ik->ij', kernel[:, 1:, :], message) + kernel[:, 0, :]

        return message
    
    def aggregate(self, graph, message):
        node_in, node_out = graph.edge_list.t()[:2]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        
        if self.aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node) 
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)
        return update

    def combine(self, input, update):
        output = self.linear2(update)
        return output

    def forward(self, graph, input, edge_input):
        input = self.input_batch_norm(input)
        layer_input = self.dropout(self.activation(input))
        
        message = self.message(graph, layer_input, edge_input)
        update = self.aggregate(graph, message)
        update = self.dropout(self.activation(self.update_batch_norm(update)))
        
        output = self.combine(input, update)
        output = self.output_batch_norm(output)
        return output
    

class GeometricRelationalGraphConv(nn.Module):
    eps = 1e-6

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, 
                batch_norm=False, activation="relu"):
        super(GeometricRelationalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input, edge_input=None):
        node_in = graph.edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        if edge_input is not None:
            assert edge_input.shape == message.shape
            message += edge_input
        return message
    
    def aggregate(self, graph, message):
        assert graph.num_relation == self.num_relation

        node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation)
        update = update.view(graph.num_node, self.num_relation * self.input_dim)

        return update
    
    def combine(self, input, update):
        output = self.linear(update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
    
    def forward(self, graph, input, edge_input=None):
        message = self.message(graph, input, edge_input)
        update = self.aggregate(graph, message)
        output = self.combine(input, update)
        return output


class SpatialLineGraph(nn.Module):
    def __init__(self, num_angle_bin=8):
        super(SpatialLineGraph, self).__init__()
        self.num_angle_bin = num_angle_bin

    def forward(self, graph):
        """
        Generate the spatial line graph of the input graph.
        The edge types are decided by the angles between two adjacent edges in the input graph.

        Parameters:
            graph (PackedGraph): :math:`n` graph(s)

        Returns:
            graph (PackedGraph): the spatial line graph
        """
        line_graph = construct_line_graph(graph)
        node_in, node_out = graph.edge_list[:, :2].t()
        edge_in, edge_out = line_graph.edge_list.t()

        # compute the angle ijk
        node_i = node_out[edge_out]
        node_j = node_in[edge_out]
        node_k = node_in[edge_in]
        vector1 = graph.node_position[node_i] - graph.node_position[node_j]
        vector2 = graph.node_position[node_k] - graph.node_position[node_j]
        x = (vector1 * vector2).sum(dim=-1)
        y = torch.cross(vector1, vector2).norm(dim=-1)
        angle = torch.atan2(y, x)
        relation = (angle / math.pi * self.num_angle_bin).long()
        edge_list = torch.cat([line_graph.edge_list, relation.unsqueeze(-1)], dim=-1)

        return Data(
            edge_list=edge_list, edge_weight=line_graph.edge_weight, num_nodes=line_graph.num_nodes, offsets=line_graph.offsets,
            num_edges=line_graph.num_edges, num_relation=self.num_angle_bin, node_feature=line_graph.node_feature)


def _get_offsets(graph, num_nodes=None, num_edges=None, num_cum_nodes=None, num_cum_edges=None):
    if num_nodes is None:
        prepend = torch.tensor([0], device=graph.node_feature.device)
        num_nodes = torch.diff(num_cum_nodes, prepend=prepend)
    if num_edges is None:
        prepend = torch.tensor([0], device=graph.node_feature.device)
        num_edges = torch.diff(num_cum_edges, prepend=prepend)
    if num_cum_nodes is None:
        num_cum_nodes = num_nodes.cumsum(0)
    return (num_cum_nodes - num_nodes).repeat_interleave(num_edges)


def construct_line_graph(graph):
    """
    Construct a packed line graph of this packed graph.
    The node features of the line graphs are inherited from the edge features of the original graphs.

    In the line graph, each node corresponds to an edge in the original graph.
    For a pair of edges (a, b) and (b, c) that share the same intermediate node in the original graph,
    there is a directed edge (a, b) -> (b, c) in the line graph.

    Returns:
        PackedGraph
    """
    node_in, node_out = graph.edge_list.t()[:2]
    edge_index = torch.arange(graph.num_edge, device=graph.node_feature.device)
    edge_in = edge_index[node_out.argsort()]
    edge_out = edge_index[node_in.argsort()]

    degree_in = node_in.bincount(minlength=graph.num_node)
    degree_out = node_out.bincount(minlength=graph.num_node)
    size = degree_out * degree_in
    starts = (size.cumsum(0) - size).repeat_interleave(size)
    range = torch.arange(size.sum(), device=graph.node_feature.device)
    # each node u has degree_out[u] * degree_in[u] local edges
    local_index = range - starts
    local_inner_size = degree_in.repeat_interleave(size)
    edge_in_offset = (degree_out.cumsum(0) - degree_out).repeat_interleave(size)
    edge_out_offset = (degree_in.cumsum(0) - degree_in).repeat_interleave(size)
    edge_in_index = torch.div(local_index, local_inner_size, rounding_mode="floor") + edge_in_offset
    edge_out_index = local_index % local_inner_size + edge_out_offset

    edge_in = edge_in[edge_in_index]
    edge_out = edge_out[edge_out_index]
    edge_list = torch.stack([edge_in, edge_out], dim=-1)
    node_feature = getattr(graph, "edge_feature", None)
    num_nodes = graph.num_edges
    num_edges = scatter_add(size, graph.node2graph, dim=0, dim_size=graph.batch_size)
    offsets = _get_offsets(graph, num_nodes, num_edges)
    edge_weight = torch.ones(len(edge_list))

    return Data(
        edge_list=edge_list, edge_weight=edge_weight, num_nodes=num_nodes, num_edges=num_edges, offsets=offsets,
        node_feature=node_feature)