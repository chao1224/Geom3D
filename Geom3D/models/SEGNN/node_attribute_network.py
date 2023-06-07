from torch_geometric.nn import MessagePassing


class NodeAttributeNetwork(MessagePassing):
    """
        Computes the node and edge attributes based on relative positions
    """

    def __init__(self):
        super(NodeAttributeNetwork, self).__init__(node_dim=-2, aggr="mean")  # <---- Mean of all edge features

    def forward(self, edge_index, edge_attr):
        """ Simply sums the edge attributes """
        node_attr = self.propagate(edge_index, edge_attr=edge_attr)  # TODO: continue here!
        return node_attr

    def message(self, edge_attr):
        """ The message is the edge attribute """
        return edge_attr

    def update(self, node_attr):
        """ The input to update is the aggregated messages, and thus the node attribute """
        return node_attr
