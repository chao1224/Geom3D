import random

import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader

import networkx as nx
from Geom3D.datasets import graph_data_obj_to_nx_simple, nx_to_graph_data_obj_simple


def reset_idxes(G):
    """ Resets node indices such that they are numbered from 0 to num_nodes - 1
    :return: copy of G with relabelled node indices, mapping """
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping


class ExtractSubstructureContextPair:

    def __init__(self, k, l1, l2):
        """
        Randomly selects a node from the data object, and adds attributes
        that contain the substructure that corresponds to k hop neighbours
        rooted at the node, and the context substructures that corresponds to
        the subgraph that is between l1 and l2 hops away from the root node. """
        self.k = k
        self.l1 = l1
        self.l2 = l2

        # for the special case of 0, addresses the quirk with
        # single_source_shortest_path_length
        if self.k == 0:
            self.k = -1
        if self.l1 == 0:
            self.l1 = -1
        if self.l2 == 0:
            self.l2 = -1

    def __call__(self, data, root_idx=None):
        """
        :param data: pytorch geometric data object
        :param root_idx: If None, then randomly samples an atom idx.
        Otherwise sets atom idx of root (for debugging only)
        :return: None. Creates new attributes in original data object:
        data.center_substruct_idx
        data.x_substruct
        data.edge_attr_substruct
        data.edge_index_substruct
        data.x_context
        data.edge_attr_context
        data.edge_index_context
        data.overlap_context_substruct_idx """
        num_atoms = data.x.size()[0]
        if root_idx is None:
            root_idx = random.sample(range(num_atoms), 1)[0]

        G = graph_data_obj_to_nx_simple(data)  # same ordering as input data obj

        # Get k-hop subgraph rooted at specified atom idx
        substruct_node_idxes = nx.single_source_shortest_path_length(G, root_idx, self.k).keys()
        if len(substruct_node_idxes) > 0:
            substruct_G = G.subgraph(substruct_node_idxes)
            substruct_G, substruct_node_map = reset_idxes(substruct_G)  # need
            # to reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            substruct_data = nx_to_graph_data_obj_simple(substruct_G)
            data.x_substruct = substruct_data.x
            data.edge_attr_substruct = substruct_data.edge_attr
            data.edge_index_substruct = substruct_data.edge_index
            data.center_substruct_idx = torch.tensor([substruct_node_map[root_idx]])  # need
            # to convert center idx from original graph node ordering to the
            # new substruct node ordering

        # Get subgraphs that is between l1 and l2 hops away from the root node
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx, self.l1).keys()
        l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx, self.l2).keys()
        context_node_idxes = set(l1_node_idxes).symmetric_difference(set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)  # need to
            # reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            context_data = nx_to_graph_data_obj_simple(context_G)
            data.x_context = context_data.x
            data.edge_attr_context = context_data.edge_attr
            data.edge_index_context = context_data.edge_index

        # Get indices of overlapping nodes between substruct and context,
        # WRT context ordering
        context_substruct_overlap_idxes = list(set(context_node_idxes).intersection(
            set(substruct_node_idxes)))
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [
                context_node_map[old_idx]
                for old_idx in context_substruct_overlap_idxes]
            # need to convert the overlap node idxes, which is from the
            # original graph node ordering to the new context node ordering
            data.overlap_context_substruct_idx = \
                torch.tensor(context_substruct_overlap_idxes_reorder)

        return data

    def __repr__(self):
        return '{}(k={},l1={}, l2={})'.format(
            self.__class__.__name__, self.k, self.l1, self.l2)


class BatchSubstructContext3D(Data):
    """A plain old python object modeling a batch of graphs as one big
    (disconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier. """

    ''' Specialized batching for substructure context pair! '''

    def __init__(self, batch=None, **kwargs):
        super(BatchSubstructContext3D, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        """Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        original_keys = [set(data.keys) for data in data_list]
        original_keys = list(set.union(*original_keys))

        batch = BatchSubstructContext3D()
        keys = [
            'center_substruct_idx', 'edge_attr_substruct',
            'edge_index_substruct', 'x_substruct', 'overlap_context_substruct_idx',
            'edge_attr_context', 'edge_index_context', 'x_context',
            'positions', 'x', 'edge_attr', 'edge_index',  # these three are newly added to the non-3D version
        ]
        if 'radius_edge_index' in original_keys:
            keys.append('radius_edge_index')

        for key in keys:
            batch[key] = []

        batch.batch = []
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0
        for data in data_list:
            if hasattr(data, 'x_context'):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
                batch.batch_overlapped_context.append(torch.full((len(data.overlap_context_substruct_idx),), i, dtype=torch.long))
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                # batching for the main graph
                for key in ['x', 'edge_attr', 'edge_index', 'positions', 'radius_edge_index']:
                    if key not in data:
                        continue
                    item = data[key]
                    if key in ['edge_index', 'radius_edge_index']:
                       item = item + cumsum_main
                    batch[key].append(item)

                # batching for the substructure graph
                for key in ['center_substruct_idx', 'edge_attr_substruct', 'edge_index_substruct', 'x_substruct']:
                    item = data[key]
                    item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    batch[key].append(item)

                # batching for the context graph
                for key in ['overlap_context_substruct_idx', 'edge_attr_context', 'edge_index_context', 'x_context']:
                    item = data[key]
                    item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)

                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct
                cumsum_context += num_nodes_context
                i += 1
        
        for key in keys:
            batch[key] = torch.cat(batch[key], dim=batch.__cat_dim__(key))

        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)

        return batch.contiguous()

    def __cat_dim__(self, key):
        return -1 if key in ['edge_index', 'radius_edge_index', 'edge_index_substruct', 'edge_index_context'] else 0

    def cumsum(self, key, item):
        """If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute. """
        return key in [
            'edge_index', 'radius_edge_index',
            'edge_index_substruct', 'edge_index_context',
            'overlap_context_substruct_idx', 'center_substruct_idx']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class DataLoaderSubstructContext3D(DataLoader):
    """Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`) """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderSubstructContext3D, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchSubstructContext3D.from_data_list(data_list),
            **kwargs)
