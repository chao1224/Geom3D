import random
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader


class BatchPeriodicCrystal(Data):
    """A plain old python object modeling a batch of graphs as one big
    (disconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier."""

    def __init__(self, batch=None, **kwargs):
        super(BatchPeriodicCrystal, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        """Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchPeriodicCrystal()

        periodic_gathered = False  # image_expanded
        if "gathered_x" in keys:
            periodic_gathered = True  # image_gathered

        for key in keys:
            batch[key] = []
        batch.batch = []
        if periodic_gathered:
            batch.gathered_batch = []

        cumsum_node, cumsum_node_gathered, cumsum_edge = 0, 0, 0

        for i, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                if key in ['edge_index']:
                    item = item + cumsum_node
                elif key in ["periodic_index_mapping", "gathered_edge_index"]:
                    item = item + cumsum_node_gathered
                batch[key].append(item)

            num_nodes = data.x.size()[0]
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

            if periodic_gathered:
                num_nodes = data.gathered_x.size()[0]
                batch.gathered_batch.append(torch.full((num_nodes,), i, dtype=torch.long))
                cumsum_node_gathered += num_nodes

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        batch.batch = torch.cat(batch.batch, dim=-1)
        if periodic_gathered:
            batch.gathered_batch = torch.cat(batch.gathered_batch, dim=-1)

        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class DataLoaderPeriodicCrystal(DataLoader):
    """Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`) """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderPeriodicCrystal, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchPeriodicCrystal.from_data_list(data_list),
            **kwargs)