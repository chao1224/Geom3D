import itertools
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader


class AtomTupleExtractor:
    def __init__(self, ratio=1, option="permutation"):
        self.ratio = ratio
        self.option = option
        return

    def __call__(self, data):
        N = len(data.x)
        super_edge_index = []
        if N >= 2:
            if self.option == "permutation":
                super_edge_index = list(itertools.permutations(np.arange(N), 2))
            else:
                super_edge_index = list(itertools.combinations(np.arange(N), 2))
            super_edge_index = np.array(super_edge_index).T

            if self.ratio < 1:
                M = super_edge_index.shape[1]
                sampled_M = int(M * self.ratio)
                sampled_edge_index = np.random.choice(M, sampled_M, replace=False)
                super_edge_index = super_edge_index[:, sampled_edge_index]
  
            super_edge_index = torch.tensor(super_edge_index, dtype=torch.long)
        else:
            super_edge_index = torch.empty((2, 0), dtype=torch.long)

        data.super_edge_index = super_edge_index
    
        return data


class BatchAtomTuple(Data):
    def __init__(self, batch=None, **kwargs):
        super(BatchAtomTuple, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchAtomTuple()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0

        for i, data in enumerate(data_list):
            num_nodes = data.x.size()[0]
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'radius_edge_index', 'super_edge_index']:
                    item = item + cumsum_node
                batch[key].append(item)

            cumsum_node += num_nodes

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class DataLoaderAtomTuple(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderAtomTuple, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchAtomTuple.from_data_list(data_list),
            **kwargs)