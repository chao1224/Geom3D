import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data


class BatchLEP(Data):
    def __init__(self, **kwargs):
        super(BatchLEP, self).__init__(**kwargs)
        return

    @staticmethod
    def from_data_list(data_list):
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))

        batch = BatchLEP()

        for key in keys:
            batch[key] = []
        batch.batch_active = []
        batch.batch_inactive = []

        cumsum_node_active = 0
        cumsum_node_inactive = 0

        for i, data in enumerate(data_list):
            num_nodes_active = data.x_active.size()[0]
            num_nodes_inactive = data.x_inactive.size()[0]
            batch.batch_active.append(torch.full((num_nodes_active,), i, dtype=torch.long))
            batch.batch_inactive.append(torch.full((num_nodes_inactive,), i, dtype=torch.long))
            
            for key in data.keys:
                item = data[key]
                # Only update the edge index
                if key in ['radius_edge_index_active', 'full_edge_index_active']:
                    item = item + cumsum_node_active
                if key in ['radius_edge_index_inactive', 'full_edge_index_active']:
                    item = item + cumsum_node_inactive
                batch[key].append(item)

            cumsum_node_active += num_nodes_active
            cumsum_node_inactive += num_nodes_inactive

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch_active = torch.cat(batch.batch_active, dim=-1)
        batch.batch_inactive = torch.cat(batch.batch_inactive, dim=-1)

        temp = (batch.positions_active.sum(1) == 0).sum()
        assert temp == 0
        temp = (batch.positions_inactive.sum(1) == 0).sum()
        assert temp == 0
        return batch.contiguous()
    
    @property
    def num_graphs(self):
        '''Returns the number of graphs in the batch.'''
        return self.batch[-1].item() + 1


class DataLoaderLEP(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderLEP, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchLEP.from_data_list(data_list),
            **kwargs)