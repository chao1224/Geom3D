import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from Geom3D.datasets.dataset_GemNet_utils import get_id_data_list


class BatchGemNetLEP(Data):
    def __init__(self, **kwargs):
        super(BatchGemNetLEP, self).__init__(**kwargs)
        return

    @staticmethod
    def from_data_list(data_list, cutoff, int_cutoff, triplets_only):
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        keys_active = []
        keys_inactive = []
        keys_common = []
        for key in keys:
            if "_active" in key:
                keys_active.append(key)
            elif "_inactive" in key:
                keys_inactive.append(key)
            else:
                # only label y
                keys_common.append(key)
        
        index_keys = [
            "id_undir", "id_swap",
            "id_c", "id_a",
            "id3_expand_ba", "id3_reduce_ca",
            "Kidx3",
        ]
        if not triplets_only:
            index_keys += [
                "id4_int_b", "id4_int_a",
                "id4_reduce_ca", "id4_expand_db", "id4_reduce_cab", "id4_expand_abd",
                "Kidx4",
                "id4_reduce_intm_ca", "id4_expand_intm_db", "id4_reduce_intm_ab", "id4_expand_intm_ab",
            ]

        batch_active = BatchGemNetLEP()
        batch_inactive = BatchGemNetLEP()

        # initialize keys
        for key in keys_active:
            batch_active[key.replace("_active", "")] = []
        for key in keys_inactive:
            batch_inactive[key.replace("_inactive", "")] = []
        for key in keys_common:
            batch_active[key] = []
            batch_inactive[key] = []
        batch_active.batch = []
        batch_inactive.batch = []

        # load key values from data_list
        for i, data in enumerate(data_list):
            num_nodes_active = data.x_active.size()[0]
            num_nodes_inactive = data.x_inactive.size()[0]
            batch_active.batch.append(torch.full((num_nodes_active,), i, dtype=torch.long))
            batch_inactive.batch.append(torch.full((num_nodes_inactive,), i, dtype=torch.long))
            
            for key in keys_active:
                item = data[key]
                batch_active[key.replace("_active", "")].append(item)
            for key in keys_inactive:
                item = data[key]
                batch_inactive[key.replace("_inactive", "")].append(item)
            for key in keys_common:
                item = data[key]
                batch_active[key].append(item)
                batch_inactive[key].append(item)

        for key_active in keys_active:
            key = key_active.replace("_active", "")
            batch_active[key] = torch.cat(batch_active[key], dim=data_list[0].__cat_dim__(key, batch_active[key][0]))
        for key_active in keys_inactive:
            key = key_active.replace("_inactive", "")
            batch_inactive[key] = torch.cat(batch_inactive[key], dim=data_list[0].__cat_dim__(key, batch_inactive[key][0]))
        for key in keys_common:
            batch_active[key] = torch.cat(batch_active[key], dim=data_list[0].__cat_dim__(key, batch_active[key][0]))
            batch_inactive[key] = torch.cat(batch_inactive[key], dim=data_list[0].__cat_dim__(key, batch_inactive[key][0]))
        batch_active.batch = torch.cat(batch_active.batch, dim=-1)
        batch_inactive.batch = torch.cat(batch_inactive.batch, dim=-1)

        # obtain values for GemNet
        data_list_active, data_list_inactive = [], []
        for data in data_list:
            # TODO: this is very specific, may generalize in the future
            data_list_active.append(Data(x=data.x_active, positions=data.positions_active))
            data_list_inactive.append(Data(x=data.x_inactive, positions=data.positions_inactive))
        index_batch_active = get_id_data_list(data_list_active, cutoff=cutoff, int_cutoff=int_cutoff, index_keys=index_keys, triplets_only=triplets_only)
        index_batch_inactive = get_id_data_list(data_list_inactive, cutoff=cutoff, int_cutoff=int_cutoff, index_keys=index_keys, triplets_only=triplets_only)

        for index_key in index_keys:
            batch_active[index_key] = torch.tensor(index_batch_active[index_key], dtype=torch.int64)
            batch_inactive[index_key] = torch.tensor(index_batch_inactive[index_key], dtype=torch.int64)

        batch = Data(batch_active=batch_active, batch_inactive=batch_inactive)
        return batch.contiguous()
    
    @property
    def num_graphs(self):
        '''Returns the number of graphs in the batch.'''
        return self.batch[-1].item() + 1


class DataLoaderGemNetLEP(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, cutoff, int_cutoff, triplets_only, **kwargs):
        super(DataLoaderGemNetLEP, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda data_list: BatchGemNetLEP.from_data_list(data_list, cutoff, int_cutoff, triplets_only),
            **kwargs)