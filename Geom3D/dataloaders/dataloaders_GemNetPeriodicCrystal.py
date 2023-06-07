import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from Geom3D.datasets.dataset_GemNet_utils import get_id_data_list_for_material


class BatchGemNetPeriodicCrystal(Data):
    def __init__(self, **kwargs):
        super(BatchGemNetPeriodicCrystal, self).__init__(**kwargs)
        return

    @staticmethod
    def from_data_list(data_list, cutoff, int_cutoff, triplets_only):
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        
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

        batch = BatchGemNetPeriodicCrystal()

        periodic_gathered = False  # image_expanded
        if "gathered_x" in keys:
            periodic_gathered = True  # image_gathered

        for key in keys:
            batch[key] = []
        batch.batch = []
        if periodic_gathered:
            batch.gathered_batch = []

        cumsum_node_gathered = 0

        for i, data in enumerate(data_list):
            num_nodes = data.x.size()[0]
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ["periodic_index_mapping"]:
                    item = item + cumsum_node_gathered
                batch[key].append(item)

            if periodic_gathered:
                num_nodes = data.gathered_x.size()[0]
                batch.gathered_batch.append(torch.full((num_nodes,), i, dtype=torch.long))
                cumsum_node_gathered += num_nodes

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        batch.batch = torch.cat(batch.batch, dim=-1)
        if periodic_gathered:
            batch.gathered_batch = torch.cat(batch.gathered_batch, dim=-1)

        index_batch = get_id_data_list_for_material(data_list, index_keys=index_keys)

        for index_key in index_keys:
            batch[index_key] = torch.tensor(index_batch[index_key], dtype=torch.int64)

        return batch.contiguous()
    
    @property
    def num_graphs(self):
        '''Returns the number of graphs in the batch.'''
        return self.batch[-1].item() + 1


class DataLoaderGemNetPeriodicCrystal(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, cutoff, int_cutoff, triplets_only, **kwargs):
        super(DataLoaderGemNetPeriodicCrystal, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda data_list: BatchGemNetPeriodicCrystal.from_data_list(data_list, cutoff, int_cutoff, triplets_only),
            **kwargs)