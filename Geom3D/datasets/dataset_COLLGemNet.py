import os
from tqdm import tqdm

import torch
from torch_geometric.data import InMemoryDataset
from Geom3D.datasets.dataset_GemNet_utils import get_id_data_single


class DatasetCOLLGemNet(InMemoryDataset):
    def __init__(
        self,
        root,
        preprcessed_dataset,
        cutoff,
        int_cutoff,
    ):
        self.root = root
        self.preprcessed_dataset = preprcessed_dataset
        self.mode = preprcessed_dataset.mode
        self.cutoff = cutoff
        self.int_cutoff = int_cutoff
        self.transform, self.pre_transform, self.pre_filter = preprcessed_dataset.transform, preprcessed_dataset.pre_transform, preprcessed_dataset.pre_filter

        self.index_keys = [
            "id_undir", "id_swap", "id_c", "id_a", "id3_expand_ba", "id3_reduce_ca",
            "Kidx3",
        ]
        self.index_keys += [
            "id4_int_b", "id4_int_a",
            "id4_reduce_ca", "id4_expand_db",
            "id4_expand_abd", "id4_reduce_cab",
            "Kidx4",
            "id4_reduce_intm_ab", "id4_expand_intm_ab",
            "id4_reduce_intm_ca", "id4_expand_intm_db",
        ]
        self.triplets_only = False # set to False for PyG dataset generation
        self.cutoff = cutoff
        self.int_cutoff = int_cutoff
        self.keys = ["x", "positions", "y", "force"]
        
        super(DatasetCOLLGemNet, self).__init__(root, self.transform, self.pre_transform, self.pre_filter)

        if self.preprcessed_dataset is not None:
            del self.preprcessed_dataset
        self.data, self.slices = torch.load(self.processed_paths[0])
        return

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return "geometric_data_processed_{}.pt".format(self.mode)

    def process(self):
        data_list = []
        for idx in tqdm(range(len(self.preprcessed_dataset))):
            data = self.preprcessed_dataset.get(idx)

            idx_data = get_id_data_single(data, self.cutoff, self.int_cutoff, self.index_keys, self.triplets_only)
            for k in idx_data.keys():
                assert "id" in k
                data[k] = torch.tensor(idx_data[k], dtype=torch.int64)
            data_list.append(data)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        del self.preprcessed_dataset
        self.preprcessed_dataset = None
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return

