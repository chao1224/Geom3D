import os
from tqdm import tqdm
from itertools import repeat

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import radius_graph


class DatasetCOLLRadius(InMemoryDataset):
    def __init__(self, root, preprcessed_dataset, radius):
        self.root = root
        self.mode = preprcessed_dataset.mode
        self.preprcessed_dataset = preprcessed_dataset
        self.radius = radius

        self.transform = preprcessed_dataset.transform
        self.pre_transform = preprcessed_dataset.pre_transform
        self.pre_filter = preprcessed_dataset.pre_filter

        super(DatasetCOLLRadius, self).__init__(root, self.transform, self.pre_transform, self.pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        return

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return "geometric_data_processed_{}.pt".format(self.mode)

    def process(self):
        print("Preprocessing on COLL Radius ...")

        data_list = []
        for i in tqdm(range(len(self.preprcessed_dataset))):
            data = self.preprcessed_dataset.get(i)
            radius_edge_index = radius_graph(data.positions, r=self.radius, loop=False)
            data.radius_edge_index = radius_edge_index
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return
