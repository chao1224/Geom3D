import os
import shutil
from itertools import repeat
import numpy as np

import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import radius_graph
from torch_geometric.utils import subgraph, to_networkx


class MoleculeDataset3DRemoveCenter(InMemoryDataset):
    def __init__(self, root, preprcessed_dataset, remove_center=False):
        self.root = root
        self.dataset = preprcessed_dataset.dataset
        self.preprcessed_dataset = preprcessed_dataset
        self.remove_center = remove_center
        
        # TODO: rotation_transform is left for the future
        # self.rotation_transform = preprcessed_dataset.rotation_transform
        self.transform = preprcessed_dataset.transform
        self.pre_transform = preprcessed_dataset.pre_transform
        self.pre_filter = preprcessed_dataset.pre_filter

        super(MoleculeDataset3DRemoveCenter, self).__init__(root, self.transform, self.pre_transform, self.pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print("Dataset: {}\nData: {}".format(self.dataset, self.data))

        return

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def mean(self):
        y = torch.stack([self.get(i).y for i in range(len(self))], dim=0)
        y = y.mean(dim=0)
        return y

    def std(self):
        y = torch.stack([self.get(i).y for i in range(len(self))], dim=0)
        y = y.std(dim=0)
        return y

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        if self.remove_center:
            center = data.positions.mean(dim=0)
            data.positions -= center

        return data
