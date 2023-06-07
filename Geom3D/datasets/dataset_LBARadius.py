import os
from tqdm import tqdm
from itertools import repeat
import shutil

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import radius_graph


class DatasetLBARadius(InMemoryDataset):
    def __init__(self, root, preprcessed_dataset, radius):
        self.root = root
        self.preprcessed_dataset = preprcessed_dataset
        self.dataset = preprcessed_dataset.dataset
        self.radius = radius

        self.transform = preprcessed_dataset.transform
        self.pre_transform = preprcessed_dataset.pre_transform
        self.pre_filter = preprcessed_dataset.pre_filter
        
        self.year = preprcessed_dataset.year
        self.urlmap = preprcessed_dataset.urlmap
        self.dist = preprcessed_dataset.dist

        self.sanitize = preprcessed_dataset.sanitize
        self.add_hs = preprcessed_dataset.add_hs
        self.remove_hs = preprcessed_dataset.remove_hs
        self.dataframe_transformer = preprcessed_dataset.dataframe_transformer
        self.use_complex = preprcessed_dataset.use_complex

        os.makedirs(root, exist_ok=True)
        os.makedirs(os.path.join(root, "processed"), exist_ok=True)

        super(DatasetLBARadius, self).__init__(root, self.transform, self.pre_transform, self.pre_filter)

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
        return "geometric_data_processed_{}.pt".format(self.year)

    def process(self):
        print("Preprocessing on LBA Radius ...")

        preprocessed_id2data_json = os.path.join(self.preprcessed_dataset.processed_dir, "pdb_id2data_id_{}.json".format(self.year))
        id2data_json = os.path.join(self.processed_dir, "pdb_id2data_id_{}.json".format(self.year))
        shutil.copyfile(preprocessed_id2data_json, id2data_json)

        preprocessed_indices_folder = os.path.join(self.preprcessed_dataset.processed_dir, "indices")
        indices_folder = os.path.join(self.processed_dir, "indices")
        if not os.path.isdir(indices_folder):
            shutil.copytree(preprocessed_indices_folder, indices_folder)

        preprocessed_targets_folder = os.path.join(self.preprcessed_dataset.processed_dir, "targets")
        targets_folder = os.path.join(self.processed_dir, "targets")
        if not os.path.isdir(targets_folder):
            shutil.copytree(preprocessed_targets_folder, targets_folder)

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
