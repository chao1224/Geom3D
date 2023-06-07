import os
import shutil
from itertools import repeat
import numpy as np

import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset


class MoleculeDataset3DFull(InMemoryDataset):
    def __init__(self, root, preprcessed_dataset):
        self.root = root
        self.dataset = preprcessed_dataset.dataset
        self.preprcessed_dataset = preprcessed_dataset
        
        # TODO: rotation_transform is left for the future
        # self.rotation_transform = preprcessed_dataset.rotation_transform
        self.transform = preprcessed_dataset.transform
        self.pre_transform = preprcessed_dataset.pre_transform
        self.pre_filter = preprcessed_dataset.pre_filter

        super(MoleculeDataset3DFull, self).__init__(
            root, self.transform, self.pre_transform, self.pre_filter
        )
        self.data, self.slices = torch.load(self.processed_paths[0])
        print("Dataset: {}\nData: {}".format(self.dataset, self.data))

        return

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
        return data

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def process(self):
        print("Preprocessing on {} with Full Edges ...".format(self.dataset))

        preprocessed_smiles_path = os.path.join(self.preprcessed_dataset.processed_dir, "smiles.csv")
        smiles_path = os.path.join(self.processed_dir, "smiles.csv")
        shutil.copyfile(preprocessed_smiles_path, smiles_path)

        if self.dataset == "qm9":
            preprocessed_data_name_file = os.path.join(self.preprcessed_dataset.processed_dir, "name.csv")
            data_name_file = os.path.join(self.processed_dir, "name.csv")
            shutil.copyfile(preprocessed_data_name_file, data_name_file)

        data_list = []        
        for idx in tqdm(range(len(self.preprcessed_dataset))):
            data = self.preprcessed_dataset.get(idx)
            full_edge_list = []
            N = len(data.x)
            if N > 0:
                for i in range(N):
                    for j in range(i + 1, N):
                        full_edge_list.append((i, j))
                        full_edge_list.append((j, i))
                full_edge_index = torch.tensor(np.array(full_edge_list).T, dtype=torch.long)
            else:
                full_edge_index = torch.empty((2, 0), dtype=torch.long)
            data.full_edge_index = full_edge_index
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        return
