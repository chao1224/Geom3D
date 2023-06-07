import os
import shutil
import itertools
from itertools import repeat
import numpy as np

import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset


def get_angle(vec1, vec2):
    '''Credit to https://github.com/PaddlePaddle/PaddleHelix/blob/dev/pahelix/utils/compound_tools.py#L415'''
    # TODO: will double-check
    norm1 = torch.linalg.norm(vec1)
    norm2 = torch.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
    vec2 = vec2 / (norm2 + 1e-5)
    angle = torch.arccos(torch.dot(vec1, vec2))
    return angle


class MoleculeDataset3DTorsionAngle(InMemoryDataset):
    def __init__(self, root, preprcessed_dataset, ratio):
        self.root = root
        self.dataset = preprcessed_dataset.dataset
        self.preprcessed_dataset = preprcessed_dataset
        self.ratio = ratio
        
        # TODO: rotation_transform is left for the future
        # self.rotation_transform = preprcessed_dataset.rotation_transform
        self.transform = preprcessed_dataset.transform
        self.pre_transform = preprcessed_dataset.pre_transform
        self.pre_filter = preprcessed_dataset.pre_filter

        super(MoleculeDataset3DTorsionAngle, self).__init__(
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
        print("Preprocessing on {} with Triplet Nodes to calculate Torsion Angle ...".format(self.dataset))

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
            positions = data.positions
            super_edge_index, super_edge_angle = [], []
            N = len(data.x)
            
            if N >= 3:
                super_edge_index = list(itertools.permutations(np.arange(N), 3))
                super_edge_index = np.array(super_edge_index).T

                if self.ratio < 1:
                    M = super_edge_index.shape[1]
                    sampled_M = int(M * self.ratio)
                    sampled_edge_index = np.random.choice(M, sampled_M, replace=False)
                    super_edge_index = super_edge_index[:, sampled_edge_index]

                for (i,j,k) in super_edge_index.T:
                    def _add(a, b, c):
                        edge_0 = positions[b] - positions[a]
                        edge_1 = positions[c] - positions[a]
                        angle = get_angle(edge_0, edge_1)
                        super_edge_angle.append(angle)

                    _add(i, j, k)

                super_edge_angle = np.array(super_edge_angle)

                super_edge_index = torch.tensor(super_edge_index, dtype=torch.long)
                super_edge_angle = torch.tensor(super_edge_angle, dtype=torch.float)
            else:
                super_edge_index = torch.empty((3, 0), dtype=torch.long)
                super_edge_angle = torch.empty((0), dtype=torch.float)

            data.super_edge_index = super_edge_index
            data.super_edge_angle = super_edge_angle
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        return
