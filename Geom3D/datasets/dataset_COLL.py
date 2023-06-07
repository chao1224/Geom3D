import numpy as np
import os
from tqdm import tqdm
from itertools import repeat

import torch
from torch_geometric.data import Data, InMemoryDataset


class DatasetCOLL(InMemoryDataset):
    def __init__(
        self,
        root,
        mode,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        self.mode = mode
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        super(DatasetCOLL, self).__init__(root, self.transform, self.pre_transform, self.pre_filter)

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

    def _load_npz(self, path):
        keys = ["N", "Z", "R", "F", "E"]
        data = np.load(path, allow_pickle=True)
        print("data keys", data.keys)
        for key in keys:
            if key not in data.keys():
                if key != "F":
                    raise UserWarning(f"Can not find key {key} in the dataset.")
            else:
                setattr(self, key, data[key])
        return

    def process(self):
        path = "{}/raw/coll_v1.2_{}.npz".format(self.root, self.mode)
        self._load_npz(path)

        assert self.R is not None # positions for each atom
        assert self.N is not None # number of atoms for each molecule
        assert self.Z is not None # atomic number for each atom
        assert self.E is not None # energy for each molecule
        assert self.F is not None # force for each atom

        assert len(self.E) > 0
        assert len(self.F) > 0

        self.E = self.E[:, None]  # shape=(nMolecules,1)
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])

        data_list = []
        for idx, (start, end) in tqdm(enumerate(zip(self.N_cumsum[:-1], self.N_cumsum[1:]))):
            data = Data()

            atomic_number_list = torch.tensor(self.Z[start: end], dtype=torch.int64)
            R_ = torch.tensor(self.R[start: end], dtype=torch.float32)
            F_ = torch.tensor(self.F[start: end], dtype=torch.float32)
            E_ = torch.tensor(self.E[idx], dtype=torch.float32)

            X_ = []
            for atomic_number in atomic_number_list:
                atom_features = atomic_number - 1
                X_.append(atom_features)
            X_ = torch.tensor(X_, dtype=torch.int64)

            data = Data(x=X_, positions=R_, y=E_, force=F_)

            data_list.append(data)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return
