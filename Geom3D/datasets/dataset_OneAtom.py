import os
import shutil
from itertools import repeat

import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset


class MoleculeDatasetOneAtom(InMemoryDataset):
    def __init__(self, root, preprcessed_dataset):
        self.root = root
        self.dataset = preprcessed_dataset.dataset
        self.preprcessed_dataset = preprcessed_dataset
        
        # self.rotation_transform = preprcessed_dataset.rotation_transform
        self.transform = preprcessed_dataset.transform
        self.pre_transform = preprcessed_dataset.pre_transform
        self.pre_filter = preprcessed_dataset.pre_filter

        self.node_class = 9

        super(MoleculeDatasetOneAtom, self).__init__(
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
        if self.dataset == "lep":
            return "geometric_data_processed_{}.pt".format(self.preprcessed_dataset.split_option)
        return "geometric_data_processed.pt"

    def process(self):
        print("Preprocessing on One-atom ...")

        if self.dataset == "qm9":
            preprocessed_smiles_path = os.path.join(self.preprcessed_dataset.processed_dir, "smiles.csv")
            smiles_path = os.path.join(self.processed_dir, "smiles.csv")
            shutil.copyfile(preprocessed_smiles_path, smiles_path)

            preprocessed_data_name_file = os.path.join(self.preprcessed_dataset.processed_dir, "name.csv")
            data_name_file = os.path.join(self.processed_dir, "name.csv")
            shutil.copyfile(preprocessed_data_name_file, data_name_file)

        elif self.dataset == "lba":
            preprocessed_id2data_json = os.path.join(self.preprcessed_dataset.processed_dir, "pdb_id2data_id_{}.json".format(self.preprcessed_dataset.year))
            id2data_json = os.path.join(self.processed_dir, "pdb_id2data_id_{}.json".format(self.preprcessed_dataset.year))
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
            if self.dataset == "lep":
                data.x_active = ((self.node_class-1) * torch.ones_like(data.x_active)).long()
                data.x_inactive = ((self.node_class-1) * torch.ones_like(data.x_inactive)).long()
            else:
                data.x = ((self.node_class-1) * torch.ones_like(data.x)).long()
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        return
