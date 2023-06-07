import json
import os
import pickle
import random
from itertools import repeat
from os.path import join
from collections import defaultdict

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import copy

from Geom3D.datasets.dataset_utils import mol_to_graph_data_obj_simple_3D
from Geom3D.datasets.dataset_3D import extend_graph


class MoleculeDatasetGEOMDrugs(InMemoryDataset):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        empty=False,
        **kwargs
    ):
        file_name_dict = {
            "train": "train_data_39k.pkl",
            "val": "val_data_5k.pkl",
            # "test": "test_data_200.pkl",
        }
        self.root = root
        self.split = split
        self.file_name = file_name_dict[self.split]

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(join(self.root, "raw"), exist_ok=True)
        os.makedirs(join(self.root, self.split, "processed"), exist_ok=True)
        
        self.pre_transform, self.pre_filter = pre_transform, pre_filter

        super(MoleculeDatasetGEOMDrugs, self).__init__(
            root, transform, pre_transform, pre_filter
        )

        if not empty:
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
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.split, 'processed')

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        return

    def process(self):
        pickle_file_name = join(self.root, "raw", self.file_name)
        with open(pickle_file_name, "rb") as f:
            mol_list = pickle.load(f)
        mol_list = [x.rdmol for x in mol_list]
        print("original molecule list len: {}".format(len(mol_list)))
        
        valid_count, invalid_count = 0, 0
        data_list = []
        for i, mol in tqdm(enumerate(mol_list)):
            if "." in Chem.MolToSmiles(mol):
                invalid_count += 1
                continue
            if mol.GetNumBonds() < 1:
                invalid_count += 1
                continue
            valid_count += 1
        
            data, atom_count = mol_to_graph_data_obj_simple_3D(mol, pure_atomic_num=False)
            data_list.append(data)

        print("invalid: {}\nvalid: {}".format(invalid_count, valid_count))
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return


class MoleculeDatasetGEOMDrugsTest(InMemoryDataset):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        empty=False,
        use_extend_graph=False,
        **kwargs
    ):
        file_name_dict = {
            # "train": "train_data_39k.pkl",
            # "val": "val_data_5k.pkl",
            "test": "test_data_200.pkl",
        }
        self.root = root
        self.split = split
        self.file_name = file_name_dict[self.split]
        self.use_extend_graph = use_extend_graph

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(join(self.root, "raw"), exist_ok=True)
        os.makedirs(join(self.root, self.split, "processed"), exist_ok=True)
        
        super(MoleculeDatasetGEOMDrugsTest, self).__init__(
            root, transform, pre_transform, pre_filter
        )

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

        return

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        if self.use_extend_graph:
            data = extend_graph(data)
        return data

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.split, 'processed')

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def load(self):
        pickle_file_name = join(self.root, "raw", self.file_name)
        with open(pickle_file_name, "rb") as f:
            data_list = pickle.load(f)
        mol_list = [x.rdmol for x in data_list]
        smiles_list = [x.smiles for x in data_list]
        
        # valid_count, invalid_count = 0, 0
        # packed_data_dict = defaultdict(list)
        # for i, (mol, smiles) in enumerate(zip(mol_list, smiles_list)):
        #     if "." in Chem.MolToSmiles(mol):
        #         invalid_count += 1
        #         continue
        #     if mol.GetNumBonds() < 1:
        #         invalid_count += 1
        #         continue
        #     valid_count += 1
        
        #     data, atom_count = mol_to_graph_data_obj_simple_3D(mol, pure_atomic_num=False)
        #     position_center = data.positions.mean(dim=0)
        #     print(position_center)
        #     data.positions -= position_center
        #     packed_data_dict[smiles].append(data)
        # print("invalid conformer: {}\nvalid conformer: {}".format(invalid_count, valid_count))
        # print("len of conformer: {}".format(len(data_list)))
        # print("len of molecule: {}".format(len(packed_data_dict)))

        # smiles_list, data_list = [], []
        # for smiles, data_list_each_mol in packed_data_dict.items():
        #     neo_data = copy.deepcopy(data_list_each_mol[0])
        #     del neo_data.positions
        #     positions_list = []
        #     for data in data_list_each_mol:
        #         positions_list.append(data.positions)
        #     positions_list = torch.cat(positions_list, 0)

        #     neo_data.positions_ref = positions_list  # (ref * N, 3)
        #     neo_data.num_positions_ref = torch.tensor(len(data_list_each_mol))
        #     print(neo_data.num_positions_ref)

        #     data_list.append(neo_data)
        #     smiles_list.append(smiles)

        valid_count, invalid_count = 0, 0
        data_smiles_list = []
        data_list = []
        data_rdmol_list = []
        for i, (mol, smiles) in enumerate(zip(mol_list, smiles_list)):
            if "." in Chem.MolToSmiles(mol):
                invalid_count += 1
                continue
            if mol.GetNumBonds() < 1:
                invalid_count += 1
                continue
            valid_count += 1
        
            data, atom_count = mol_to_graph_data_obj_simple_3D(mol, pure_atomic_num=False)
            position_center = data.positions.mean(dim=0)
            data.positions -= position_center
            data_list.append(data)
            data_smiles_list.append(smiles)
            data_rdmol_list.append(mol)
        print("invalid conformer: {}\nvalid conformer: {}".format(invalid_count, valid_count))
        print("len of conformer: {}".format(len(data_list)))
        return data_smiles_list, data_list, data_rdmol_list


    def process(self):
        data_smiles_list, data_list, data_rdmol_list = self.load()

        data_smiles_list = pd.Series(data_smiles_list)
        saver_path = os.path.join(self.processed_dir, 'smiles.csv')
        print('saving to {}'.format(saver_path))
        data_smiles_list.to_csv(saver_path, index=False, header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        return

if __name__ == "__main__":
    pickle_file_name = "../../data/GEOM_Drugs/raw/old_test_data_200.pkl"
    with open(pickle_file_name, "rb") as f:
        data_list = pickle.load(f)
    mol_list = [x.rdmol for x in data_list]
    smiles_list = [x.smiles for x in data_list]

    valid_count, invalid_count = 0, 0
    packed_mol_dict = defaultdict(list)
    for i, (mol, smiles) in tqdm(enumerate(zip(mol_list, smiles_list))):
        if "." in Chem.MolToSmiles(mol):
            invalid_count += 1
            continue
        if mol.GetNumBonds() < 1:
            invalid_count += 1
            continue
        valid_count += 1
    
        # if len(packed_mol_dict[smiles]) >= 10:
        #     continue
        packed_mol_dict[smiles].append(mol)
    print("invalid conformer: {}\nvalid conformer: {}".format(invalid_count, valid_count))
    print("len of conformer: {}".format(len(data_list)))
    print("len of molecule: {}".format(len(packed_mol_dict)))

    data_list, smiles_list = [], []
    valid_smiles_set = set()
    for smiles, mol_list_each_mol in packed_mol_dict.items():
        valid_smiles_set.add(smiles)
        if len(valid_smiles_set) >= 21:
            continue

        for rdmol in mol_list_each_mol:
            data = Data()
            data.smiles = smiles
            data.rdmol = rdmol
            data_list.append(data)
    print("data list {}".format(len(data_list)))

    neo_pickle_file_name = "../../data/GEOM_Drugs/raw/test_data_200.pkl"
    with open(neo_pickle_file_name, "wb") as f:
        pickle.dump(data_list, f)
    print('save train done')
