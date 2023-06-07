import json
import os
import pickle
import random
from itertools import repeat
from os.path import join

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from Geom3D.datasets.dataset_utils import mol_to_graph_data_obj_simple_3D


class MoleculeDatasetGEOM(InMemoryDataset):
    def __init__(
        self,
        path_prefix,
        root,
        n_mol,
        n_conf,
        n_upper,
        transform=None,
        seed=777,
        pre_transform=None,
        pre_filter=None,
        empty=False,
        **kwargs
    ):
        os.makedirs(root, exist_ok=True)
        os.makedirs(join(root, "raw"), exist_ok=True)
        os.makedirs(join(root, "processed"), exist_ok=True)
        if "smiles_copy_from_3D_file" in kwargs:
            self.smiles_copy_from_3D_file = kwargs["smiles_copy_from_3D_file"]
        else:
            self.smiles_copy_from_3D_file = None

        self.root, self.seed = root, seed
        self.path_prefix = path_prefix
        self.n_mol, self.n_conf, self.n_upper = n_mol, n_conf, n_upper
        self.pre_transform, self.pre_filter = pre_transform, pre_filter

        super(MoleculeDatasetGEOM, self).__init__(
            root, transform, pre_transform, pre_filter
        )

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print("root: {},\ndata: {},\nn_mol: {},\nn_conf: {}".format(
                self.root, self.data, self.n_mol, self.n_conf))

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
        return "geometric_data_processed.pt"

    def download(self):
        return

    def process(self):
        data_list = []
        data_smiles_list = []

        if self.smiles_copy_from_3D_file is None:
            # load 3D structure
            dir_name = "{}/rdkit_folder".format(self.path_prefix)
            drugs_file = "{}/summary_drugs.json".format(dir_name)
            with open(drugs_file, "r") as f:
                drugs_summary = json.load(f)
            drugs_summary = list(drugs_summary.items())
            # expected: 304,466 molecules
            print("# of SMILES: {}".format(len(drugs_summary)))

            random.seed(self.seed)
            random.shuffle(drugs_summary)
            mol_idx, idx, notfound = 0, 0, 0
            for smiles, sub_dic in tqdm(drugs_summary):
                ##### Path should match #####
                # pdb.set_trace()
                if sub_dic.get("pickle_path", "") == "":
                    # pdb.set_trace()
                    notfound += 1
                    continue

                mol_path = join(dir_name, sub_dic["pickle_path"])
                with open(mol_path, "rb") as f:
                    mol_dic = pickle.load(f)
                    conformer_list = mol_dic["conformers"]

                    ##### count should match #####
                    conf_n = len(conformer_list)
                    if conf_n < self.n_conf or conf_n > self.n_upper:
                        # print(smiles, len(conformer_list))
                        notfound += 1
                        continue

                    ##### SMILES should match #####
                    #  export *=https://github.com/learningmatter-mit/geom
                    #  Ref: */issues/4#issuecomment-853486681
                    #  Ref: */blob/master/tutorials/02_loading_rdkit_mols.ipynb
                    conf_list = [
                        Chem.MolToSmiles(
                            Chem.MolFromSmiles(
                                Chem.MolToSmiles(rd_mol["rd_mol"])
                            )
                        )
                        for rd_mol in conformer_list[: self.n_conf]
                    ]

                    conf_list_raw = [
                        Chem.MolToSmiles(rd_mol["rd_mol"])
                        for rd_mol in conformer_list[: self.n_conf]
                    ]
                    # check that they're all the same
                    same_confs = len(list(set(conf_list))) == 1
                    same_confs_raw = len(list(set(conf_list_raw))) == 1
                    # pdb.set_trace()
                    if not same_confs:
                        # print(list(set(conf_list)))
                        if same_confs_raw is True:
                            print("Interesting")
                        notfound += 1
                        continue

                    for conformer_dict in conformer_list[: self.n_conf]:
                        # pdb.set_trace()
                        # select the first n_conf conformations
                        rdkit_mol = conformer_dict["rd_mol"]
                        data, _ = mol_to_graph_data_obj_simple_3D(rdkit_mol)
                        data.id = torch.tensor([idx])
                        data.mol_id = torch.tensor([mol_idx])
                        data_smiles_list.append(smiles)
                        data_list.append(data)
                        idx += 1
                        # print(data.id, '\t', data.mol_id)

                # select the first n_mol molecules
                if mol_idx + 1 >= self.n_mol:
                    break
                if same_confs:
                    mol_idx += 1

            print("mol id: [0, {}]\t# smiles: {}\t# set(smiles): {}".format(
                mol_idx, len(data_smiles_list), len(set(data_smiles_list))))

        else:
            # load 2D structure from 3D files
            with open(self.smiles_copy_from_3D_file, "r") as f:
                lines = f.readlines()
            for smiles in lines:
                data_smiles_list.append(smiles.strip())
            data_smiles_list = list(dict.fromkeys(data_smiles_list))

            # load 3D structure
            dir_name = "{}/rdkit_folder".format(self.path_prefix)
            drugs_file = "{}/summary_drugs.json".format(dir_name)
            with open(drugs_file, "r") as f:
                drugs_summary = json.load(f)
            # expected: 304,466 molecules
            print("# SMILES: {}".format(len(drugs_summary.items())))

            mol_idx, idx, notfound = 0, 0, 0

            for smiles in tqdm(data_smiles_list):
                sub_dic = drugs_summary[smiles]
                mol_path = join(dir_name, sub_dic["pickle_path"])
                with open(mol_path, "rb") as f:
                    mol_dic = pickle.load(f)
                    conformer_list = mol_dic["conformers"]
                    conformer = conformer_list[0]
                    rdkit_mol = conformer["rd_mol"]
                    data, _ = mol_to_graph_data_obj_simple_3D(rdkit_mol)
                    data.mol_id = torch.tensor([mol_idx])
                    data.id = torch.tensor([idx])
                    data_list.append(data)
                    mol_idx += 1
                    idx += 1

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = join(self.processed_dir, "smiles.csv")
        print("saving to {}".format(saver_path))
        data_smiles_series.to_csv(saver_path, index=False, header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("%d molecules do not meet the requirements" % notfound)
        print("%d molecules have been processed" % mol_idx)
        print("%d conformers have been processed" % idx)
        return
