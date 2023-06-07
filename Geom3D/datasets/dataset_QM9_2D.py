import os
from itertools import repeat

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.constants import physical_constants
from torch_geometric.data import (Data, InMemoryDataset, download_url, extract_zip)

from Geom3D.datasets.dataset_utils import mol_to_graph_data_obj_simple_3D


class MoleculeDatasetQM92D(InMemoryDataset):
    raw_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
    raw_url2 = "https://ndownloader.figshare.com/files/3195404"
    raw_url3 = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm9.csv"
    raw_url4 = "https://springernature.figshare.com/ndownloader/files/3195395"

    def __init__(
        self,
        root,
        dataset,
        task,
        rotation_transform=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        calculate_thermo=True,
    ):
        """
        The complete columns are
        A,B,C,mu,alpha,homo,lumo,gap,r2,zpve,u0,u298,h298,g298,cv,u0_atom,u298_atom,h298_atom,g298_atom
        and we take
        mu,alpha,homo,lumo,gap,r2,zpve,u0,u298,h298,g298,cv
        """
        self.root = root
        self.rotation_transform = rotation_transform
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        self.target_field = ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298", "cv", "gap_02"]
        self.pd_target_field = ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298", "cv"]
        
        self.task = task
        if self.task == "qm9":
            self.task_id = None
        else:
            self.task_id = self.target_field.index(task)
        self.calculate_thermo = calculate_thermo
        self.atom_dict = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

        self.hartree2eV = physical_constants["hartree-electron volt relationship"][0]

        self.conversion = {
            "mu": 1.0,
            "alpha": 1.0,
            "homo": self.hartree2eV,
            "lumo": self.hartree2eV,
            "gap": self.hartree2eV,
            "gap_02": self.hartree2eV,
            "r2": 1.0,
            "zpve": self.hartree2eV,
            "u0": self.hartree2eV,
            "u298": self.hartree2eV,
            "h298": self.hartree2eV,
            "g298": self.hartree2eV,
            "cv": 1.0,
        }

        super(MoleculeDatasetQM92D, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        self.dataset = dataset
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
        if self.rotation_transform is not None:
            data.positions = self.rotation_transform(data.positions)
        return data

    @property
    def raw_file_names(self):
        return [
            "gdb9.sdf",
            "gdb9.sdf.csv",
            "uncharacterized.txt",
            "qm9.csv",
            "atomref.txt",
        ]

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        file_path = download_url(self.raw_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

        download_url(self.raw_url2, self.raw_dir)
        os.rename(
            os.path.join(self.raw_dir, "3195404"),
            os.path.join(self.raw_dir, "uncharacterized.txt"),
        )

        download_url(self.raw_url3, self.raw_dir)

        download_url(self.raw_url4, self.raw_dir)
        os.rename(
            os.path.join(self.raw_dir, "3195395"),
            os.path.join(self.raw_dir, "atomref.txt"),
        )
        return

    def get_thermo_dict(self):
        gdb9_txt_thermo = self.raw_paths[4]
        # Loop over file of thermochemical energies
        therm_targets = ["zpve", "u0", "u298", "h298", "g298", "cv"]
        therm_targets = [6, 7, 8, 9, 10, 11]

        # Dictionary that
        id2charge = self.atom_dict

        # Loop over file of thermochemical energies
        therm_energy = {target: {} for target in therm_targets}
        with open(gdb9_txt_thermo) as f:
            for line in f:
                # If line starts with an element, convert the rest to a list of energies.
                split = line.split()

                # Check charge corresponds to an atom
                if len(split) == 0 or split[0] not in id2charge.keys():
                    continue

                # Loop over learning targets with defined thermochemical energy
                for therm_target, split_therm in zip(therm_targets, split[1:]):
                    therm_energy[therm_target][id2charge[split[0]]] = float(split_therm)

        return therm_energy

    def process(self):
        therm_energy = self.get_thermo_dict()
        print("therm_energy\t", therm_energy)

        df = pd.read_csv(self.raw_paths[1])
        df = df[self.pd_target_field]
        df["gap_02"] = df["lumo"] - df["homo"]

        target = df.to_numpy()
        target = torch.tensor(target, dtype=torch.float)

        with open(self.raw_paths[2], "r") as f:
            # These are the mis-matched molecules, according to `uncharacerized.txt` file.
            skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

        data_df = pd.read_csv(self.raw_paths[3])
        whole_smiles_list = data_df["smiles"].tolist()
        print("TODO\t", whole_smiles_list[:100])

        # getNumImplicitHs() called without preceding call to calcImplicitValence()
        # rdkit.Chem.rdchem.AtomValenceException: Explicit valence for atom # 1 C, 5, is greater than permitted
        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=True)

        print("suppl: {}\tsmiles_list: {}".format(len(suppl), len(whole_smiles_list)))

        data_list, data_smiles_list, data_name_list, idx, invalid_count = [], [], [], 0, 0

        for i, mol in enumerate(suppl):
            if i in skip:
                print("Exception with (skip)\t", i)
                invalid_count += 1
                continue
            
            if mol is None:
                continue
            
            data, atom_count = mol_to_graph_data_obj_simple_3D(mol, pure_atomic_num=False)

            data.id = torch.tensor([idx])
            temp_y = target[i]
            if self.calculate_thermo:
                for atom, count in atom_count.items():
                    if atom not in self.atom_dict.values():
                        continue
                    for target_id, atom_sub_dic in therm_energy.items():
                        temp_y[target_id] -= atom_sub_dic[atom] * count

            # convert units
            for idx, col in enumerate(self.target_field):
                temp_y[idx] *= self.conversion[col]
            data.y = temp_y

            name = mol.GetProp("_Name")
            smiles = whole_smiles_list[i]

            # TODO: need double-check this
            temp_mol = AllChem.MolFromSmiles(smiles)
            if temp_mol is None:
                print("Exception with (invalid mol)\t", i)
                invalid_count += 1
                continue

            data_smiles_list.append(smiles)
            data_name_list.append(name)
            data_list.append(data)
            idx += 1

        print(
            "mol id: [0, {}]\tlen of smiles: {}\tlen of set(smiles): {}".format(
                idx - 1, len(data_smiles_list), len(set(data_smiles_list))
            )
        )
        print("{} invalid molecules".format(invalid_count))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # TODO: need double-check later, the smiles list are identical here?
        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = os.path.join(self.processed_dir, "smiles.csv")
        print("saving to {}".format(saver_path))
        data_smiles_series.to_csv(saver_path, index=False, header=False)

        data_name_series = pd.Series(data_name_list)
        saver_path = os.path.join(self.processed_dir, "name.csv")
        print("saving to {}".format(saver_path))
        data_name_series.to_csv(saver_path, index=False, header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        return