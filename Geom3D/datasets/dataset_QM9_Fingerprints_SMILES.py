import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import selfies as sf
from scipy.constants import physical_constants
import json

import torch
from torch.utils.data import Dataset
from Geom3D.datasets.dataset_utils import mol_to_graph_data_obj_simple_3D


class MoleculeDatasetQM9FingerprintsSMILES(Dataset):
    def __init__(self, root, task, data_type="ecfp", ecfp_radius=2, ecfp_length=1024, calculate_thermo=True,):
        self.root = root
        self.data_type = data_type
        self.ecfp_radius = ecfp_radius
        self.ecfp_length = ecfp_length

        self.target_field = [
            "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
            "u0", "u298", "h298", "g298", "cv", "gap_02",
        ]
        self.pd_target_field = [
            "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
            "u0", "u298", "h298", "g298", "cv",
        ]
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

        self.raw_paths = [
            os.path.join(self.root, "raw", "gdb9.sdf"),
            os.path.join(self.root, "raw", "gdb9.sdf.csv"),
            os.path.join(self.root, "raw", "uncharacterized.txt"),
            os.path.join(self.root, "raw", "qm9.csv"),
            os.path.join(self.root, "raw", "atomref.txt"),
        ]

        self.processed_dir = os.path.join(self.root, "processed_ecfp_smiles")
        self.processed_ecfp_file = os.path.join(self.processed_dir, "ecfp.csv")
        self.processed_smiles_file = os.path.join(self.processed_dir, "smiles.csv")
        self.processed_smiles_json_file = os.path.join(self.processed_dir, "smiles_vocba.json")
        self.processed_selfies_file = os.path.join(self.processed_dir, "selfies.csv")
        self.processed_selfies_vocab_file = os.path.join(self.processed_dir, "selfies_vocab.json")
        self.processed_label_file = os.path.join(self.processed_dir, "label")

        if not os.path.exists(self.processed_ecfp_file):
            self.process()

        if self.data_type == "ecfp":
            ecfp_list = [line.strip("\r\n ") for line in open(self.processed_ecfp_file)]
            self.data_list = [[float(x) for x in ecfp] for ecfp in ecfp_list]
            self.pad_to_len = None
            self.pad_symbol = None
            self.pad_index = None
            self.vocab_size = None
            self.vocab_file = None

        elif self.data_type == "smiles":
            smiles_list = [line.strip("\r\n ") for line in open(self.processed_smiles_file)]

            alphabet = set()
            for smiles in smiles_list:
                for char in smiles:
                    alphabet.add(char)
            pad_symbol = "[PAD]"
            alphabet.add(pad_symbol)
            alphabet = list(sorted(alphabet))
            pad_to_len = max(len(s) for s in smiles_list)  # 34
            symbol_to_idx = {s: i for i, s in enumerate(alphabet)}

            with open(self.processed_smiles_json_file, "w") as f:
                json.dump(symbol_to_idx, f)

            self.pad_to_len = pad_to_len
            self.pad_symbol = pad_symbol
            self.pad_index = symbol_to_idx[self.pad_symbol]
            self.vocab_size = len(alphabet)
            self.vocab_file = self.processed_smiles_json_file

            self.data_list = smiles_list

        elif self.data_type == "selfies":
            selfies_list = [line.strip("\r\n ") for line in open(self.processed_selfies_file)]
            
            alphabet = sf.get_alphabet_from_selfies(selfies_list)
            pad_symbol = "[nop]"  # [nop] is a special padding symbol
            alphabet.add(pad_symbol)
            alphabet = list(sorted(alphabet))
            pad_to_len = max(sf.len_selfies(s) for s in selfies_list)  # 21
            symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
            with open(self.processed_selfies_vocab_file, "w") as f:
                json.dump(symbol_to_idx, f)

            # TODO: will see how to create our customized tokenizer.
            selfies_index_list = []
            for selfies in selfies_list:
                index, _ = sf.selfies_to_encoding(selfies=selfies, vocab_stoi=symbol_to_idx, pad_to_len=pad_to_len, enc_type="both")
                selfies_index_list.append(index)
            selfies_index_list = np.array(selfies_index_list)  # 130831, 21
            self.data_list = selfies_index_list

            self.pad_to_len = pad_to_len
            self.pad_symbol = pad_symbol
            self.pad_index = symbol_to_idx[self.pad_symbol]
            self.vocab_size = len(alphabet)
            self.vocab_file = self.processed_selfies_vocab_file
        
        label_data = np.load(self.processed_label_file+".npz")
        self.label_list = label_data["label"]

        return

    def mean(self):
        y = torch.stack(self.label_list, dim=0)
        y = y.mean(dim=0)
        return y

    def std(self):
        y = torch.stack(self.label_list, dim=0)
        y = y.std(dim=0)
        return y

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.data_type == "ecfp":
            data = torch.Tensor(data)
        label = self.label_list[idx]
        label = torch.Tensor(label)
        return data, label

    def __len__(self):
        return self.label_list.shape[0]

    def get_thermo_dict(self):
        gdb9_txt_thermo = os.path.join(self.raw_paths[4])
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

        with open(self.raw_paths[2], "r") as f:
            # These are the mis-matched molecules, according to `uncharacerized.txt` file.
            skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

        data_df = pd.read_csv(self.raw_paths[3])
        whole_smiles_list = data_df["smiles"].tolist()
        print("TODO\t", whole_smiles_list[:100])

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)
        print("suppl: {}\tsmiles_list: {}".format(len(suppl), len(whole_smiles_list)))

        data_ecfp_list, data_smiles_list, data_y_list, invalid_count = [], [], [], 0

        for i, mol in enumerate(suppl):
            if i in skip:
                print("Exception with (skip)\t", i)
                invalid_count += 1
                continue

            _, atom_count = mol_to_graph_data_obj_simple_3D(mol, pure_atomic_num=True)

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

            smiles = whole_smiles_list[i]
            RDKit_mol = AllChem.MolFromSmiles(smiles)
            smiles = Chem.MolToSmiles(RDKit_mol)
            if RDKit_mol is None:
                print("Exception with (invalid mol)\t", i)
                invalid_count += 1
                continue
            
            ecfp = AllChem.GetMorganFingerprintAsBitVect(RDKit_mol, self.ecfp_radius, self.ecfp_length).ToBitString()

            data_ecfp_list.append(ecfp)
            data_smiles_list.append(smiles)
            data_y_list.append(temp_y)

        print("{} invalid molecules".format(invalid_count))
        print("{} valid molecules".format(len(data_ecfp_list)))

        os.makedirs(self.processed_dir, exist_ok=True)

        data_ecfp_series = pd.Series(data_ecfp_list)
        data_ecfp_series.to_csv(self.processed_ecfp_file, index=False, header=False)

        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(self.processed_smiles_file, index=False, header=False)

        data_y_list = np.array(data_y_list)
        np.savez(self.processed_label_file, label=data_y_list)
        
        selfies_list = [sf.encoder(smiles) for smiles in data_smiles_list]
        selfies_series = pd.Series(selfies_list)
        selfies_series.to_csv(self.processed_selfies_file, index=False, header=False)

        return