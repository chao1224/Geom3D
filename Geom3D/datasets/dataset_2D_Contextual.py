import os
from itertools import repeat
import json
from tqdm import tqdm
from rdkit import Chem
import pandas as pd

import torch
from torch_geometric.data import Data, InMemoryDataset
from .dataset_2D_Contextual_utils import MolVocab, atom_to_vocab, bond_to_vocab
from Geom3D.datasets.dataset_utils import mol_to_graph_data_obj_simple_2D

        
class MoleculeContextualDataset(InMemoryDataset):
    def __init__(self, root, dataset="zinc250k",
                 transform=None, pre_transform=None, pre_filter=None, empty=False):
        self.dataset = dataset
        self.original_root = root
        self.root = root + '_Contextual'

        # TODO: may check how to merge this into InMemoryDataset
        raw_dir = self.raw_dir
        print("raw_dir", raw_dir)
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)

        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        super(MoleculeContextualDataset, self).__init__(self.root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print("Dataset: {}\nData: {}".format(self.dataset, self.data))
        self.smiles_file = os.path.join(self.root, "processed", "smiles.csv")
        self.data_smiles_list = self.load_smiles_list()
        self.molecule_list = None

        ########## Extract atom vocabulary ##########
        self.atom_vocab_file = "{}_vocab.pkl".format("atom")
        self.atom_vocab_label_file = "{}_vocab_label.json".format("atom")
        self.atom_vocab_save_path = os.path.join(self.root, "processed", self.atom_vocab_file)
        self.atom_vocab_label_save_path = os.path.join(self.root, "processed", self.atom_vocab_label_file)
        if (not os.path.exists(self.atom_vocab_save_path)) or (not os.path.exists(self.atom_vocab_label_save_path)):
            if self.molecule_list is None:
                self.molecule_list = self.load_mol_list()
        self.atom_vocab = self.process_contextual_vocabulary(vocab_type="atom", vocab_save_path=self.atom_vocab_save_path)
        self.atom2vocab_label = self.process_atom_contextual_label_with_vocabulary()
        print('len of atom_vocab\t', len(self.atom_vocab))
        print('len of atom2vocab_label', len(self.atom2vocab_label))

        ########## Extract bond vocabulary ##########
        self.bond_vocab_file = "{}_vocab.pkl".format("bond")
        self.bond_vocab_label_file = "{}_vocab_label.json".format("bond")
        self.bond_vocab_save_path = os.path.join(self.root, "processed", self.bond_vocab_file)
        self.bond_vocab_label_save_path = os.path.join(self.root, "processed", self.bond_vocab_label_file)
        if (not os.path.exists(self.bond_vocab_save_path)) or (not os.path.exists(self.bond_vocab_label_save_path)):
            if self.molecule_list is None:
                self.molecule_list = self.load_mol_list()
        self.bond_vocab = self.process_contextual_vocabulary(vocab_type="bond", vocab_save_path=self.bond_vocab_save_path)
        self.bond2vocab_label = self.process_bond_contextual_label_with_vocabulary()
        print('len of bond_vocab\t', len(self.bond_vocab))
        print('len of bond2vocab_label', len(self.bond2vocab_label))

        return

    def load_smiles_list(self):
        data_smiles_list = []
        with open(self.smiles_file, 'r') as f:
            lines = f.readlines()
        for smiles in lines:
            data_smiles_list.append(smiles.strip())
        return data_smiles_list

    def load_mol_list(self):
        molecule_list = [Chem.MolFromSmiles(smiles) for smiles in self.data_smiles_list]
        return molecule_list

    def process(self):
        self.smiles_file = os.path.join(self.original_root, "processed", "smiles.csv")
        data_smiles_list = self.load_smiles_list()
        data_list = []
        for smiles in data_smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            data = mol_to_graph_data_obj_simple_2D(mol)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = os.path.join(self.processed_dir, 'smiles.csv')
        print('saving to {}'.format(saver_path))
        data_smiles_series.to_csv(saver_path, index=False, header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return

    def process_contextual_vocabulary(self, vocab_type, vocab_save_path):
        if os.path.exists(vocab_save_path):
            print('Loading from vocab_save_path\t', vocab_save_path)
            vocab = MolVocab.load_vocab(vocab_save_path)
            return vocab

        vocab = MolVocab(
            molecule_list=self.molecule_list,
            max_size=None,
            min_freq=1,
            num_workers=100,
            vocab_type=vocab_type)
        print("{} vocab size: {}".format(vocab_type, len(vocab)))
        print("Saving to vocab_save_path\t", vocab_save_path)
        vocab.save_vocab(vocab_save_path)
        return vocab

    def process_atom_contextual_label_with_vocabulary(self):
        if os.path.exists(self.atom_vocab_label_save_path):
            print('Loading from atom_vocab_label_save_path\t', self.atom_vocab_label_save_path)
            with open(self.atom_vocab_label_save_path, 'r') as f:
                atom2vocab_label = json.load(f)
            keys_list = list(atom2vocab_label.keys())
            neo = {}
            for key in keys_list:
                neo[int(key)] = atom2vocab_label[key]
            return neo

        atom2vocab_label = {}

        for idx, mol in tqdm(enumerate(self.molecule_list)):
            mlabel = [0] * mol.GetNumAtoms()
            n_atoms = mol.GetNumAtoms()

            for p in range(n_atoms):
                atom = mol.GetAtomWithIdx(int(p))
                mlabel[p] = self.atom_vocab.stoi.get(atom_to_vocab(mol, atom), self.atom_vocab.other_index)

            atom2vocab_label[idx] = mlabel

        print("Saving to atom_vocab_label_save_path\t", self.atom_vocab_label_save_path)
        with open(self.atom_vocab_label_save_path, 'w') as f:
            json.dump(atom2vocab_label, f)
        return atom2vocab_label

    def process_bond_contextual_label_with_vocabulary(self):
        if os.path.exists(self.bond_vocab_label_save_path):
            print('Loading from bond_vocab_label_save_path\t', self.bond_vocab_label_save_path)
            with open(self.bond_vocab_label_save_path, 'r') as f:
                bond2vocab_label = json.load(f)
            keys_list = list(bond2vocab_label.keys())
            neo = {}
            for key in keys_list:
                neo[int(key)] = bond2vocab_label[key]
            return neo

        bond2vocab_label = {}

        for idx, mol in tqdm(enumerate(self.molecule_list)):
            mlabel = []
            n_atoms = mol.GetNumAtoms()

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                label = self.bond_vocab.stoi.get(bond_to_vocab(mol, bond), self.bond_vocab.other_index)
                mlabel.extend([label])

            bond2vocab_label[idx] = mlabel

        print("Saving to bond_vocab_label_save_path\t", self.bond_vocab_label_save_path)
        with open(self.bond_vocab_label_save_path, 'w') as f:
            json.dump(bond2vocab_label, f)
        return bond2vocab_label

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        data.atom_vocab_label = torch.LongTensor(self.atom2vocab_label[idx])
        return data

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        return