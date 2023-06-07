import os
import pandas as pd
from itertools import repeat
from tqdm import tqdm
import torch
from torch_geometric.data import Data, InMemoryDataset
from Geom3D.datasets.dataset_utils import mol_to_graph_data_obj_MMFF_3D
from .dataset_3D import extend_graph

from Geom3D.datasets.dataset_MoleculeNet_2D import \
    _load_tox21_dataset, _load_hiv_dataset, _load_bace_dataset, _load_bbbp_dataset, \
    _load_clintox_dataset, _load_sider_dataset, _load_toxcast_dataset, _load_esol_dataset, \
    _load_freesolv_dataset, _load_lipophilicity_dataset, _load_malaria_dataset, _load_cep_dataset, _load_muv_dataset


class MoleculeNetDataset3D(InMemoryDataset):
    def __init__(self, root_2D, root_3D, dataset, num_conformers,
                 transform=None, pre_transform=None, pre_filter=None, empty=False):
        self.root_2D = root_2D
        self.root_3D = root_3D
        self.dataset = dataset
        self.num_conformers = num_conformers
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform

        super(MoleculeNetDataset3D, self).__init__(root_3D, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('Dataset: {}\nData: {}'.format(self.dataset, self.data))

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_dir(self):
        return os.path.join(self.root_2D, 'raw')

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. No download allowed')

    def process(self):

        def shared_extractor(smiles_list, rdkit_mol_objs, labels):
            data_list = []
            data_smiles_list = []
            for i in tqdm(range(len(smiles_list))):
                rdkit_mol = rdkit_mol_objs[i]
                # TODO: minor fix, need to double-check the valid num of molecules
                if rdkit_mol is None:
                    continue
                data = mol_to_graph_data_obj_MMFF_3D(rdkit_mol, self.num_conformers)
                data.id = torch.tensor([i])
                data.y = torch.tensor(labels[i])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

            return data_list, data_smiles_list

        if self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = _load_tox21_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'hiv':
            smiles_list, rdkit_mol_objs, labels = _load_hiv_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'bace':
            smiles_list, rdkit_mol_objs, _, labels = _load_bace_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'bbbp':
            smiles_list, rdkit_mol_objs, labels = _load_bbbp_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'clintox':
            smiles_list, rdkit_mol_objs, labels = _load_clintox_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = _load_sider_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'toxcast':
            smiles_list, rdkit_mol_objs, labels = _load_toxcast_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'muv':
            smiles_list, rdkit_mol_objs, labels = _load_muv_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'esol':
            smiles_list, rdkit_mol_objs, labels = _load_esol_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'freesolv':
            smiles_list, rdkit_mol_objs, labels = _load_freesolv_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'lipophilicity':
            smiles_list, rdkit_mol_objs, labels = _load_lipophilicity_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'malaria':
            smiles_list, rdkit_mol_objs, labels = _load_malaria_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'cep':
            smiles_list, rdkit_mol_objs, labels = _load_cep_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(smiles_list, rdkit_mol_objs, labels)

        else:
            raise ValueError('Dataset {} not included.'.format(self.dataset))

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


class MoleculeNetDataset2D_SDE3D(InMemoryDataset):
    def __init__(self, root, path, transform=None,
                 pre_transform=None, pre_filter=None, empty=False, use_extend_graph=False):

        self.root = root
        self.path = path
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.use_extend_graph = use_extend_graph

        super(MoleculeNetDataset2D_SDE3D, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('Dataset path: {}\nData: {}'.format(self.path, self.data))

    @property
    def processed_paths(self):
        return [self.path]

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