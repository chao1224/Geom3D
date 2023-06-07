import os
import torch
import pandas as pd

from tqdm import tqdm
from rdkit import Chem
from itertools import repeat
from torch_geometric.data import InMemoryDataset
from Geom3D.datasets.dataset_utils import mol_to_graph_data_obj_simple_3D


class PCQM4Mv2(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        super(PCQM4Mv2, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        print("root: {},\ndata: {}".format(self.root, self.data))
        return

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        name = 'pcqm4m-v2-train.sdf '
        return name

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        pass

    def process(self):
        data_df = pd.read_csv(os.path.join(self.raw_dir, 'data.csv.gz'))
        # smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        data_list, data_smiles_list = [], []

        sdf_file = "{}/{}".format(self.raw_dir, self.raw_file_names).strip()

        suppl = Chem.SDMolSupplier(sdf_file)
        for idx, mol in tqdm(enumerate(suppl)):
            data, _ = mol_to_graph_data_obj_simple_3D(mol)
            data_list.append(data)

            smiles = Chem.MolToSmiles(mol)
            data_smiles_list.append(smiles)

            data.y = homolumogap_list[idx]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = os.path.join(self.processed_dir, "smiles.csv")
        print("saving to {}".format(saver_path))
        data_smiles_series.to_csv(saver_path, index=False, header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        return data
