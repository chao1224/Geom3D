import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data


'''
Credit to https://github.com/divelab/DIG/blob/dig/dig/threedgraph/dataset/PygDatasetMD17.py
'''
class DatasetMD17(InMemoryDataset):
    def __init__(self, root, task, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.dataset = "MD17"
        self.task = task
        self.folder = osp.join(root, self.task)
        self.url = 'http://quantum-machine.org/gdml/data/npz/' + self.task + '_dft.npz'

        super(DatasetMD17, self).__init__(self.folder, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        return

    @property
    def raw_file_names(self):
        return self.task + '_dft.npz'

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data = np.load(osp.join(self.raw_dir, self.raw_file_names))

        E = data['E']
        F = data['F']
        R = data['R']
        z = data['z']

        data_list = []
        for i in tqdm(range(len(E))):
            R_i = torch.tensor(R[i], dtype=torch.float32)
            atomic_number_list = torch.tensor(z, dtype=torch.int64)
            E_i = torch.tensor(E[i], dtype=torch.float32)
            F_i = torch.tensor(F[i], dtype=torch.float32)

            X_i = []
            for atomic_number in atomic_number_list:
                atom_features = atomic_number - 1
                X_i.append(atom_features)
            X_i = torch.tensor(X_i, dtype=torch.int64)

            data = Data(x=X_i, positions=R_i, y=E_i, force=F_i)

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])
        return

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict
