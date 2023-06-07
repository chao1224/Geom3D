"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

---
Being checked on Aug 15.
"""
import os
import bisect
import logging
import math
import pickle
import random
import warnings
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.data import Batch, Data


def pyg2_data_transform(data: Data):
    # if we're on the new pyg (2.0 or later), we need to convert the data to the new format
    if torch_geometric.__version__ >= "2.0":
        return Data(
            **{k: v for k, v in data.__dict__.items() if v is not None}
        )
    return data


def s2ef_data_transform(data_object: Data):
    # from 0 index of embedding 
    data_object.x = data_object.atomic_numbers - 1   # 
    data_object.x = data_object.x.long()
    # available keys: y; force; pos; atomic_numbers
    data_object.y = torch.tensor([data_object.y], dtype=torch.float32)  # batch it 
    return data_object


def is2re_data_transform(data_object: Data):
    # from 0
    data_object.x = data_object.atomic_numbers - 1 
    data_object.x = data_object.x.long()
    # available keys: y_relaxed; y_init (> or >> y_relaxed); force; pos_relaxed; pos; atomic_numbers
    data_object.y_relaxed = torch.tensor([data_object.y_relaxed], dtype=torch.float32)
    return data_object


class DatasetOC20(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.

    May too large to fit into CPU memory such that will not use InMemoeryDataset cls

    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.

    Args:
        root (dict): Dataset directory, for example, 'oc20/s2ef/200k/train/'
        transform (callable, optional): Data transform function.
                (default: :obj:`None`)
    """

    def __init__(self, root, transform=None):
        super(DatasetOC20).__init__()

        self.path = Path(root)
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"

            self.metadata_path = self.path / "metadata.npz"

            self._keys, self.envs = [], []
            for db_path in db_paths:
                self.envs.append(self.connect_db(db_path))
                _length = self.envs[-1].begin().get("length".encode("ascii"))
                if _length is None:  # which happens for is2*
                    length = self.envs[-1].stat()['entries']
                else:
                    length = pickle.loads(_length)
                self._keys.append(list(range(length)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            self.metadata_path = self.path.parent / "metadata.npz"
            self.env = self.connect_db(self.path)
            self._keys = [
                f"{j}".encode("ascii")
                for j in range(self.env.stat()["entries"])
            ]
            self.num_samples = len(self._keys)
        self.transform = transform
    
    def len(self): 
        return self.__len__()

    def __len__(self):
        return self.num_samples

    def get(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        """return single pyg.Data() instance
        """
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(self._keys[idx])
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))

        if self.transform is not None:
            data_object = self.transform(data_object)
        
        return data_object

    def connect_db(self, lmdb_path=None):
        """Bug report: 
            lmdb: Too many open files
        Fixed: add `ulimit -n 4096`
        """
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()


def _load_dataset_and_print_len(name, folder_name):
    root = os.path.join(folder_name, name)
    print('>>>', name)
    d = DatasetOC20(root)
    print('\tlen=', len(d))
    print('\t', d[0])
    d.close_db()


if __name__ == '__main__':
    # bugs for is2* datasets: 'length' does not exist as a key
    # fixed by accessing the stat(): 'entries'

    # name_list = [
    #     's2ef/200k/train', 's2ef/2M/train', 
    #     *['s2ef/all/' + i for i in ['val_id', 'test_id', 'test_ood_ads', 'test_ood_cat', 'test_ood_both']],
    #     'is2re/10k/train', 'is2re/100k/train', 
    #     *['is2re/all/' + i  for i in ['train', 'val_id', 'val_ood_ads', 'val_ood_cat', 'val_ood_both', 'test_id', 'test_ood_ads', 'test_ood_cat', 'test_ood_both']],
    #     ]
    # print(name_list)
    name_list = [
        'is2re/10k/train', 'is2re/100k/train', 
        *['is2re/all/' + i  for i in ['train', 'val_id', 'val_ood_ads', 'val_ood_cat', 'val_ood_both', 'test_id', 'test_ood_ads', 'test_ood_cat', 'test_ood_both']],
    ]
    print(name_list)
    for name in name_list:
        print(name)
        _load_dataset_and_print_len(name, "../../data/OC20_data")
