import os
import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm
from itertools import repeat

import torch
from torch_geometric.data import Data, InMemoryDataset
from atom3d.datasets import LMDBDataset, extract_coordinates_as_numpy_arrays


# Credits to https://github.com/drorlab/atom3d/blob/master/examples/lep/enn/data.py#L222
# class EnvironmentSelection(object):
class TransformLEP(object):
    """
    Selects a region of protein coordinates within a certain distance from the alpha carbon of the mutated residue.
    :param df: Atoms data
    :type df: pandas.DataFrame
    :param dist: Distance from the alpha carbon of the mutated residue
    :type dist: float
    :return new_df: Transformed atoms data
    :rtype new_df: pandas.DataFrame
    """
    def __init__(self, dist, maxnum, droph):
        self._dist = dist
        self._maxnum = maxnum
        self._droph = droph

    def _drop_hydrogen(self, df):
        df_noh = df[df['element'] != 'H']
        # print('Number of atoms after dropping hydrogen:', len(df_noh))
        return df_noh

    # Need to match with https://github.com/drorlab/atom3d/blob/master/atom3d/util/formats.py#L531-L542
    def _replace(self, df):
        new_elements = []
        for i in range(len(df["element"])):
            element = df["element"][i]
            if len(element) > 1:
                element = element[0] + element[1].lower()
            new_elements.append(element)
        df["element"] = new_elements
        return df

    def _select_env_by_dist(self, df, chain):
        # # TODO: debug
        # temp = df['chain'].to_list()
        # print('chain:', set(temp))
        # """
        # chain: {'L', 'A'}
        # chain: {'L', 'D', 'E', 'B', 'G'}
        # chain: {'C', 'A', 'B', 'L'}
        # chain: {'L', 'A', 'B'}
        # chain: {'L', 'D', 'C'}
        # """

        # Separate pocket and ligand
        ligand = df[df['chain']==chain]
        pocket = df[df['chain']!=chain]
        # Extract coordinates
        ligand_coords = np.array([ligand.x, ligand.y, ligand.z]).T
        pocket_coords = np.array([pocket.x, pocket.y, pocket.z]).T
        # Select the environment around the mutated residue
        kd_tree = sp.spatial.KDTree(pocket_coords)
        key_pts = kd_tree.query_ball_point(ligand_coords, r=self._dist, p=2.0)
        key_pts = np.unique([k for l in key_pts for k in l])
        # Construct the new data frame
        new_df = pd.concat([ pocket.iloc[key_pts], ligand ], ignore_index=True)
        # print('Number of atoms after distance selection:', len(new_df))
        return new_df

    def _select_env_by_num(self, df, chain):
        # Separate pocket and ligand
        ligand = df[df['chain']==chain]
        pocket = df[df['chain']!=chain]
        # Max. number of protein atoms
        num = int(max([1, self._maxnum - len(ligand.x)]))
        # Extract coordinates
        ligand_coords = np.array([ligand.x, ligand.y, ligand.z]).T
        pocket_coords = np.array([pocket.x, pocket.y, pocket.z]).T
        # Select the environment around the mutated residue
        kd_tree = sp.spatial.KDTree(pocket_coords)
        dd, ii = kd_tree.query(ligand_coords, k=len(pocket.x), p=2.0)
        # Get minimum distance to any lig atom for each protein atom
        dist = [min(dd[ii==j]) for j in range(len(pocket.x)) ]
        # Sort indices by distance
        indices = np.argsort(dist)
        # Select the num closest atoms
        indices = np.sort(indices[:num])
        # Construct the new data frame
        new_df = pd.concat([pocket.iloc[indices], ligand], ignore_index=True)
        # print('Number of atoms after number selection:', len(new_df))
        return new_df

    def __call__(self, x):
        # Select the ligand! 
        chain = 'L'
        # Replace rare atoms
        x['atoms_active'] = self._replace(x['atoms_active'])
        x['atoms_inactive'] = self._replace(x['atoms_inactive'])
        # Drop the hydrogen atoms
        if self._droph:
            x['atoms_active'] = self._drop_hydrogen(x['atoms_active'])
            x['atoms_inactive'] = self._drop_hydrogen(x['atoms_inactive'])
        # Select the environment
        x['atoms_active'] = self._select_env_by_dist(x['atoms_active'], chain)
        x['atoms_active'] = self._select_env_by_num(x['atoms_active'], chain)
        x['atoms_inactive'] = self._select_env_by_dist(x['atoms_inactive'], chain)
        x['atoms_inactive'] = self._select_env_by_num(x['atoms_inactive'], chain)
        return x


class DatasetLEP(InMemoryDataset):
    def __init__(
        self,
        root,
        split_option,
        dataframe_transformer,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        self.split_option = split_option
        self.dataframe_transformer = dataframe_transformer
        self.dataset = "lep"

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        super(DatasetLEP, self).__init__(self.root, self.transform, self.pre_transform, self.pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        return

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return "geometric_data_processed_{}.pt".format(self.split_option)

    def calc_stats(self):
        self.stats = {key: (val.mean(), val.std()) for key, val in self.lmdb_data.items() if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}
        print("stats: ", self.stats)

    def convert_units(self, units_dict):
        # TODO: no longer used?
        for key in self.lmdb_data.keys():
            if key in units_dict:
                self.lmdb_data[key] *= units_dict[key]
        self.calc_stats()

    def __lmdb_len__(self):
        return self.num_pts
    
    def __lmdb_getitem__(self, idx):
        return {key: val[idx] for key, val in self.lmdb_data.items()}

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    # Credit to https://github.com/drorlab/atom3d/blob/master/examples/lep/enn/data.py#L153
    def load_lmdb(self):
        key_names = ['index', 'num_atoms', 'charges', 'positions']

        folder = os.path.join(self.root, "raw/split-by-protein/data", self.split_option)
        print("Loading from ", folder)
        dataset = LMDBDataset(folder, transform=self.dataframe_transformer)

        # # TODO: debugging
        # for idx, item in enumerate(dataset):
        #     print(idx)
        #     print('atoms_active', item['atoms_active'].shape)
        #     print('atoms_inactive', item['atoms_inactive'].shape)

        # Load original atoms
        act = extract_coordinates_as_numpy_arrays(dataset, atom_frames=['atoms_active'])
        for k in key_names:
            act[k+'_active'] = act.pop(k)
            """
            act ['index_active', 'num_atoms_active', 'charges_active', 'positions_active']
            """
        
        # Load mutated atoms
        ina = extract_coordinates_as_numpy_arrays(dataset, atom_frames=['atoms_inactive'])
        for k in key_names:
            ina[k+'_inactive'] = ina.pop(k)
            """
            act ['index_inactive', 'num_atoms_inactive', 'charges_inactive', 'positions_inactive']
            """

        # Merge datasets with atoms
        dsdict = {**act, **ina}
        ldict = {'A':1, 'I':0}
        labels = [ldict[dataset[i]['label']] for i in range(len(dataset))]
        dsdict['label'] = np.array(labels, dtype=int)

        self.lmdb_data = {key: torch.from_numpy(val) for key, val in dsdict.items()}
        print("Done loading from {}.".format(folder))
        return

    # Credits to https://github.com/drorlab/atom3d/blob/master/examples/lep/enn/data.py#L13
    def preprocess_lmdb(self):
        # Get the size of all parts of the dataset
        ds_sizes = [len(self.lmdb_data[key]) for key in self.lmdb_data.keys()]
        # Make sure all parts of the dataset have the same length
        for size in ds_sizes[1:]:
            assert size == ds_sizes[0]

        # Set the dataset size
        self.num_pts = ds_sizes[0]

        # Get a dictionary of statistics for all properties that are one-dimensional tensors.
        self.calc_stats()
        return

    def process(self):
        print("Preprocessing LEP {} ...".format(self.split_option))

        # first, load LMDB format
        self.load_lmdb()

        # second, preprocess using the LMDB format
        self.preprocess_lmdb()

        # third, transform it into pyg format
        data_list = []
        for i in tqdm(range(self.__lmdb_len__())):
            lmdb_data = self.__lmdb_getitem__(i)
            
            to_keep1 = (lmdb_data['charges_active'] > 0).sum()
            to_keep2 = (lmdb_data['charges_inactive'] > 0).sum()
            num_atoms_active = lmdb_data["num_atoms_active"]
            num_atoms_inactive = lmdb_data["num_atoms_inactive"]
            assert to_keep1 == num_atoms_active
            assert to_keep2 == num_atoms_inactive

            charges_active = lmdb_data["charges_active"][:num_atoms_active].long()
            charges_inactive = lmdb_data["charges_inactive"][:num_atoms_inactive].long()

            x_active = []
            for charge in charges_active:
                atom_features = charge - 1
                x_active.append(atom_features)
            x_inactive = []
            for charge in charges_inactive:
                atom_features = charge - 1
                x_inactive.append(atom_features)
            x_active = torch.tensor(x_active, dtype=torch.long)
            x_inactive = torch.tensor(x_inactive, dtype=torch.long)

            positions_active = lmdb_data["positions_active"][:num_atoms_active].float()
            positions_inactive = lmdb_data["positions_inactive"][:num_atoms_inactive].float()

            label = lmdb_data["label"]
            
            data = Data(
                x_active=x_active,
                positions_active=positions_active,
                x_inactive=x_inactive,
                positions_inactive=positions_inactive,
                y=label,
            )
            data_list.append(data)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return
