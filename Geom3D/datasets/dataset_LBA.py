import os
import numpy as np
import pandas as pd
import scipy as sp
import copy
from tqdm import tqdm
from itertools import repeat
from collections import defaultdict
import Bio
import json

import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset

import atom3d.protein.sequence as seq
import atom3d.util.formats as fo
from Geom3D.datasets.PDBBind_utils import get_pocket_res, get_pdb_code, PocketSelect, find_files, identity_split, write_index_to_file


# Credit to https://github.com/drorlab/atom3d/blob/master/examples/lba/enn/data.py#L294
class TransformLBA:
    def __init__(self, dist, maxnum, move_lig=True):
        self._dist = dist
        self._maxnum = maxnum
        self._dx = 0
        if move_lig:
            self._dx = 1000

    def _move(self, df):
       df_moved = copy.deepcopy(df) 
       df_moved["x"] += self._dx
       return df_moved 

    def _drop_hydrogen(self, df):
        df_noh = df[df["element"] != "H"]
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

    def _select_env_by_dist(self, pocket, ligand):
        # Extract coordinates
        ligand_coords = np.array([ligand.x, ligand.y, ligand.z]).T
        pocket_coords = np.array([pocket.x, pocket.y, pocket.z]).T
        # Select the environment around the mutated residue
        kd_tree = sp.spatial.KDTree(pocket_coords)
        key_pts = kd_tree.query_ball_point(ligand_coords, r=self._dist, p=2.0)
        key_pts = np.unique([k for l in key_pts for k in l])
        # Construct the new data frame
        new_pocket = pd.concat([ pocket.iloc[key_pts] ], ignore_index=True)
        return new_pocket

    def _select_env_by_num(self, pocket, ligand):
        # Max. number of protein atoms 
        num = int(max([1, self._maxnum - len(ligand.x)]))
        #print("Select a maximum of",num,"atoms.")
        # Extract coordinates
        ligand_coords = np.array([ligand.x, ligand.y, ligand.z]).T
        pocket_coords = np.array([pocket.x, pocket.y, pocket.z]).T
        # Select the environment around the mutated residue
        kd_tree = sp.spatial.KDTree(pocket_coords)
        dd, ii = kd_tree.query(ligand_coords, k=len(pocket.x), p=2.0)
        # Get minimum distance to any lig atom for each protein atom
        dis = [ min(dd[ii==j]) for j in range(len(pocket.x)) ]
        # Sort indices by distance
        idx = np.argsort(dis)
        # Select the num closest atoms
        idx = np.sort(idx[:num])
        # Construct the new data frame
        new_pocket = pd.concat([ pocket.iloc[idx] ], ignore_index=True)
        return new_pocket

    def __call__(self, pocket_df, ligand_df):
        # Replace rare atoms
        pocket_df = self._replace(pocket_df)
        ligand_df = self._replace(ligand_df)
        # Drop hydrogen atoms
        pocket_df = self._drop_hydrogen(pocket_df)
        ligand_df = self._drop_hydrogen(ligand_df)
        # Select the environment
        pocket_df = self._select_env_by_dist(pocket_df,ligand_df)
        pocket_df = self._select_env_by_num(pocket_df,ligand_df)

        ligand_moved_df = None
        # Move the ligand far away 
        if self._dx != 0:
            # shift x-axis of the ligand by a large margin,
            # then we don't need to change the code, while the ligand and protein are well separated
            ligand_moved_df = self._move(ligand_df)
        return pocket_df, ligand_df, ligand_moved_df


class DatasetLBA(InMemoryDataset):
    def __init__(
        self,
        root,
        year="2020",
        dist=6,
        dataframe_transformer=None,
        use_complex=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        self.pre_transform, self.pre_filter = pre_transform, pre_filter
        self.year = year
        self.urlmap = {
            "2015": "http://www.pdbbind.org.cn/download/pdbbind_v2015_refined_set.tar.gz",
            "2018": "http://www.pdbbind.org.cn/download/pdbbind_v2018_refined.tar.gz",
            "2019": "http://www.pdbbind.org.cn/download/pdbbind_v2019_refined.tar.gz",
            "2020": "http://www.pdbbind.org.cn/download/PDBbind_v2020_refined.tar.gz",
        }
        self.dataset = "lba"
        self.dist = dist
        self.sanitize = False
        self.add_hs = False
        self.remove_hs = False
        self.dataframe_transformer = dataframe_transformer
        self.use_complex=use_complex

        os.makedirs(root, exist_ok=True)
        os.makedirs(os.path.join(root, "raw"), exist_ok=True)
        os.makedirs(os.path.join(root, "intermediate"), exist_ok=True)
        os.makedirs(os.path.join(root, "processed"), exist_ok=True)
        self.intermediate_dir = os.path.join(root, "intermediate")

        super(DatasetLBA, self).__init__(root, transform, pre_transform, pre_filter)

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

    def get_split(self):
        return

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return "geometric_data_processed_{}.pt".format(self.year)

    def process(self):
        raw_input_dir = os.path.join(self.raw_dir, "refined-set")
        print("processing from", raw_input_dir)

        structure_dict = defaultdict(dict)

        # load protein
        pdb_files = find_files(raw_input_dir, "pdb")
        for f in tqdm(pdb_files, desc="loading proteins"):
            f = str(f)
            pdb_id = get_pdb_code(f)
            if "_protein" in f:
                protein = fo.read_any(f)
                structure_dict[pdb_id]["protein"] = protein

        # load ligand
        lig_files = find_files(raw_input_dir, "sdf")
        for f in tqdm(lig_files, desc="loading ligands"):
            f = str(f)
            pdb_id = get_pdb_code(f)
            ligand = Chem.SDMolSupplier(f, sanitize=self.sanitize, removeHs=self.remove_hs)[0]
            if self.add_hs and ligand is not None:
                ligand = Chem.AddHs(ligand, addCoords=True)
            structure_dict[pdb_id]["ligand"] = ligand

        # load pocket
        for pdb_id, data in tqdm(structure_dict.items(), desc="loading pockets"):
            protein = structure_dict[pdb_id]["protein"]
            ligand = structure_dict[pdb_id]["ligand"]
            if ligand is None:
                continue
            pocket_res = get_pocket_res(protein, ligand, self.dist)
            pocket = PocketSelect(pocket_res)
            structure_dict[pdb_id]["pocket"] = pocket

        index_file = os.path.join(self.raw_dir, "refined-set", "index", "INDEX_refined_data.{}".format(self.year))
        with open(index_file) as f:
            for line in f:
                line = str(line)
                if line.startswith("#"):
                    continue
                l = line.strip().split()
                pdb_id = l[0]
                if pdb_id not in structure_dict:
                    continue
                structure_dict[pdb_id]["logKd_label"] = float(l[3])

        data_list = []
        all_chain_sequences = []
        pdb_id2data_id = {}
        for data_id, (pdb_id, data) in enumerate(tqdm(structure_dict.items(), desc="creating pyg data")):
            protein, pocket, ligand, logKd_label = data["protein"], data["pocket"], data["ligand"], data["logKd_label"]
            pdb_id2data_id[pdb_id] = data_id
            protein_df = fo.bp_to_df(protein)
            protein_seq = seq.get_chain_sequences(protein_df)
            all_chain_sequences.append(protein_seq)

            # write protein to mmCIF file
            io = Bio.PDB.MMCIFIO()
            io.set_structure(protein)
            io.save(os.path.join(self.intermediate_dir, f"{pdb_id}_protein.cif"))

            # write pocket to mmCIF file
            io.save(os.path.join(self.intermediate_dir, f"{pdb_id}_pocket.cif"), pocket)
            neo_pocket = fo.read_mmcif(os.path.join(self.intermediate_dir, f"{pdb_id}_pocket.cif"))
            pocket_df = fo.bp_to_df(neo_pocket)

            m = ligand
            ligand_df = fo.mol_to_df(
                m, residue=data_id,
                ensemble=m.GetProp("_Name"), structure=m.GetProp("_Name"), model=m.GetProp("_Name"))

            pocket_df, ligand_df, ligand_moved_df = self.dataframe_transformer(pocket_df=pocket_df, ligand_df=ligand_df)
            # TODO: ligand_moved_df is related to
            # https://github.com/drorlab/atom3d/blob/master/examples/lba/enn/data.py#L223
            # controlled by self.use_complex

            merged_df = pd.concat([pocket_df, ligand_df], ignore_index=True)

            atom_features_list, position_list = [], []
            for ia in range(len(merged_df)):
                atom_type = merged_df['element'][ia]
                atomic_number = fo.atomic_number[atom_type]
                atom_features = atomic_number - 1
                x = merged_df['x'][ia]
                y = merged_df['y'][ia]
                z = merged_df['z'][ia]
                atom_features_list.append(atom_features)
                position_list.append([x, y, z])

            atom_features_list = torch.tensor(atom_features_list, dtype=torch.long)
            position_list = torch.tensor(position_list, dtype=torch.float)

            data = Data(
                x=atom_features_list,
                positions=position_list,
                y=logKd_label
            )         

            data_list.append(data)
        
        # TODO: double-check the data splitting
        # train_indices, val_indices, test_indices = identity_split(all_chain_sequences, cutoff=0.3)
        # print('train_indices: {}\tval_indices: {}\ttest_indices: {}'.format(
        #     train_indices, val_indices, test_indices))
        
        # write_index_to_file(train_indices, os.path.join(self.root, 'processed', 'train_indices.txt'))
        # write_index_to_file(val_indices, os.path.join(self.root, 'processed', 'val_indices.txt'))
        # write_index_to_file(test_indices, os.path.join(self.root, 'processed', 'test_indices.txt'))

        with open(os.path.join(self.root, 'processed', 'pdb_id2data_id_{}.json'.format(self.year)), 'w') as outfile:
            json.dump(pdb_id2data_id, outfile)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        return
