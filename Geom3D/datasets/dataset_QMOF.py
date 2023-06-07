import pandas as pd
import os
from tqdm import tqdm
from itertools import repeat
from ase.io import read
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

from Geom3D.datasets.dataset_utils import get_graphs_within_cutoff, preiodic_augmentation_with_lattice, make_edges_into_two_direction


class DatasetQMOF(InMemoryDataset):
    def __init__(self, root, task, cutoff, periodic_data_augmentation, max_neighbours=None, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.dataset = "QMOF"
        self.task = task  # BG_PBE, E_PBE
        self.cutoff = cutoff
        self.max_neighbours = max_neighbours
        self.periodic_data_augmentation = periodic_data_augmentation
        
        super(DatasetQMOF, self).__init__(self.root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        return
    
    @property
    def raw_file_names(self):
        if self.task == "BG_PBE":
            file_name = "qmof-bandgaps.csv"
        elif self.task == "E_PBE":
            file_name = "qmof-energies.csv"
        return file_name

    @property
    def raw_dir(self):
        return os.path.join(self.root, "qmof_database")

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    @property
    def processed_dir(self):
        if int(self.cutoff) == self.cutoff:
            cutoff_ = int(self.cutoff)
        else:
            cutoff_ = self.cutoff
        
        if self.max_neighbours is not None:
            return os.path.join(self.root, "{}_{}_{}_neighbor".format(self.task, cutoff_, self.max_neighbours), "processed")
        else:
            return os.path.join(self.root, "{}_{}".format(self.task, cutoff_), "processed")

    def get(self, idx):
        if self.periodic_data_augmentation == "image_gathered":
            target_keys = ["expanded_x", "y", "expanded_positions", "expanded_edge_index", "gathered_x", "periodic_index_mapping"]
        elif self.periodic_data_augmentation == "image_expanded":
            target_keys = ["expanded_x", "y", "expanded_positions", "expanded_edge_index"]

        data = Data()
        for key in target_keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            neo_key = key.replace("expanded_", "")
            s[data.__cat_dim__(neo_key, item)] = slice(slices[idx], slices[idx + 1])
            data[neo_key] = item[s]
        return data

    def process(self):
        from pymatgen.core.structure import Structure
        
        df = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names))
        refcode_list = df["refcode"].tolist()
        label_list = df[self.task].tolist()
        refcode2label = {}
        for refcode, label in zip(refcode_list, label_list):
            refcode2label[refcode] = label
        print("len of refcode2label", len(refcode2label))
        
        data_list = []
        for refcode in tqdm(refcode_list):
            file_name = os.path.join(self.raw_dir, "qmof-cifs", "{}.cif".format(refcode))
            label = refcode2label[refcode]

            crystal_structure = Structure.from_file(file_name)
            lattice = crystal_structure.lattice._matrix
            charge = crystal_structure.charge
            
            atom_num_list, positions_list = [], []
            # symbol_list = np.unique(crystal_structure.get_chemical_symbols())

            for i in range(len(crystal_structure)):
                # atom_type = crystal_structure[i].specie
                # print(atom_type)
                atom_num = crystal_structure[i].specie.number - 1
                atom_num_list.append(atom_num)

                positions = crystal_structure[i].coords
                positions_list.append(positions)

            # crystal augmentation with lattice shift
            center_and_image_edge_index_list, image_shift_list, range_distance_list = get_graphs_within_cutoff(crystal_structure, cutoff=self.cutoff, max_neighbours=self.max_neighbours)
            expanded_atom_num_list, expanded_positions_list, expanded_edge_index, expanded_edge_distance, periodic_index_mapping = preiodic_augmentation_with_lattice(
                atom_num_list=atom_num_list, positions_list=positions_list, lattice=lattice,
                center_and_image_edge_index_list=center_and_image_edge_index_list, image_shift_list=image_shift_list, range_distance_list=range_distance_list)

            center_and_image_edge_index_list, range_distance_list = make_edges_into_two_direction(center_and_image_edge_index_list, range_distance_list)
            edge_index = np.array(center_and_image_edge_index_list)
            edge_index = edge_index.T

            gathered_atom_num_list = torch.tensor(atom_num_list, dtype=torch.int64)
            gathered_positions_list = torch.tensor(positions_list, dtype=torch.float32)
            gathered_edge_index = torch.tensor(edge_index, dtype=torch.int64)
            gathered_edge_distance = torch.tensor(range_distance_list, dtype=torch.float32)

            expanded_atom_num_list = torch.tensor(expanded_atom_num_list, dtype=torch.int64)
            expanded_positions_list = torch.tensor(expanded_positions_list, dtype=torch.float32)
            expanded_edge_index = torch.tensor(expanded_edge_index, dtype=torch.int64)
            expanded_edge_distance = torch.tensor(expanded_edge_distance, dtype=torch.float32)

            periodic_index_mapping = torch.tensor(periodic_index_mapping, dtype=torch.int64)

            lattice = torch.tensor(lattice, dtype=torch.float32)
            
            data = Data(
                gathered_x=gathered_atom_num_list,
                gathered_positions=gathered_positions_list,
                gathered_edge_index=gathered_edge_index,
                gathered_edge_distance=gathered_edge_distance,

                expanded_x=expanded_atom_num_list,
                expanded_positions=expanded_positions_list,
                expanded_edge_index=expanded_edge_index,
                expanded_edge_distance=expanded_edge_distance,

                periodic_index_mapping=periodic_index_mapping,

                lattice=lattice,
                charge=charge,
                y=label,
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

        return


if __name__ == "__main__":
    cutoff_list = [5, 10]

    for cutoff in cutoff_list:
        dataset = DatasetQMOF("../../data/QMOF", task="BG_PBE", cutoff=cutoff, periodic_data_augmentation="image_gathered")
        dataset = DatasetQMOF("../../data/QMOF", task="E_PBE", cutoff=cutoff, periodic_data_augmentation="image_gathered")
