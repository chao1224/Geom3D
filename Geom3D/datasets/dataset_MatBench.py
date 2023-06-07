import os
from tqdm import tqdm
from ase.io import read
from itertools import repeat
import numpy as np
import json
from tqdm import tqdm

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

from Geom3D.datasets.dataset_utils import get_graphs_within_cutoff, preiodic_augmentation_with_lattice, make_edges_into_two_direction


class DatasetMatBench(InMemoryDataset):
    def __init__(self, root, task, cutoff, periodic_data_augmentation, max_neighbours=None, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.dataset = "MatBench"
        self.task = task
        self.cutoff = cutoff
        self.max_neighbours = max_neighbours
        self.periodic_data_augmentation = periodic_data_augmentation

        super(DatasetMatBench, self).__init__(self.root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        return
    
    @property
    def raw_file_names(self):
        file_name = "{}.json".format(self.task)
        return file_name

    @property
    def raw_dir(self):
        return os.path.join(self.root)

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
        
        print("task: {}".format(self.task))
        json_file = os.path.join(self.raw_dir, self.raw_file_names)
        f = open(json_file, "r")
        json_data = json.load(f)
        print(json_data.keys())
        json_index_list = json_data["index"]
        json_columns_list = json_data["columns"]
        json_data_list = json_data["data"]
        print("json_index_list", len(json_index_list))
        print("json_data_list", len(json_data_list))
        print("json_columns_list", json_columns_list)
        
        data_list = []
        for json_index, json_data in tqdm(zip(json_index_list, json_data_list)):
            crystal_structure = Structure.from_dict(json_data[0])
            label = json_data[1]
            lattice = crystal_structure.lattice.matrix
            charge = crystal_structure.charge

            atom_num_list, positions_list = [], []

            for i in range(len(crystal_structure)):
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
    task_list = [
        # "expt_is_metal",  # composition / no structure
        # "expt_gap",  # composition / no structure
        # "glass",  # composition / no structure
        "perovskites",
        "dielectric",
        "log_gvrh",
        "log_kvrh",
        "jdft2d",
        # "steels",  # composition / no structure
        "phonons",
        "mp_is_metal",
        "mp_e_form",
        "mp_gap",
    ]
    cutoff_list = [5, 10]

    task_list = [
        "phonons"
    ]
    cutoff_list = [5]

    for cutoff in cutoff_list:
        for task in task_list:
            dataset = DatasetMatBench("../../data/MatBench", task=task, cutoff=cutoff, periodic_data_augmentation="image_gathered")
            print()
