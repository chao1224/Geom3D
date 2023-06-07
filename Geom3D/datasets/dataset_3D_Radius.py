import os
import shutil
from itertools import repeat
import numpy as np

import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import radius_graph
from torch_geometric.utils import subgraph, to_networkx
from .dataset_3D import extend_graph


class MoleculeDataset3DRadius(InMemoryDataset):
    def __init__(self, root, preprcessed_dataset, radius, mask_ratio=0, remove_center=False, use_extend_graph=False):
        self.root = root
        self.dataset = preprcessed_dataset.dataset
        self.preprcessed_dataset = preprcessed_dataset
        self.radius = radius
        self.mask_ratio = mask_ratio
        self.remove_center = remove_center
        self.use_extend_graph = use_extend_graph
        
        # TODO: rotation_transform is left for the future
        # self.rotation_transform = preprcessed_dataset.rotation_transform
        self.transform = preprcessed_dataset.transform
        self.pre_transform = preprcessed_dataset.pre_transform
        self.pre_filter = preprcessed_dataset.pre_filter

        super(MoleculeDataset3DRadius, self).__init__(root, self.transform, self.pre_transform, self.pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print("Dataset: {}\nData: {}".format(self.dataset, self.data))

        return

    def mean(self):
        y = torch.stack([self.get(i).y for i in range(len(self))], dim=0)
        y = y.mean(dim=0)
        return y

    def std(self):
        y = torch.stack([self.get(i).y for i in range(len(self))], dim=0)
        y = y.std(dim=0)
        return y

    def subgraph(self, data):
        G = to_networkx(data)
        node_num = data.x.size()[0]
        sub_num = int(node_num * (1 - self.mask_ratio))

        idx_sub = [np.random.randint(node_num, size=1)[0]]
        idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])

        # BFS
        while len(idx_sub) <= sub_num:
            if len(idx_neigh) == 0:
                idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
                idx_neigh = set([np.random.choice(idx_unsub)])
            sample_node = np.random.choice(list(idx_neigh))

            idx_sub.append(sample_node)
            idx_neigh = idx_neigh.union(
                set([n for n in G.neighbors(idx_sub[-1])])).difference(set(idx_sub))

        idx_nondrop = idx_sub
        idx_nondrop.sort()

        edge_idx, edge_attr = subgraph(
            subset=idx_nondrop,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            relabel_nodes=True,
            num_nodes=node_num
        )
        data.edge_index = edge_idx
        data.edge_attr = edge_attr
        data.x = data.x[idx_nondrop]
        data.positions = data.positions[idx_nondrop]
        data.__num_nodes__ = data.x.size()[0]

        radius_edge_index, _ = subgraph(
            subset=idx_nondrop,
            edge_index=data.radius_edge_index,
            relabel_nodes=True, 
            num_nodes=node_num)
        data.radius_edge_index = radius_edge_index
        
        if "extended_edge_index" in data:
            extended_edge_index, _ = subgraph(
                subset=idx_nondrop,
                edge_index=data.extended_edge_index,
                relabel_nodes=True, 
                num_nodes=node_num)
            data.extended_edge_index = extended_edge_index
        # TODO: will also need to do this for other edge_index

        return data

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        if self.use_extend_graph:
            data = extend_graph(data)
        
        if self.mask_ratio > 0:
            data = self.subgraph(data)

        if self.remove_center:
            center = data.positions.mean(dim=0)
            data.positions -= center

        return data

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def process(self):
        print("Preprocessing on {} with Radius Edges ...".format(self.dataset))

        if self.dataset == "qm9":
            print("Preprocessing on QM9 Radius ...")
            preprocessed_smiles_path = os.path.join(self.preprcessed_dataset.processed_dir, "smiles.csv")
            smiles_path = os.path.join(self.processed_dir, "smiles.csv")
            shutil.copyfile(preprocessed_smiles_path, smiles_path)
            
            preprocessed_data_name_file = os.path.join(self.preprcessed_dataset.processed_dir, "name.csv")
            data_name_file = os.path.join(self.processed_dir, "name.csv")
            shutil.copyfile(preprocessed_data_name_file, data_name_file)
        
        elif self.dataset == "md17":
            print("Preprocessing on MD17 Radius ...")
            pass
        
        elif self.dataset == "Molecule3D":
            print("Preprocessing on Molecule3D Radius ...")
            preprocessed_smiles_path = os.path.join(self.preprcessed_dataset.processed_dir, "smiles.csv")
            smiles_path = os.path.join(self.processed_dir, "smiles.csv")
            shutil.copyfile(preprocessed_smiles_path, smiles_path)
        
        elif "GEOM" in self.dataset:
            print("Preprocessing on GEOM Radius ...")
            preprocessed_smiles_path = os.path.join(self.preprcessed_dataset.processed_dir, "smiles.csv")
            smiles_path = os.path.join(self.processed_dir, "smiles.csv")
            shutil.copyfile(preprocessed_smiles_path, smiles_path)

        data_list = []        
        for i in tqdm(range(len(self.preprcessed_dataset))):
            data = self.preprcessed_dataset.get(i)
            radius_edge_index = radius_graph(data.positions, r=self.radius, loop=False)
            data.radius_edge_index = radius_edge_index
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        return
