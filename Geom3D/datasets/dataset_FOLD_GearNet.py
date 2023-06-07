import os
import h5py
import torch
import warnings
from tqdm import tqdm
from collections import defaultdict
import os.path as osp
import copy

import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


class DatasetFOLDGearNet(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, split='train'):
        self.split = split
        self.root = root

        super(DatasetFOLDGearNet, self).__init__(
            root, transform, pre_transform, pre_filter)
        
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = 'processed_GearNet'
        return osp.join(self.root, name, self.split)

    @property
    def raw_file_names(self):
        name = self.split + '.txt'
        return name

    @property
    def processed_file_names(self):
        return 'data.pt'

    def protein_to_graph(self, pFilePath, graph_construction_model):
        from torchdrug import data

        h5File = h5py.File(pFilePath, "r")

        node_position = torch.as_tensor(h5File["atom_pos"][(0)])
        num_atom = node_position.shape[0]
        atom_type = torch.as_tensor(h5File["atom_types"][()])
        atom_name = h5File["atom_names"][()]
        atom_name = torch.as_tensor([data.Protein.atom_name2id.get(name.decode(), -1) for name in atom_name])
        atom2residue = torch.as_tensor(h5File["atom_residue_id"][()])
        residue_type_name = h5File["atom_residue_names"][()]
        residue_type = []
        residue_feature = []
        lst_residue = -1
        for i in range(num_atom):
            if atom2residue[i] != lst_residue:
                residue_type.append(data.Protein.residue2id.get(residue_type_name[i].decode(), 0))
                residue_feature.append(data.feature.onehot(residue_type_name[i].decode(), data.feature.residue_vocab, allow_unknown=True))
                lst_residue = atom2residue[i]
        residue_type = torch.as_tensor(residue_type)
        residue_feature = torch.as_tensor(residue_feature)
        num_residue = residue_type.shape[0]

        edge_list = torch.cat([
            torch.as_tensor(h5File["cov_bond_list"][()]),
            torch.as_tensor(h5File["cov_bond_list_hb"][()])
        ], dim=0)
        bond_type = torch.zeros(edge_list.shape[0], dtype=torch.long)
        edge_list = torch.cat([edge_list, bond_type.unsqueeze(-1)], dim=-1)

        protein = data.Protein(
            edge_list, atom_type, bond_type, num_node=num_atom, num_residue=num_residue,
            node_position=node_position, atom_name=atom_name,
            atom2residue=atom2residue, residue_feature=residue_feature, 
            residue_type=residue_type)

        protein = data.Protein.pack([protein])
        protein = graph_construction_model(protein)

        return_data = Data()
        return_data.edge_list = protein.edge_list
        return_data.edge_weight = torch.ones(len(protein.edge_list))
        return_data.num_residue = protein.num_residue
        return_data.num_node = protein.num_node
        return_data.num_edge = protein.num_edge
        return_data.x = atom_type  # This is important to hack the code
        return_data.node_feature = F.one_hot(atom_type, num_classes=21)
        return_data.num_relation = protein.num_relation
        return return_data

    def process(self):
        print('Beginning Processing ...')
        # This requires the installment of TorchDrug

        from torchdrug import transforms, layers
        from torchdrug.layers import geometry

        graph_construction_model = layers.GraphConstruction(
            node_layers=[geometry.AlphaCarbonNode()], 
            edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5), geometry.KNNEdge(k=10, min_distance=5), geometry.SequentialEdge(max_distance=2)],
            edge_feature="gearnet")

        # Load the file with the list of functions.
        classes_ = {}
        with open(self.root+"/class_map.txt", 'r') as mFile:
            for line in mFile:
                lineList = line.rstrip().split('\t')
                classes_[lineList[0]] = int(lineList[1])

        # Get the file list.
        fileList_ = []
        cathegories_ = []
        with open(self.root+"/"+self.split+".txt", 'r') as mFile:
            for curLine in mFile:
                splitLine = curLine.rstrip().split('\t')
                curClass = classes_[splitLine[-1]]
                fileList_.append(self.root+"/"+self.split+"/"+splitLine[0])
                cathegories_.append(curClass)

        # Load the dataset
        print("Reading the data")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_list = []
            for fileIter, curFile in tqdm(enumerate(fileList_)):
                fileName = curFile.split('/')[-1]
                curProtein = self.protein_to_graph(curFile+".hdf5", graph_construction_model=graph_construction_model)
                # curProtein.id = fileName           
                curProtein.y = torch.tensor(cathegories_[fileIter])
                if not curProtein.num_node is None:
                    data_list.append(curProtein)     
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('Done!')

    def collate_fn(batch):
        num_nodes = []
        num_edges = []
        num_residues = []
        num_cum_node = 0
        num_cum_edge = 0
        num_cum_residue = 0
        num_graph = 0
        data_dict = defaultdict(list)
        for graph in batch:
            num_nodes.append(graph.num_node)
            num_edges.append(graph.num_edge)
            num_residues.append(graph.num_residue)
            for k, v in graph.items():
                if k in ["num_relation", "num_node", "num_edge", "num_residue"]:
                    continue
                elif k in ["edge_list"]:
                    neo_v = v.clone()
                    neo_v[:, 0] += num_cum_node
                    neo_v[:, 1] += num_cum_node
                    data_dict[k].append(neo_v)
                    continue

                data_dict[k].append(v)
            num_cum_node += graph.num_node
            num_cum_edge += graph.num_edge
            num_cum_residue += graph.num_residue
            num_graph += 1

        data_dict = {k: torch.cat(v) for k, v in data_dict.items()}
        
        num_nodes = torch.cat(num_nodes)
        num_edges = torch.cat(num_edges)
        num_residues = torch.cat(num_residues)
        node2graph = torch.repeat_interleave(num_nodes)
        num_node = torch.sum(num_nodes)
        num_edge = torch.sum(num_edges)

        return Data(
            num_node=num_node, num_nodes=num_nodes, num_edge=num_edge, num_edges=num_edges, num_residues=num_residues, num_relation=batch[0].num_relation,
            node2graph=node2graph, **data_dict)
        