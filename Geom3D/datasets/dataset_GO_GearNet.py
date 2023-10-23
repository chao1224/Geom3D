import os.path as osp
import os
import numpy as np
import warnings
from tqdm import tqdm
from sklearn.preprocessing import normalize
from collections import defaultdict
import h5py
import itertools
import argparse

import torch, math
import torch.nn.functional as F
import torch_cluster

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa
import sys
import Bio.PDB
import Bio.PDB.StructureBuilder
from Bio.PDB.Residue import Residue

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


class DatasetGOGearNet(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, split='train', level="mf", percent=0.95):
        self.split = split
        self.root = root
        self.level = level
        self.percent = percent

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12, "X":20}

        super(DatasetGOGearNet, self).__init__(
            root, transform, pre_transform, pre_filter)
        
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_dir(self):
        name = 'processed_GO_GearNet_' + self.level
        if self.split != "test":
            return osp.join(self.root, name, self.split)
        else:
            return osp.join(self.root, name, self.split + "_" + str(self.percent))

    @property
    def raw_file_names(self):
        name = self.split + '.txt'
        return name

    @property
    def processed_file_names(self):
        return 'data.pt'


    def extract_protein_data(self, pFilePath, graph_construction_model):
        from torchdrug import data
        protein = data.Protein.from_pdb(pFilePath)
        protein = data.Protein.pack([protein])
        protein = graph_construction_model(protein)
        item = {"graph": protein}

        if self.transform:
            item = self.transform(item)
        
        protein = item["graph"]
        seq = protein.to_sequence()[0]
        residue_type = []
        residue_feature = []
        
        for i in seq:
            residue_type.append(data.Protein.residue_symbol2id.get(i, 0))
            residue_feature.append(data.feature.onehot(data.Protein.id2residue.get(data.Protein.residue_symbol2id.get(i)), data.feature.residue_vocab, allow_unknown=True))
        return_data = Data()
        return_data.edge_list = protein.edge_list
        return_data.edge_weight = torch.ones(len(protein.edge_list))
        return_data.num_residue = protein.num_residue
        return_data.num_node = protein.num_node
        return_data.num_edge = protein.num_edge
        return_data.x = residue_type  # This is important to hack the code
        return_data.node_feature = residue_feature
        return_data.num_relation = protein.num_relation
        return_data.node_position = protein.node_position
        return_data.edge_feature = protein.edge_feature

        return return_data


    def collate_fn(batch):
        num_nodes = []
        num_edges = []
        num_residues = []
        node_positions = []
        y = []
        num_cum_node = 0
        num_cum_edge = 0
        num_cum_residue = 0
        num_graph = 0
        data_dict = defaultdict(list)

        for graph in batch:
            num_nodes.append(graph.num_node)
            num_edges.append(graph.num_edge)
            num_residues.append(graph.num_residue)
            node_positions.append(graph.node_position)
            y.append(graph.y[0])
            for k, v in graph.items():
                if k in ["num_relation", "num_node", "num_edge", "num_residue", "node_position", "y", "id"]:
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

        data_dict = {k: torch.cat([torch.tensor(v) for v in lst]) for k, lst in data_dict.items()}

        num_nodes = torch.cat(num_nodes)
        num_edges = torch.cat(num_edges)
        num_residues = torch.cat(num_residues)
        node_positions = torch.cat(node_positions)
        node2graph = torch.repeat_interleave(num_nodes)
        num_node = torch.sum(num_nodes)
        num_edge = torch.sum(num_edges)

        return Data(
            num_nodes=num_nodes, num_node=num_node, num_edges=num_edges, num_edge=num_edge, num_residues=num_residues, num_relation=batch[0].num_relation, node_position=node_positions,
            node2graph=node2graph, batch_size=len(batch), y=torch.stack(y), **data_dict)        

    def process(self):  
        print('Beginning Processing ...')

        from torchdrug import transforms, layers
        from torchdrug.layers import geometry

        self.transform = transforms.ProteinView("residue")

        if self.split != "test":
            with open(os.path.join(self.root, f"nrPDB-GO_{self.split}.txt"), 'r') as file:
                self.data = set([line.strip() for line in file])
        else:
            self.data = set()
            with open(os.path.join(self.root, "nrPDB-GO_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if self.percent == 0.3 and arr[1] == '1':
                        self.data.add(arr[0])
                    elif self.percent == 0.4 and arr[2] == '1':
                        self.data.add(arr[0])
                    elif self.percent == 0.5 and arr[3] == '1':
                        self.data.add(arr[0])
                    elif self.percent == 0.7 and arr[4] == '1':
                        self.data.add(arr[0])
                    elif self.percent == 0.95 and arr[5] == '1':
                        self.data.add(arr[0])
                    else:
                        pass

        structure_file_dir = osp.join(
            self.root, f"{self.split}"
        )
        files = os.listdir(structure_file_dir)

        level_idx = 0
        go_cnt = 0
        go_num = {}
        go_annotations = {}
        self.labels = {}
        with open(osp.join(self.root, 'nrPDB-GO_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1 and self.level == "mf":
                    level_idx = 1
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 5 and self.level == "bp":
                    level_idx = 2
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 9 and self.level == "cc":
                    level_idx = 3
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx > 12:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_go_list = arr[level_idx]
                        protein_go_list = protein_go_list.split(',')
                        for go in protein_go_list:
                            if len(go) > 0:
                                protein_labels.append(go_annotations[go])
                                go_num[go] += 1
                    self.labels[arr[0]] = np.array(protein_labels)
        
        self.num_class = len(go_annotations)

        graph_construction_model = layers.GraphConstruction(
            node_layers=[geometry.AlphaCarbonNode()], 
            edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5), geometry.KNNEdge(k=10, min_distance=5), geometry.SequentialEdge(max_distance=2)],
            edge_feature="gearnet")

        data_list = []
        for i in tqdm(range(len(files))):
            if files[i].split("_")[0] in self.data and files[i].split("_")[0] not in ["1X18-E", "2UV2-A", "1EIS-A", "4UPV-Q", "1DIN-A"]:
                file_name = osp.join(self.root, self.split, files[i])
                try:
                    protein = self.extract_protein_data(file_name, graph_construction_model)
                except:
                    protein = None
                label = np.zeros((self.num_class,)).astype(np.float32)

                if len(self.labels[osp.basename(file_name).split("_")[0]]) > 0:
                    label[self.labels[osp.basename(file_name).split("_")[0]]] = 1.0

                if protein is not None:
                    protein.id = files[i]
                    protein.y = torch.tensor(label).unsqueeze(0)
                    data_list.append(protein)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('Done!')
