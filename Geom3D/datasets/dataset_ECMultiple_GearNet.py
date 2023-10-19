import os.path as osp
import os
import numpy as np
import warnings
from tqdm import tqdm
from sklearn.preprocessing import normalize
import h5py
import itertools
from collections import defaultdict

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


class SloppyStructureBuilder(Bio.PDB.StructureBuilder.StructureBuilder):
    """Cope with resSeq < 10,000 limitation by just incrementing internally."""

    def __init__(self, verbose=False):
        Bio.PDB.StructureBuilder.StructureBuilder.__init__(self)
        self.max_resseq = -1
        self.verbose = verbose

    def init_residue(self, resname, field, resseq, icode):
        """Initiate a new Residue object.
        Arguments:
            resname: string, e.g. "ASN"
            field: hetero flag, "W" for waters, "H" for hetero residues, otherwise blanc.
            resseq: int, sequence identifier
            icode: string, insertion code
        Return:
            None
        """
        if field != " ":
            if field == "H":
                # The hetero field consists of
                # H_ + the residue name (e.g. H_FUC)
                field = "H_" + resname
        res_id = (field, resseq, icode)

        if resseq > self.max_resseq:
            self.max_resseq = resseq

        if field == " ":
            fudged_resseq = False
            while self.chain.has_id(res_id) or resseq == 0:
                # There already is a residue with the id (field, resseq, icode)
                # resseq == 0 catches already wrapped residue numbers which
                # do not trigger the has_id() test.
                #
                # Be sloppy and just increment...
                # (This code will not leave gaps in resids... I think)
                #
                # XXX: shouldn't we also do this for hetero atoms and water??
                self.max_resseq += 1
                resseq = self.max_resseq
                res_id = (field, resseq, icode)  # use max_resseq!
                fudged_resseq = True

            if fudged_resseq and self.verbose:
                sys.stderr.write(
                    "Residues are wrapping (Residue "
                    + "('%s', %i, '%s') at line %i)."
                    % (field, resseq, icode, self.line_counter)
                    + ".... assigning new resid %d.\n" % self.max_resseq
                )
        residue = Residue(res_id, resname, self.segid)
        self.chain.add(residue)
        self.residue = residue
        return None


class DatasetECMultipleGearNet(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, split='train', percent=0.3):
        self.split = split
        self.root = root
        self.percent = percent

        self.letter_to_num = {
            'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
            'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
            'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
            'N': 2, 'Y': 18, 'M': 12, "X":20}

        super(DatasetECMultipleGearNet, self).__init__(
            root, transform, pre_transform, pre_filter)
        
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_dir(self):
        if self.split != "test":
            name = 'processed_ECMultiple_GearNet_{}'.format(self.split)
            return osp.join(self.root, name)
        else:
            name = 'processed_ECMultiple_test_GearNet_{}'.format(self.percent)
            return osp.join(self.root, name)

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

    def process(self):  
        print('Beginning Processing ...')

        from torchdrug import transforms, layers
        from torchdrug.layers import geometry

        self.transform = transforms.ProteinView("residue")

        if self.split != "test":
            with open(os.path.join(self.root, f"nrPDB-EC_{self.split}.txt"), 'r') as file:
                self.data = set([line.strip() for line in file])
        else:
            self.data = set()
            with open(os.path.join(self.root, "nrPDB-EC_test.csv"), 'r') as f:
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

        structure_file_dir = os.path.join(
            self.root, f"{self.split}"
        )
        files = os.listdir(structure_file_dir)


        level_idx = 1
        ec_cnt = 0
        ec_num = {}
        ec_annotations = {}
        self.labels = {}

        with open(os.path.join(self.root, 'nrPDB-EC_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1:
                    arr = line.rstrip().split('\t')
                    for ec in arr:
                        ec_annotations[ec] = ec_cnt
                        ec_num[ec] = 0
                        ec_cnt += 1

                elif idx > 2:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_ec_list = arr[level_idx]
                        protein_ec_list = protein_ec_list.split(',')
                        for ec in protein_ec_list:
                            if len(ec) > 0:
                                protein_labels.append(ec_annotations[ec])
                                ec_num[ec] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_class = len(ec_annotations)

        graph_construction_model = layers.GraphConstruction(
            node_layers=[geometry.AlphaCarbonNode()], 
            edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5), geometry.KNNEdge(k=10, min_distance=5), geometry.SequentialEdge(max_distance=2)],
            edge_feature="gearnet")

        data_list = []
        for i in tqdm(range(len(files))):
            if files[i].split("_")[0] in self.data and files[i].split("_")[0] not in ["2UV2-A", "1ENM-A", "1DIN-A"]:
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
