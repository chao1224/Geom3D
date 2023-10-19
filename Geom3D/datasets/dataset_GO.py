import os.path as osp
import os
import numpy as np
import warnings
from tqdm import tqdm
from sklearn.preprocessing import normalize
import h5py

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
    


class DatasetGO(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, split='train', level= "mf", percent=0.3):
        self.split = split
        self.root = root
        self.level = level
        self.percent = percent

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12, "X":20}

        super(DatasetGO, self).__init__(
            root, transform, pre_transform, pre_filter)
        
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_dir(self):
        name = 'processed_GO_' + self.level
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

    def get_side_chain_angle_encoding(self, pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        v1, v2, v3, v4, v5, v6, v7 = pos_ca - pos_n, pos_cb - pos_ca, pos_g - pos_cb, pos_d - pos_g, pos_e - pos_d, pos_z - pos_e, pos_h - pos_z

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        angle1 = torch.unsqueeze(self.diherals_ProNet(v1, v2, v3),1)
        angle2 = torch.unsqueeze(self.diherals_ProNet(v2, v3, v4),1)
        angle3 = torch.unsqueeze(self.diherals_ProNet(v3, v4, v5),1)
        angle4 = torch.unsqueeze(self.diherals_ProNet(v4, v5, v6),1)
        angle5 = torch.unsqueeze(self.diherals_ProNet(v5, v6, v7),1)

        side_chain_angles = torch.cat((angle1, angle2, angle3, angle4),1)
        side_chain_embs = torch.cat((torch.sin(side_chain_angles), torch.cos(side_chain_angles)),1)
        
        return side_chain_embs
    
    def get_backbone_angle_encoding(self, X):   
        # X should be a num_residues x 3 x 3, order N, C-alpha, and C atoms of each residue
        # N coords: X[:,0,:]
        # CA coords: X[:,1,:]
        # C coords: X[:,2,:]
        # return num_residues x 6 
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X, [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self._normalize(dX, dim=-1)
        u0 = U[:-2]
        u1 = U[1:-1]
        u2 = U[2:]

        angle = self.diherals_ProNet(u0, u1, u2)
        
        # add phi[0], psi[-1], omega[-1] with value 0
        angle = F.pad(angle, [1, 2]) 
        angle = torch.reshape(angle, [-1, 3])
        angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
        return angle_features
    
    def diherals_ProNet(self, v1, v2, v3):
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))
        return torsion
    
    def _normalize(self, tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def three_to_one_standard(self, res):
        if not is_aa(res, standard=True):
            return "X"
        
        return three_to_one(res)

    def chain_info(self, chain, name):
        """Convert a PDB chain in to coordinates of target atoms from all
        AAs

        Args:
            chain: a Bio.PDB.Chain object
            target_atoms: Target atoms which residues will be resturned.
            name: String. Name of the protein.
        Returns:
            Dictonary containing protein sequence `seq`, 3D coordinates `coord` and name `name`.

        """
        atom_names, atom_amino_id, atom_pos, residue_types = [], [], [], []
        pdb_seq = ""
        residue_index = 0
        for residue in chain.get_residues():
            if is_aa(residue) and any(atom.get_name() == "CA" for atom in residue.get_atoms()):
                residue_name = self.three_to_one_standard(residue.get_resname())
                pdb_seq += residue_name
                residue_types.append(self.letter_to_num[residue_name])

                for atom in residue.get_atoms():
                    atom_names.append(atom.get_name())
                    atom_amino_id.append(residue_index)
                    atom_pos.append(atom.coord)
                    
                residue_index += 1

        mask_n = np.char.equal(atom_names, 'N')
        mask_ca = np.char.equal(atom_names, 'CA')
        mask_c = np.char.equal(atom_names, 'C')
        mask_cb = np.char.equal(atom_names, 'CB')
        mask_g = np.char.equal(atom_names, 'CG') | np.char.equal(atom_names, 'SG') | np.char.equal(atom_names, 'OG') | np.char.equal(atom_names, 'CG1') | np.char.equal(atom_names, 'OG1')
        mask_d = np.char.equal(atom_names, 'CD') | np.char.equal(atom_names, 'SD') | np.char.equal(atom_names, 'CD1') | np.char.equal(atom_names, 'OD1') | np.char.equal(atom_names, 'ND1')
        mask_e = np.char.equal(atom_names, 'CE') | np.char.equal(atom_names, 'NE') | np.char.equal(atom_names, 'OE1')
        mask_z = np.char.equal(atom_names, 'CZ') | np.char.equal(atom_names, 'NZ')
        mask_h = np.char.equal(atom_names, 'NH1')
        
        atom_amino_id = np.array(atom_amino_id)
        atom_pos = np.array(atom_pos)

        pos_n = np.full((len(pdb_seq), 3),np.nan)
        pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
        pos_n = torch.FloatTensor(pos_n)

        pos_ca = np.full((len(pdb_seq), 3),np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)

        pos_c = np.full((len(pdb_seq), 3),np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        pos_c = torch.FloatTensor(pos_c)

        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        pos_cb = np.full((len(pdb_seq), 3),np.nan)
        pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
        pos_cb = torch.FloatTensor(pos_cb)

        pos_g = np.full((len(pdb_seq), 3),np.nan)
        pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
        pos_g = torch.FloatTensor(pos_g)

        pos_d = np.full((len(pdb_seq), 3),np.nan)
        pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
        pos_d = torch.FloatTensor(pos_d)

        pos_e = np.full((len(pdb_seq), 3),np.nan)
        pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
        pos_e = torch.FloatTensor(pos_e)

        pos_z = np.full((len(pdb_seq), 3),np.nan)
        pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
        pos_z = torch.FloatTensor(pos_z)

        pos_h = np.full((len(pdb_seq), 3),np.nan)
        pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
        pos_h = torch.FloatTensor(pos_h)

        chain_struc = {
                    'name': name,
                    'pos_n': pos_n,
                    'pos_ca': pos_ca,
                    'pos_c': pos_c,
                    'pos_cb': pos_cb,
                    'pos_g': pos_g,
                    'pos_d': pos_d,
                    'pos_e': pos_e,
                    'pos_z': pos_z,
                    'pos_h': pos_h,
                    'atom_names': atom_names,
                    'atom_pos': atom_pos,
                    'residue_types': residue_types
                }

        if len(pdb_seq) <= 1:
            # has no or only 1 AA in the chain
            return None
        
        return chain_struc
    

    def extract_protein_data(self, pFilePath):
        data = Data()

        pdb_parser = PDBParser(
            QUIET=True,
            PERMISSIVE=True,
            structure_builder=SloppyStructureBuilder(),
        )
    
        name = os.path.basename(pFilePath).split("_")[0]

        try:
            structure = pdb_parser.get_structure(name, pFilePath)
        except Exception as e:
            print(pFilePath, "raised an error:")
            print(e)
            return None
        
        records = []
        chain_ids = []

        for chain in structure.get_chains():
            if chain.id in chain_ids:  # skip duplicated chains
                continue
            chain_ids.append(chain.id)
            record = self.chain_info(chain, "{}-{}".format(name.split("-")[0], chain.id))
            if record is not None:
                records.append(record)

        records = [rec for rec in records if rec["name"] in self.data]

        for i in records:
            if i["name"] == name:
                pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h, atom_names, atom_pos, residue_types = (i[k] for k in ["pos_n", "pos_ca", "pos_c", "pos_cb", "pos_g", "pos_d", "pos_e", "pos_z", "pos_h", "atom_names", "atom_pos", "residue_types"])

                # calculate side chain torsion angles, up to four
                # do encoding
                side_chain_angle_encoding = self.get_side_chain_angle_encoding(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
                side_chain_angle_encoding[torch.isnan(side_chain_angle_encoding)] = 0

                # three backbone torsion angles
                backbone_angle_encoding = self.get_backbone_angle_encoding(torch.cat((torch.unsqueeze(pos_n,1), torch.unsqueeze(pos_ca,1), torch.unsqueeze(pos_c,1)),1))
                backbone_angle_encoding[torch.isnan(backbone_angle_encoding)] = 0
                
                data.seq = torch.LongTensor(residue_types)
                data.side_chain_angle_encoding = side_chain_angle_encoding
                data.backbone_angle_encoding = backbone_angle_encoding
                data.coords_ca = pos_ca
                data.coords_n = pos_n
                data.coords_c = pos_c
                data.x = atom_names
                data.atom_pos = torch.tensor(atom_pos)
                data.num_nodes = len(pos_ca) 

                return data

    def process(self):  
        print('Beginning Processing ...')

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


        # 2. Parse the structure files and save to json files
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

        invalid_PDB_file_name_list = ["1X18-E_5719.pdb", "2UV2-A_11517.pdb", "1EIS-A_990.pdb", "4UPV-Q_24858.pdb", "1DIN-A_746.pdb"]
        
        data_list = []
        for i in tqdm(range(len(files))):
            if files[i].split("_")[0] in self.data:
                if files[i] in invalid_PDB_file_name_list:
                    print("Skipping invalid file {}...".format(files[i]))
                    continue
                file_name = osp.join(self.root, self.split, files[i])
                protein = self.extract_protein_data(file_name)
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

# if __name__ == "__main__":
#         pdb_parser = PDBParser(
#         QUIET=True,
#         PERMISSIVE=True,
#         structure_builder=SloppyStructureBuilder(),
#     )
        
#         for level in ["mf", "bp", "cc"]:
#             for split in ['test', 'train', 'valid']:
#                 print('#### Now processing {} data ####'.format(split))
#                 if split != "test":
#                     dataset = DatasetGO(root="/lustre07/scratch/liusheng/GearNet/GeneOntology", level=level, split=split)
#                 else:
#                     for cutoff in [0.3, 0.4, 0.5, 0.7, 0.95]:
#                         dataset = DatasetGO(root="/lustre07/scratch/liusheng/GearNet/GeneOntology", level=level, split=split, percent=cutoff)
                