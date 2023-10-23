import os.path as osp
import numpy as np
import warnings
from tqdm import tqdm
from sklearn.preprocessing import normalize
import h5py

import torch, math
import torch.nn.functional as F
import torch_cluster

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


class DatasetECSingle(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, split='train'):
        self.split = split
        self.root = root
 
        super(DatasetECSingle, self).__init__(
            root, transform, pre_transform, pre_filter)
        
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = 'processed_ECSingle'
        return osp.join(self.root, name, self.split)

    @property
    def raw_file_names(self):
        name = self.split + '.txt'
        return name

    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def _normalize(self, tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def get_key_atom_pos(self, amino_types, atom_names, atom_amino_id, atom_pos):
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        mask_n = np.char.equal(atom_names, b'N')
        mask_ca = np.char.equal(atom_names, b'CA')
        mask_c = np.char.equal(atom_names, b'C')
        mask_cb = np.char.equal(atom_names, b'CB')
        mask_g = np.char.equal(atom_names, b'CG') | np.char.equal(atom_names, b'SG') | np.char.equal(atom_names, b'OG') | np.char.equal(atom_names, b'CG1') | np.char.equal(atom_names, b'OG1')
        mask_d = np.char.equal(atom_names, b'CD') | np.char.equal(atom_names, b'SD') | np.char.equal(atom_names, b'CD1') | np.char.equal(atom_names, b'OD1') | np.char.equal(atom_names, b'ND1')
        mask_e = np.char.equal(atom_names, b'CE') | np.char.equal(atom_names, b'NE') | np.char.equal(atom_names, b'OE1')
        mask_z = np.char.equal(atom_names, b'CZ') | np.char.equal(atom_names, b'NZ')
        mask_h = np.char.equal(atom_names, b'NH1')

        pos_n = np.full((len(amino_types),3),np.nan)
        pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
        pos_n = torch.FloatTensor(pos_n)

        pos_ca = np.full((len(amino_types),3),np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)

        pos_c = np.full((len(amino_types),3),np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        pos_c = torch.FloatTensor(pos_c)

        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        pos_cb = np.full((len(amino_types),3),np.nan)
        pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
        pos_cb = torch.FloatTensor(pos_cb)

        pos_g = np.full((len(amino_types),3),np.nan)
        pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
        pos_g = torch.FloatTensor(pos_g)

        pos_d = np.full((len(amino_types),3),np.nan)
        pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
        pos_d = torch.FloatTensor(pos_d)

        pos_e = np.full((len(amino_types),3),np.nan)
        pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
        pos_e = torch.FloatTensor(pos_e)

        pos_z = np.full((len(amino_types),3),np.nan)
        pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
        pos_z = torch.FloatTensor(pos_z)

        pos_h = np.full((len(amino_types),3),np.nan)
        pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
        pos_h = torch.FloatTensor(pos_h)

        return pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h
    
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

    def extract_protein_data(self, pFilePath):
        h5File = h5py.File(pFilePath+".hdf5", "r")
        data = Data()
 
        amino_types = h5File['amino_types'][()] # residue or amino acid, size: (n_amino,)
        mask = amino_types == -1
        if np.sum(mask) > 0:
            amino_types[mask] = 25 # for amino acid types, set the value of -1 to 25
        atom_amino_id = h5File['atom_amino_id'][()] # size: (n_atom,)
        atom_names = h5File['atom_names'][()] # size: (n_atom,)
        atom_pos = h5File['atom_pos'][()][0] #size: (n_atom,3)

        ##### compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1 #####
        # extract key atom (e.g., backbone) positions
        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = self.get_key_atom_pos(amino_types, atom_names, atom_amino_id, atom_pos)

        # calculate side chain torsion angles, up to four
        # do encoding
        side_chain_angle_encoding = self.get_side_chain_angle_encoding(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
        side_chain_angle_encoding[torch.isnan(side_chain_angle_encoding)] = 0

        # three backbone torsion angles
        backbone_angle_encoding = self.get_backbone_angle_encoding(torch.cat((torch.unsqueeze(pos_n,1), torch.unsqueeze(pos_ca,1), torch.unsqueeze(pos_c,1)),1))
        backbone_angle_encoding[torch.isnan(backbone_angle_encoding)] = 0

        data.seq = torch.LongTensor(amino_types)
        data.side_chain_angle_encoding = side_chain_angle_encoding
        data.backbone_angle_encoding = backbone_angle_encoding
        data.coords_ca = pos_ca
        data.coords_n = pos_n
        data.coords_c = pos_c
        data.x = atom_names
        data.atom_pos = torch.tensor(atom_pos)
        data.num_nodes = len(pos_ca)

        h5File.close()

        return data
    
    def process(self):  
        print('Beginning Processing ...')

        # Load the file with the list of functions.
        functions_ = []
        with open(self.root+"/unique_functions.txt", 'r') as mFile:
            for line in mFile:
                functions_.append(line.rstrip())
        
        # Get the file list.
        if self.split == "Train":
            splitFile = "/training.txt"
        elif self.split == "Val":
            splitFile = "/validation.txt"
        elif self.split == "Test":
            splitFile = "/testing.txt"

        proteinNames_ = []
        fileList_ = []
        with open(self.root+splitFile, 'r') as mFile:
            for line in mFile:
                proteinNames_.append(line.rstrip())
                fileList_.append(self.root+"/data/"+line.rstrip())
        print("current split {}, data size {}".format(self.split, len(fileList_)))
        
        # Load the functions.
        print("Reading protein functions")
        protFunct_ = {}
        with open(self.root+"/chain_functions.txt", 'r') as mFile:
            for line in mFile:
                splitLine = line.rstrip().split(',')
                if splitLine[0] in proteinNames_: 
                    protFunct_[splitLine[0]] = int(splitLine[1])

        # Load the dataset
        print("Reading the data")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_list = []
            for fileIter, curFile in tqdm(enumerate(fileList_)):
                print(curFile)
                fileName = curFile.split('/')[-1]
                protein = self.extract_protein_data(curFile) 
                protein.id = fileName           
                protein.y = torch.tensor(protFunct_[proteinNames_[fileIter]])
                if not protein.seq is None:
                    data_list.append(protein)     
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('Done!')
