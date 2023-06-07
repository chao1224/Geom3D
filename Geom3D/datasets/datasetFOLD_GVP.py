# Credit to https://github.com/divelab/DIG/blob/dig-stable/dig/threedgraph/dataset/ECdataset.py
# Data processing credit to https://github.com/drorlab/gvp-pytorch/blob/main/gvp/atom3d.py
import os.path as osp
import numpy as np
import warnings
from tqdm import tqdm
import pandas as pd

import torch, random, scipy, math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset
import torch_cluster, torch_geometric, torch_scatter

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def _edge_features(coords, edge_index, D_max=4.5, num_rbf=16, device='cpu'):
    
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), 
               D_max=D_max, D_count=num_rbf, device=device)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num,
            (edge_s, edge_v))

    return edge_s, edge_v


class BaseTransform:
    '''
    Implementation of an ATOM3D Transform which featurizes the atomic
    coordinates in an ATOM3D dataframes into `torch_geometric.data.Data`
    graphs. This class should not be used directly; instead, use the
    task-specific transforms, which all extend BaseTransform. Node
    and edge features are as described in the EGNN manuscript.
    
    Returned graphs have the following attributes:
    -x          atomic coordinates, shape [n_nodes, 3]
    -atoms      numeric encoding of atomic identity, shape [n_nodes]
    -edge_index edge indices, shape [2, n_edges]
    -edge_s     edge scalar features, shape [n_edges, 16]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    
    Subclasses of BaseTransform will produce graphs with additional 
    attributes for the tasks-specific training labels, in addition 
    to the above.
    
    All subclasses of BaseTransform directly inherit the BaseTransform
    constructor.
    
    :param edge_cutoff: distance cutoff to use when drawing edges
    :param num_rbf: number of radial bases to encode the distance on each edge
    :device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, edge_cutoff=4.5, num_rbf=16, device='cpu'):
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.device = device
            
    def __call__(self, df):
        '''
        :param df: `pandas.DataFrame` of atomic coordinates
                    in the ATOM3D format
        
        :return: `torch_geometric.data.Data` structure graph
        '''
        _element_mapping = lambda x: {
        'H' : 0,
        'C' : 1,
        'N' : 2,
        'O' : 3,
        'F' : 4,
        'S' : 5,
        'Cl': 6, 'CL': 6,
        'P' : 7
        }.get(x, 8)

        with torch.no_grad():
            coords = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(),
                                     dtype=torch.float32, device=self.device)
            atoms = torch.as_tensor(list(map(_element_mapping, df.element)),
                                            dtype=torch.long, device=self.device)
            
            edge_index = torch_cluster.radius_graph(coords, r=self.edge_cutoff)

            edge_s, edge_v = _edge_features(coords, edge_index, 
                                D_max=self.edge_cutoff, num_rbf=self.num_rbf, device=self.device)

            return torch_geometric.data.Data(x=coords, atoms=atoms,
                        edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)


class DatasetFOLD_GVP(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, split='train'):
        self.split = split
        self.root = root

        super(DatasetFOLD_GVP, self).__init__(
            root, transform, pre_transform, pre_filter)
        
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, name, self.split)

    @property
    def raw_file_names(self):
        name = self.split + '.txt'
        return name

    @property
    def processed_file_names(self):
        return 'data.pt'

    def _normalize(self,tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def get_atom_pos(self, amino_types, atom_names, atom_amino_id, atom_pos):
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

    def side_chain_embs(self, pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        v1, v2, v3, v4, v5, v6, v7 = pos_ca - pos_n, pos_cb - pos_ca, pos_g - pos_cb, pos_d - pos_g, pos_e - pos_d, pos_z - pos_e, pos_h - pos_z

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        angle1 = torch.unsqueeze(self.compute_diherals(v1, v2, v3),1)
        angle2 = torch.unsqueeze(self.compute_diherals(v2, v3, v4),1)
        angle3 = torch.unsqueeze(self.compute_diherals(v3, v4, v5),1)
        angle4 = torch.unsqueeze(self.compute_diherals(v4, v5, v6),1)
        angle5 = torch.unsqueeze(self.compute_diherals(v5, v6, v7),1)

        side_chain_angles = torch.cat((angle1, angle2, angle3, angle4),1)
        side_chain_embs = torch.cat((torch.sin(side_chain_angles), torch.cos(side_chain_angles)),1)
        
        return side_chain_embs

    def bb_embs(self, X):   
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

        angle = self.compute_diherals(u0, u1, u2)
        
        # add phi[0], psi[-1], omega[-1] with value 0
        angle = F.pad(angle, [1, 2]) 
        angle = torch.reshape(angle, [-1, 3])
        angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
        return angle_features

    def compute_diherals(self, v1, v2, v3):
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))
        return torsion

    def protein_to_graph(self, pFilePath):
        import h5py
        h5File = h5py.File(pFilePath, "r")
        data = Data()
 
        amino_types = h5File['amino_types'][()] # size: (n_amino,)
        mask = amino_types == -1
        if np.sum(mask) > 0:
            amino_types[mask] = 25 # for amino acid types, set the value of -1 to 25
        atom_amino_id = h5File['atom_amino_id'][()] # size: (n_atom,)
        atom_names = h5File['atom_names'][()] # size: (n_atom,)
        atom_pos = h5File['atom_pos'][()][0] #size: (n_atom,3)

        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = self.get_atom_pos(amino_types, atom_names, atom_amino_id, atom_pos)
        
        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        side_chain_embs = self.side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
        side_chain_embs[torch.isnan(side_chain_embs)] = 0

        # three backbone torsion angles
        bb_embs = self.bb_embs(torch.cat((torch.unsqueeze(pos_n,1), torch.unsqueeze(pos_ca,1), torch.unsqueeze(pos_c,1)),1))
        bb_embs[torch.isnan(bb_embs)] = 0

        # backbone atoms' positions
        C_list_1 = ["C"] * pos_ca.shape[0]
        N_list = ["N"] * pos_n.shape[0]
        C_list_2 = ["C"] * pos_c.shape[0]
        element = C_list_1 + N_list + C_list_2

        backbone_coords = torch.cat((pos_ca, pos_n, pos_c))
        header = ["x", "y", "z"]
        backbone_df = pd.DataFrame(backbone_coords.numpy(), columns=header)
        backbone_df["element"] = element
        
        backbone = self.graph(backbone_df)
        data.x = backbone.atoms
        data.atoms = backbone.atoms
        data.edge_index = backbone.edge_index
        data.edge_s = backbone.edge_s
        data.edge_v = backbone.edge_v

        h5File.close()
        return data

    def process(self):  
        self.graph = BaseTransform(device="cuda")
        print('Beginning Processing ...')

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
                print(curFile)
                fileName = curFile.split('/')[-1]
                curProtein = self.protein_to_graph(curFile+".hdf5") 
                curProtein.id = fileName           
                curProtein.y = torch.tensor(cathegories_[fileIter])
                if not curProtein.x is None:
                    data_list.append(curProtein)     
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('Done!')


if __name__ == "__main__":
    for split in ['training', 'validation', 'test_fold', 'test_superfamily', 'test_family']:
        print('#### Now processing {} data ####'.format(split))
        dataset = DatasetFOLD_GVP(root='../../data/FOLD', split=split)
        print(dataset)