
from collections import defaultdict

import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Union

import torch
from torch_geometric.data import Data
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector


def create_2D_mol_from_3D_mol(mol):
    rdkit_mol = Chem.Mol()
    editable_mol = Chem.RWMol(rdkit_mol)
            
    for atom in mol.GetAtoms():
        editable_mol.AddAtom(atom)
        
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        editable_mol.AddBond(i, j, bond_type)
    mol = editable_mol.GetMol()
    
    mol.UpdatePropertyCache()
    Chem.GetSymmSSSR(mol)
    return mol


def extract_MMFF_energy_pos(rdkit_mol, num_conformers=5):
    try:
        mol = rdkit_mol
        result_list = AllChem.EmbedMultipleConfs(mol, num_conformers)
        result_list = AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s', numThreads=8)
        energy_list = [x[1] for x in result_list]
        index = np.argmin(energy_list)
        energy = energy_list[index]
        conformer = mol.GetConformer(id=int(index))
    except Exception as e:
        print(str(e))
        print("======bad")
        mol = rdkit_mol
        AllChem.Compute2DCoords(mol)
        energy = 0
        conformer = mol.GetConformer()

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        atom_feature = atomic_number-1
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    N = len(mol.GetAtoms())

    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    MMFF_data = {"x": x, "energy": energy, "positions": positions}
    return MMFF_data


# note this is different from the 2D case
# For 2D, please refer to https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py
atom_type_count = 119


def mol_to_graph_data_obj_simple_2D(mol):
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atomic_number = atom.GetAtomicNum()
        assert atomic_number-1 == atom_feature[0]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_feats_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            edges_list.append((i, j))
            edge_feats_list.append(edge_feature)
            edges_list.append((j, i))
            edge_feats_list.append(edge_feature)

        # Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        # Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_feats_list), dtype=torch.long)

    else:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    return data


def mol_to_graph_data_obj_simple_3D(mol, pure_atomic_num=False):
    # atoms
    atom_features_list = []
    atom_count = defaultdict(int)
    for atom in mol.GetAtoms():
        if not pure_atomic_num:
            atom_feature = atom_to_feature_vector(atom)
            atomic_number = atom.GetAtomicNum()
            assert atomic_number-1 == atom_feature[0]
            atom_count[atomic_number] += 1
        else:
            atomic_number = atom.GetAtomicNum()
            atom_feature = atomic_number - 1
            atom_count[atomic_number] += 1
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    conformer = mol.GetConformers()[0]
    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_feats_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            edges_list.append((i, j))
            edge_feats_list.append(edge_feature)
            edges_list.append((j, i))
            edge_feats_list.append(edge_feature)

        # Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        # Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_feats_list), dtype=torch.long)

    else:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(
        x=x,
        positions=positions,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    return data, atom_count


def mol_to_graph_data_obj_MMFF_3D(rdkit_mol, num_conformers):
    try:
        N = len(rdkit_mol.GetAtoms())
        if N > 100: # for sider
            raise Exception
        rdkit_mol = Chem.AddHs(rdkit_mol)
        mol = rdkit_mol
        result_list = AllChem.EmbedMultipleConfs(mol, num_conformers)
        result_list = AllChem.MMFFOptimizeMoleculeConfs(mol)
        mol = Chem.RemoveHs(mol)
        energy_list = [x[1] for x in result_list]
        index = np.argmin(energy_list)
        energy = energy_list[index]
        conformer = mol.GetConformer(id=int(index))
    except:
        print("======bad")
        mol = rdkit_mol
        AllChem.Compute2DCoords(mol)
        energy = 0
        conformer = mol.GetConformer()

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
        atomic_number = atom.GetAtomicNum()
        assert atomic_number-1 == atom_feature[0]
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    N = len(mol.GetAtoms())

    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    data = Data(
        x=x,
        positions=positions,
    )
    return data


# Credits to https://github.com/chao1224/GraphMVP/blob/main/src_regression/datasets_complete_feature/molecule_datasets.py#L62
def graph_data_obj_to_nx_simple(data):
    """ torch geometric -> networkx
    NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: networkx object """
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        temp_feature = atom_features[i]
        G.add_node(
            i,
            x0=temp_feature[0],
            x1=temp_feature[1],
            x2=temp_feature[2],
            x3=temp_feature[3],
            x4=temp_feature[4],
            x5=temp_feature[5],
            x6=temp_feature[6],
            x7=temp_feature[7],
            x8=temp_feature[8])
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        temp_feature= edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx,
                       e0=temp_feature[0],
                       e1=temp_feature[1],
                       e2=temp_feature[2])

    return G


# Credits to https://github.com/chao1224/GraphMVP/blob/main/src_regression/datasets_complete_feature/molecule_datasets.py#L62
def nx_to_graph_data_obj_simple(G):
    """ vice versa of graph_data_obj_to_nx_simple()
    Assume node indices are numbered from 0 to num_nodes - 1.
    NB: Uses simplified atom and bond features, and represent as indices.
    NB: possible issues with recapitulating relative stereochemistry
        since the edges in the nx object are unordered. """

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for _, node in G.nodes(data=True):
        atom_feature = [node['x0'], node['x1'], node['x2'], node['x3'], node['x4'], node['x5'], node['x6'], node['x7'], node['x8']]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 3  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge['e0'], edge['e1'], edge['e2']]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def get_graphs_within_cutoff(structure, cutoff=5.0, numerical_tol=1e-8, max_neighbours=None):
    from pymatgen.optimization.neighbors import find_points_in_spheres

    lattice_matrix = np.ascontiguousarray(np.array(structure.lattice.matrix), dtype=float)
    pbc = np.array([1, 1, 1], dtype=int)
    
    r = float(cutoff)
    cart_coords = np.ascontiguousarray(np.array(structure.cart_coords), dtype=float)
    center_indices, neighbor_indices, images, distance = find_points_in_spheres(
        cart_coords, cart_coords, r=r, pbc=pbc, lattice=lattice_matrix, tol=numerical_tol
    )
    center_indices = center_indices
    neighbor_indices = neighbor_indices
    images = images
    distance = distance
    exclude_self = (center_indices != neighbor_indices) | (distance > numerical_tol)
    center_indices, neighbor_indices, images, distance = center_indices[exclude_self], neighbor_indices[exclude_self], images[exclude_self], distance[exclude_self]
    indices = []
    for center_, neighbor_ in zip(center_indices, neighbor_indices):
        indices.append([center_, neighbor_])

    if max_neighbours is None:
        return indices, images, distance

    # do distance sorting for each of the center index
    center_lattice2distance = defaultdict(list)
    for indice, image, dist in zip(indices, images, distance):
        first_index, second_index = indice
        center_lattice2distance[first_index].append(dist)
    first_index_list = center_lattice2distance.keys()
    sorted_center_lattice2distance_threshold = {}
    for first_index in first_index_list:
        sorted_dist_list = sorted(center_lattice2distance[first_index])
        if len(sorted_dist_list) <= max_neighbours:
            sorted_center_lattice2distance_threshold[first_index] = sorted_dist_list[-1]
        else:
            sorted_center_lattice2distance_threshold[first_index] = sorted_dist_list[max_neighbours]

    # filter w.r.t. sorted_center_lattice2distance_threshold
    neo_indices, neo_images, neo_distance = [], [], []
    for indice, image, dis in zip(indices, images, distance):
        first_index = indice[0]
        if dis <= sorted_center_lattice2distance_threshold[first_index] + numerical_tol:
            neo_indices.append(indice)
            neo_images.append(image)
            neo_distance.append(dis)
    return neo_indices, neo_images, neo_distance


def range_neighbour_lattice(
    coordinates: np.ndarray, lattice: np.ndarray,
    max_distance: Union[float, None] = 4.0,
    max_neighbours: Union[int, None] = None,
    self_loops: bool = False,
    exclusive: bool = True,
    limit_only_max_neighbours: bool = False,
    numerical_tol: float = 1e-8,
    manual_super_cell_radius: float = None,
    super_cell_tol_factor: float = 0.25,
) -> list:
    r"""Generate range connections for a primitive unit cell in a periodic lattice (vectorized).
    The function generates a supercell of required radius and computes connections of neighbouring nodes
    from the primitive centered unit cell. For :obj:`max_neighbours` the supercell radius is estimated based on
    the unit cell density. Always the smallest necessary supercell is generated based on :obj:`max_distance` and
    :obj:`max_neighbours`. If a supercell for radius :obj:`max_distance` should always be generated but limited by
    :obj:`max_neighbours`, you can set :obj:`limit_only_max_neighbours` to `True`.
    .. warning::
        All atoms should be projected back into the primitive unit cell before calculating the range connections.
    .. note::
        For periodic structure, setting :obj:`max_distance` and :obj:`max_neighbours` to `inf` would also lead
        to an infinite number of neighbours and connections. If :obj:`exclusive` is set to `False`, having either
        :obj:`max_distance` or :obj:`max_neighbours` set to `inf`, will result in an infinite number of neighbours.
        If set to `None`, :obj:`max_distance` or :obj:`max_neighbours` can selectively be ignored.
    Args:
        coordinates (np.ndarray): Coordinate of nodes in the central primitive unit cell.
        lattice (np.ndarray): Lattice matrix of real space lattice vectors of shape `(3, 3)`.
            The lattice vectors must be given in rows of the matrix!
        max_distance (float, optional): Maximum distance to allow connections, can also be None. Defaults to 4.0.
        max_neighbours (int, optional): Maximum number of allowed neighbours for each central atom. Default is None.
        self_loops (bool, optional): Allow self-loops between the same central node. Defaults to False.
        exclusive (bool): Whether both distance and maximum neighbours must be fulfilled. Default is True.
        limit_only_max_neighbours (bool): Whether to only use :obj:`max_neighbours` to limit the number of neighbours
            but not use it to calculate super-cell. Requires :obj:`max_distance` to be not `None`.
            Can be used if the super-cell should be generated with certain :obj:`max_distance`. Default is False.
        numerical_tol  (float): Numerical tolerance for distance cut-off. Default is 1e-8.
        manual_super_cell_radius (float): Manual radius for supercell. This is otherwise automatically set by either
            :obj:`max_distance` or :obj:`max_neighbours` or both. For manual supercell only. Default is None.
        super_cell_tol_factor (float): Tolerance factor for supercell relative to unit cell size. Default is 0.25.
    Returns:
        list: [indices, images, dist]
    credit to https://github.com/aimat-lab/gcnn_keras/blob/1c056f9a3b2990a1adb176c2dcc58c86d2ff64cf/kgcnn/graph/methods/_geom.py#L172
    """
    # Require either max_distance or max_neighbours to be specified.
    if max_distance is None and max_neighbours is None:
        raise ValueError("Need to specify either `max_distance` or `max_neighbours` or both.")

    # Here we set the lattice matrix, with lattice vectors in either columns or rows of the matrix.
    lattice_col = np.transpose(lattice)
    lattice_row = lattice

    # Index list for nodes. Enumerating the nodes in the central unit cell.
    node_index = np.expand_dims(np.arange(0, len(coordinates)), axis=1)  # Nx1

    # Diagonals, center, volume and density of unit cell based on lattice matrix.
    center_unit_cell = np.sum(lattice_row, axis=0, keepdims=True) / 2  # (1, 3)
    max_radius_cell = np.amax(np.sqrt(np.sum(np.square(lattice_row - center_unit_cell), axis=-1)))
    max_diameter_cell = 2*max_radius_cell
    volume_unit_cell = np.sum(np.abs(np.cross(lattice[0], lattice[1]) * lattice[2]))
    density_unit_cell = len(node_index) / volume_unit_cell

    # Center cell distance. Compute the distance matrix separately for the central primitive unit cell.
    # Here one can check if self-loops (meaning loops between the nodes of the central cell) should be allowed.
    center_indices = np.indices((len(node_index), len(node_index)))
    center_indices = center_indices.transpose(np.append(np.arange(1, 3), 0))  # NxNx2
    center_dist = np.expand_dims(coordinates, axis=0) - np.expand_dims(coordinates, axis=1)  # NxNx3
    center_image = np.zeros(center_dist.shape, dtype="int")
    if not self_loops:
        def remove_self_loops(x):
            m = np.logical_not(np.eye(len(x), dtype="bool"))
            x_shape = np.array(x.shape)
            x_shape[1] -= 1
            return np.reshape(x[m], x_shape)
        center_indices = remove_self_loops(center_indices)
        center_image = remove_self_loops(center_image)
        center_dist = remove_self_loops(center_dist)

    # Check the maximum atomic distance, since in practice atoms may not be inside the unit cell. Although they SHOULD
    # be projected back into the cell.
    max_diameter_atom_pair = np.amax(center_dist) if len(coordinates) > 1 else 0.0
    max_distance_atom_origin = np.amax(np.sqrt(np.sum(np.square(coordinates), axis=-1)))

    # Mesh Grid list. For a list of indices bounding left and right make a list of a 3D mesh.
    # Function is used to pad image unit cells or their origin for super-cell.
    def mesh_grid_list(bound_left: np.array, bound_right: np.array) -> np.array:
        pos = [np.arange(i, j+1, 1) for i, j in zip(bound_left, bound_right)]
        grid_list = np.array(np.meshgrid(*pos)).T.reshape(-1, 3)
        return grid_list

    # Estimated real-space radius for max_neighbours based on density and volume of a single unit cell.
    if max_neighbours is not None:
        estimated_nn_volume = (max_neighbours + len(node_index)) / density_unit_cell
        estimated_nn_radius = abs(float(np.cbrt(estimated_nn_volume / np.pi * 3 / 4)))
    else:
        estimated_nn_radius = None

    # Determine the required size of super-cell
    if manual_super_cell_radius is not None:
        super_cell_radius = abs(manual_super_cell_radius)
    elif max_distance is None:
        super_cell_radius = estimated_nn_radius
    elif max_neighbours is None or limit_only_max_neighbours:
        super_cell_radius = max_distance
    else:
        if exclusive:
            super_cell_radius = min(max_distance, estimated_nn_radius)
        else:
            super_cell_radius = max(max_distance, estimated_nn_radius)

    # Safety for super-cell radius. We add this distance to ensure that all atoms of the outer images are within the
    # actual cutoff distance requested.
    super_cell_tolerance = max(max_diameter_cell, max_diameter_atom_pair, max_distance_atom_origin)
    super_cell_tolerance *= (1.0 + super_cell_tol_factor)

    # Bounding box of real space cube with edge length 2 or inner sphere of radius 1 transformed into index
    # space gives 'bounding_box_unit'. Simply upscale for radius of super-cell.
    # To account for node pairing within the unit cell we add 'max_diameter_cell'.
    bounding_box_unit = np.sum(np.abs(np.linalg.inv(lattice_col)), axis=1)
    bounding_box_index = bounding_box_unit * (super_cell_radius + super_cell_tolerance)
    bounding_box_index = np.ceil(bounding_box_index).astype("int")

    # Making grid for super-cell that repeats the unit cell for required indices in 'bounding_box_index'.
    # Remove [0, 0, 0] of center unit cell by hand.
    bounding_grid = mesh_grid_list(-bounding_box_index, bounding_box_index)
    bounding_grid = bounding_grid[
        np.logical_not(np.all(bounding_grid == np.array([[0, 0, 0]]), axis=-1))]  # Remove center cell
    bounding_grid_real = np.dot(bounding_grid, lattice_row)

    # Check which centers are in the sphere of cutoff, since for non-rectangular lattice vectors, the parallelepiped
    # can be overshooting the required sphere. Better do this here, before computing coordinates of nodes.
    dist_centers = np.sqrt(np.sum(np.square(bounding_grid_real), axis=-1))
    mask_centers = dist_centers <= (super_cell_radius + super_cell_tolerance + abs(numerical_tol))
    images = bounding_grid[mask_centers]
    shifts = bounding_grid_real[mask_centers]

    # Compute node coordinates of images and prepare indices for those nodes. For 'N' nodes per cell and 'C' images
    # (without the central unit cell), this will be (flatten) arrays of (N*C)x3.
    num_images = images.shape[0]
    images = np.expand_dims(images, axis=0)  # 1xCx3
    images = np.repeat(images, len(coordinates), axis=0)  # NxCx3
    coord_images = np.expand_dims(coordinates, axis=1) + shifts  # NxCx3
    coord_images = np.reshape(coord_images, (-1, 3))  # (N*C)x3
    images = np.reshape(images, (-1, 3))  # (N*C)x3
    indices = np.expand_dims(np.repeat(node_index, num_images), axis=-1)  # (N*C)x1

    # Make distance matrix of central cell to all image. This will have shape Nx(NxC).
    dist = np.expand_dims(coord_images, axis=0) - np.expand_dims(coordinates, axis=1)  # Nx(N*C)x3
    dist_indices = np.concatenate(
        [np.repeat(np.expand_dims(node_index, axis=1), len(indices), axis=1),
         np.repeat(np.expand_dims(indices, axis=0), len(node_index), axis=0)], axis=-1)  # Nx(N*C)x2
    dist_images = np.repeat(np.expand_dims(images, axis=0), len(node_index), axis=0)  # Nx(N*C)x3

    # Adding distance matrix of nodes within the central cell to the image distance matrix.
    # The resulting shape is then Nx(NxC+1).
    dist_indices = np.concatenate([center_indices, dist_indices], axis=1)  # Nx(N*C+1)x2
    dist_images = np.concatenate([center_image, dist_images], axis=1)  # Nx(N*C+1)x2
    dist = np.concatenate([center_dist, dist], axis=1)  # Nx(N*C+1)x3

    # Distance in real space.
    dist = np.sqrt(np.sum(np.square(dist), axis=-1))  # Nx(N*C+1)

    # Sorting the distance matrix. Indices and image information must be sorted accordingly.
    arg_sort = np.argsort(dist, axis=-1)
    dist_sort = np.take_along_axis(dist, arg_sort, axis=1)
    dist_indices_sort = np.take_along_axis(
        dist_indices, np.repeat(np.expand_dims(arg_sort, axis=2), dist_indices.shape[2], axis=2), axis=1)
    dist_images_sort = np.take_along_axis(
        dist_images, np.repeat(np.expand_dims(arg_sort, axis=2), dist_images.shape[2], axis=2), axis=1)

    # Select range connections based on distance cutoff and nearest neighbour limit. Uses masking.
    # Based on 'max_distance'.
    if max_distance is None:
        mask_distance = np.ones_like(dist_sort, dtype="bool")
    else:
        mask_distance = dist_sort <= max_distance + abs(numerical_tol)
    # Based on 'max_neighbours'.
    mask_neighbours = np.zeros_like(dist_sort, dtype="bool")
    if max_neighbours is None:
        max_neighbours = dist_sort.shape[-1]
    mask_neighbours[:, :max_neighbours] = True

    if exclusive:
        mask = np.logical_and(mask_neighbours, mask_distance)
    else:
        mask = np.logical_or(mask_neighbours, mask_distance)

    # Select nodes.
    out_dist = dist_sort[mask]
    out_images = dist_images_sort[mask]
    out_indices = dist_indices_sort[mask]

    return [out_indices, out_images, out_dist]

    
def preiodic_augmentation_with_lattice(
    atom_num_list, positions_list, lattice,
    center_and_image_edge_index_list, image_shift_list, range_distance_list):
    augmentation_record = defaultdict(list)
    key_index_dict = defaultdict(int)
    range_num = len(center_and_image_edge_index_list)

    key_total_index = 0
    neo_edge_index_list, neo_edge_distance_list = [], []
    periodic_index_mapping_list = []

    for first_indice, (atom_num, positions) in enumerate(zip(atom_num_list, positions_list)):
        original_image = [0, 0, 0]
        first_key = 'id: {}, image: [{}, {}, {}]'.format(first_indice, original_image[0], original_image[1], original_image[2])
        augmentation_record[first_key] = [atom_num_list[first_indice], positions_list[first_indice]]
        if first_key not in key_index_dict:
            key_index_dict[first_key] = key_total_index
            periodic_index_mapping_list.append(key_total_index)
            key_total_index += 1

    for range_idx in range(range_num):
        range_indice_two_nodes = center_and_image_edge_index_list[range_idx]
        range_image = list(image_shift_list[range_idx])
        range_distance = range_distance_list[range_idx]
        
        first_indice = range_indice_two_nodes[0]
        original_image = [0, 0, 0]
        first_key = 'id: {}, image: [{}, {}, {}]'.format(first_indice, original_image[0], original_image[1], original_image[2])
        if first_key not in augmentation_record:
            augmentation_record[first_key] = [atom_num_list[first_indice], positions_list[first_indice]]
        
        lattice_shift = np.array([0., 0., 0.])
        for direction_idx in range(3):
            if range_image[direction_idx] != 0:
                lattice_shift += lattice[direction_idx] * range_image[direction_idx]
        second_indice = range_indice_two_nodes[1]
        second_key = 'id: {}, image: [{}, {}, {}]'.format(second_indice, range_image[0], range_image[1], range_image[2])
        if second_key not in augmentation_record:
            augmentation_record[second_key] = [atom_num_list[second_indice], positions_list[second_indice] + lattice_shift]

        # Notice: first_key is already in
        if second_key not in key_index_dict:
            key_index_dict[second_key] = key_total_index
            periodic_index_mapping_list.append(second_indice)
            key_total_index += 1
        first_key_index = key_index_dict[first_key]
        second_key_index = key_index_dict[second_key]
        neo_edge_index_list.append([first_key_index, second_key_index])
        neo_edge_distance_list.append(range_distance)
        neo_edge_index_list.append([second_key_index, first_key_index])
        neo_edge_distance_list.append(range_distance)
        
        # Notice: only consider one direction to keep consistent with the input center_and_image_edge_index_list
        neo_edge_vector = positions_list[first_indice] - positions_list[second_indice] - lattice_shift
        neo_dist = np.linalg.norm(neo_edge_vector)
        assert np.abs(neo_dist - range_distance) < 1e-10
    
    neo_atom_num_list, neo_positions_list = [], []
    for key, value in augmentation_record.items():
        neo_atom_num_list.append(value[0])
        neo_positions_list.append(value[1])

    ##### only for debugging #####
    for range_idx in range(range_num):
        range_indice_two_nodes = center_and_image_edge_index_list[range_idx]
        range_image = list(image_shift_list[range_idx])
        range_distance = range_distance_list[range_idx]
        
        first_indice = range_indice_two_nodes[0]
        original_image = [0, 0, 0]
        first_key = 'id: {}, image: [{}, {}, {}]'.format(first_indice, original_image[0], original_image[1], original_image[2])
        
        lattice_shift = np.array([0., 0., 0.])
        for direction_idx in range(3):
            if range_image[direction_idx] != 0:
                lattice_shift += lattice[direction_idx] * range_image[direction_idx]
        second_indice = range_indice_two_nodes[1]
        second_key = 'id: {}, image: [{}, {}, {}]'.format(second_indice, range_image[0], range_image[1], range_image[2])

        first_key_index = key_index_dict[first_key]
        second_key_index = key_index_dict[second_key]
        first_position = neo_positions_list[first_key_index]
        second_position = neo_positions_list[second_key_index]

        assert np.linalg.norm(neo_positions_list[first_key_index] - positions_list[first_indice]) < 1e-10
        assert np.linalg.norm(neo_positions_list[second_key_index] - positions_list[second_indice] - lattice_shift) < 1e-10

    neo_edge_index_list = np.array(neo_edge_index_list)
    neo_edge_index_list = neo_edge_index_list.T
    
    return neo_atom_num_list, neo_positions_list, neo_edge_index_list, neo_edge_distance_list, periodic_index_mapping_list


def make_edges_into_two_direction(center_and_image_edge_index_list, range_distance_list):
    range_num = len(center_and_image_edge_index_list)

    neo_edge_index_list, neo_edge_distance_list = [], []
    for range_idx in range(range_num):
        first_indice, second_indice = center_and_image_edge_index_list[range_idx]
        neo_edge_index_list.append([first_indice, second_indice])
        neo_edge_index_list.append([second_indice, first_indice])

        distance = range_distance_list[range_idx]
        neo_edge_distance_list.append(distance)
        neo_edge_distance_list.append(distance)
    
    return neo_edge_index_list, neo_edge_distance_list
