import scipy.spatial
import os
from pathlib import WindowsPath
import pandas as pd
import numpy as np
import math
import random
import subprocess
from Bio.PDB.PDBIO import Select

import atom3d.protein.sequence as seq
import atom3d.util.formats as fo
import atom3d.datasets.datasets as da


def get_pocket_res(protein, ligand, dist):
    """
    Given a co-crystallized protein and ligand, extract residues within specified distance of ligand.
    :param protein: Biopython object containing receptor protein
    :type protein: Bio.PDB.Structure
    :param ligand: RDKit molecule object containing co-crystallized ligand
    :type ligand: rdkit.Chem.rdchem.Mol
    :param dist: distance cutoff for defining binding pocket, in Angstrom
    :type dist: float
    :return key_residues: key binding site residues
    :rtype key_residues:  set containing Bio.PDB.Residue objects
    """
    # get protein coordinates
    prot_atoms = [a for a in protein.get_atoms()]
    prot_coords = [atom.get_coord() for atom in prot_atoms]

    # get ligand coordinates
    lig_coords = []
    for i in range(0, ligand.GetNumAtoms()):
        pos = ligand.GetConformer().GetAtomPosition(i)
        lig_coords.append([pos.x, pos.y, pos.z])

    kd_tree = scipy.spatial.KDTree(prot_coords)
    key_pts = kd_tree.query_ball_point(lig_coords, r=dist, p=2.0)
    key_pts = set([k for l in key_pts for k in l])

    key_residues = set()
    for i in key_pts:
        atom = prot_atoms[i]
        res = atom.get_parent()
        if res.get_resname() == 'HOH':
            continue
        key_residues.add(res)
    return key_residues


def get_pdb_code(path):
    """
    Extract 4-character PDB ID code from full path.
    :param path: Path to PDB file.
    :type path: str
    :return: PDB filename.
    :rtype: str
    """
    return path.split('/')[-1][:4].lower()
    

class PocketSelect(Select):
    """
    Selection class for subsetting protein to key binding residues. This is a subclass of :class:`Bio.PDB.PDBIO.Select`.
    """
    def __init__(self, reslist):
        self.reslist = reslist
    def accept_residue(self, residue):
        if residue in self.reslist:
            return True
        else:
            return False


def find_files(path, suffix, relative=None):
    """
    Find all files in path with given suffix. =
    :param path: Directory in which to find files.
    :type path: Union[str, Path]
    :param suffix: Suffix determining file type to search for.
    :type suffix: str
    :param relative: Flag to indicate whether to return absolute or relative path.
    :return: list of paths to all files with suffix sorted by their names.
    :rtype: list[Path]
    """
    if not relative:
        find_cmd = r"find {:} -regex '.*\.{:}' | sort".format(path, suffix)
    else:
        find_cmd = r"cd {:}; find . -regex '.*\.{:}' | cut -d '/' -f 2- | sort" \
            .format(path, suffix)
    out = subprocess.Popen(
        find_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=os.getcwd(), shell=True)
    (stdout, stderr) = out.communicate()
    name_list = stdout.decode().split()
    name_list.sort()
    return name_list


def filter(dataset, maxnumat):
    # By default, keep all frames
    indices = []
    # Find indices of molecules to keep
    for i,item in enumerate(dataset):
        # Concatenate all relevant atoms
        atom_frames = [item['atoms_pocket'],item['atoms_ligand']]
        atoms = pd.concat(atom_frames, ignore_index=True)
        # Count heavy atoms
        num_heavy_atoms = sum(atoms['element'] != 'H')
        # Does the structure contain undesired elements
        allowed = ['H','C','N','O','S','Zn','Cl','F','P','Mg']
        unwanted_elements = set(atoms['element']) - set(allowed)
        # add the index
        if num_heavy_atoms <= maxnumat and len(unwanted_elements) == 0:
            indices.append(i)
    indices = np.array(indices, dtype=int)
    return indices


def write_npz(dataset, filename, indices, drop):
    # Get the coordinates
    save_dict = da.extract_coordinates_as_numpy_arrays(dataset, indices, 
        atom_frames=['atoms_pocket','atoms_ligand'], drop_elements=['H'])
    # Add the label data 
    save_dict['neglog_aff'] = np.array([dataset[i]['scores']['neglog_aff'] for i in indices])
    # Save the data
    np.savez_compressed(filename, **save_dict)


def write_index_to_file(indices, file_path):
    f = open(file_path, 'w')
    for indice in indices:
        f.write(indice)
    return


def identity_split(
        all_chain_sequences, cutoff, val_split=0.1, test_split=0.1,
        min_fam_in_split=5, blast_db=None, random_seed=None):
    # all_chain_sequences = [seq.get_chain_sequences(x['atoms']) for x in dataset]
    # Flatten.
    flat_chain_sequences = [x for sublist in all_chain_sequences for x in sublist]

    # write all sequences to BLAST-formatted database
    if blast_db is None:
        seq.write_to_blast_db(flat_chain_sequences, 'blast_db')
        blast_db = 'blast_db'

    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(all_chain_sequences)
    test_size = n * test_split
    val_size = n * val_split

    to_use = set(range(len(all_chain_sequences)))
    print('generating validation set...')
    val_indices, to_use = _create_identity_split(
        all_chain_sequences, cutoff, to_use, val_size, min_fam_in_split, blast_db)
    print('generating test set...')
    test_indices, to_use = _create_identity_split(
        all_chain_sequences, cutoff, to_use, test_size, min_fam_in_split, blast_db)
    train_indices = to_use

    return train_indices, val_indices, test_indices


def _create_identity_split(all_chain_sequences, cutoff, to_use, split_size,
                           min_fam_in_split, blast_db):
    dataset_size = len(all_chain_sequences)
    chain_to_idx = {y[0]: i for (i, x) in enumerate(all_chain_sequences) for y in x}

    all_indices = set(range(dataset_size))
    split, used = set(), all_indices.difference(to_use)
    while len(split) < split_size:
        i = random.sample(to_use, 1)[0]

        # Get chains that match.
        found = seq.find_similar(all_chain_sequences[i], blast_db, cutoff, dataset_size)
        # Map back to source.
        found = set([chain_to_idx[x] for x in found])
        found = found.difference(used)

        # ensure that at least min_fam_in_split families in each split
        max_fam_size = int(math.ceil(split_size / min_fam_in_split))
        split = split.union(list(found)[:max_fam_size])
        to_use = to_use.difference(found)
        used = used.union(found)

    return split, to_use