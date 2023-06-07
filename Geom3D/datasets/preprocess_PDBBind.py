import os
import argparse
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import Bio

from torch.utils.data import Dataset

import atom3d.util.formats as ft
from PDBBind_utils import get_label, get_ligand, get_pocket_res, get_pdb_code, PocketSelect, find_files, \
Scores, SequenceReader, SmilesReader, \
PDBDataset, SDFDataset, LMDBDataset, \
filter, write_npz, UpdateTypes


from torch.utils.data import Dataset
import atom3d.datasets.datasets as da
import atom3d.util.file as fi


def process_files(input_dir):
    """
    Process all protein (pdb) and ligand (sdf) files in input directory.
    :param input dir: directory containing PDBBind data
    :type input_dir: str
    :return structure_dict: dictionary containing each structure, keyed by PDB code. Each PDB is a dict containing protein as Biopython object and ligand as RDKit Mol object
    :rtype structure_dict: dict
    """
    structure_dict = {}
    pdb_files = find_files(input_dir, 'pdb')

    for f in tqdm(pdb_files, desc='pdb files'):
    # for f in tqdm(pdb_files[:200], desc='pdb files'):
        f = str(f)
        pdb_id = get_pdb_code(f)
        if pdb_id not in structure_dict:
            structure_dict[pdb_id] = {}
        if '_protein' in f:
            prot = ft.read_any(f)
            structure_dict[pdb_id]['protein'] = prot

    lig_files = find_files(input_dir, 'sdf')
    for f in tqdm(lig_files, desc='ligand files'):
    # for f in tqdm(lig_files[:100], desc='ligand files'):
        f = str(f)
        pdb_id = get_pdb_code(f)
        structure_dict[pdb_id]['ligand'] = get_ligand(f)

    return structure_dict


def write_files(pdbid, protein, ligand, pocket, out_path):
    """
    Writes cleaned structure files for protein, ligand, and pocket.
    :param pdbid: PDB ID for protein
    :type pdbid: str
    :param protein: object containing proteins
    :type ligand: object containing ligand molecules
    :param pocket: residues contained in pocket, as output by get_pocket_res()
    :type pocket: set containing Bio.PDB.Residue objects
    :param out_path: directory to write files to
    :type out_path: str
    """
    # write protein to mmCIF file
    io = Bio.PDB.MMCIFIO()
    io.set_structure(protein)
    io.save(os.path.join(out_path, f"{pdbid}_protein.cif"))

    # write pocket to mmCIF file
    io.save(os.path.join(out_path, f"{pdbid}_pocket.cif"), PocketSelect(pocket))

    # write ligand to file
    writer = Chem.SDWriter(os.path.join(out_path, f"{pdbid}_ligand.sdf"))
    writer.write(ligand)
    return


def produce_cleaned_dataset(structure_dict, out_path, dist=6.0):
    """
    Generate and save cleaned dataset, given dictionary of structures processed by process_files.
    :param structure_dict: dictionary containing protein and ligand structures processed from input files. Should be a nested dict with PDB ID in first level and keys 'protein', 'ligand' in second level.
    :type structure_dict: dict
    :param out_path: path to output directory
    :type out_path: str
    :param dist: distance cutoff for defining binding pocket (in Angstrom), default is 6.0
    :type dist: float, optional
    """
    pockets = []
    for pdb, data in tqdm(structure_dict.items(), desc='writing to files'):
        protein = structure_dict[pdb]['protein']
        ligand = structure_dict[pdb]['ligand']
        # check for failed ligand (due to bad structure file)
        if ligand is None:
            continue
        pocket_res = get_pocket_res(protein, ligand, dist)
        write_files(pdb, protein, ligand, pocket_res, out_path)
    return


def generate_labels(index_dir, out_path):
    valid_pdbs = [get_pdb_code(str(f)) for f in find_files(out_path, 'sdf')]
    dat = []
    with open(os.path.join(index_dir, 'INDEX_refined_data.2020')) as f:
        for line in f:
            line = str(line)
            if line.startswith('#'):
                continue
            l = line.strip().split()
            if l[0] not in valid_pdbs:
                continue
            dat.append(l[:5]+l[6:])

    refined_set = pd.DataFrame(dat, columns=['pdb','res','year','neglog_aff','affinity','ref','ligand'])

    refined_set[['measurement', 'affinity']] = refined_set['affinity'].str.split('=',expand=True)

    refined_set['ligand'] = refined_set['ligand'].str.strip('()')
    
    refined_set.to_csv(os.path.join(out_path, 'pdbbind_refined_set_cleaned.csv'), index=False)

    labels = refined_set[['pdb', 'neglog_aff']].rename(columns={'neglog_aff': 'label'})

    labels.to_csv(os.path.join(out_path, 'pdbbind_refined_set_labels.csv'), index=False)
    return


class LBADataset(Dataset):
    def __init__(self, input_file_path, pdbcodes, transform=None):
        self._protein_dataset = None
        self._pocket_dataset = None
        self._ligand_dataset = None
        self._load_datasets(input_file_path, pdbcodes)

        self._num_examples = len(self._protein_dataset)
        self._transform = transform

    def _load_datasets(self, input_file_path, pdbcodes):
        protein_list = []
        pocket_list = []
        ligand_list = []
        # for pdbcode in tqdm(pdbcodes):
        for pdbcode in tqdm(pdbcodes[:100]):
            # protein_path = os.path.join(input_file_path, f'{pdbcode:}/{pdbcode:}_protein.cif')
            # pocket_path = os.path.join(input_file_path, f'{pdbcode:}/{pdbcode:}_pocket.cif')
            # ligand_path = os.path.join(input_file_path, f'{pdbcode:}/{pdbcode:}_ligand.sdf')
            protein_path = os.path.join(input_file_path, f'{pdbcode:}_protein.cif')
            pocket_path = os.path.join(input_file_path, f'{pdbcode:}_pocket.cif')
            ligand_path = os.path.join(input_file_path, f'{pdbcode:}_ligand.sdf')
            if os.path.exists(protein_path) and os.path.exists(pocket_path) and os.path.exists(ligand_path):
                protein_list.append(protein_path)
                pocket_list.append(pocket_path)
                ligand_list.append(ligand_path)
        assert len(protein_list) == len(pocket_list) == len(ligand_list)
        print(f'Found {len(protein_list):} protein/ligand files...')

        self._protein_dataset = PDBDataset(protein_list, transform=SequenceReader(input_file_path))
        self._pocket_dataset = PDBDataset(pocket_list, transform=None)
        self._ligand_dataset = SDFDataset(ligand_list, read_bonds=True, transform=SmilesReader())
        return

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        protein = self._protein_dataset[index]
        pocket = self._pocket_dataset[index]
        ligand = self._ligand_dataset[index]
        pdbcode = fi.get_pdb_code(protein['id'])

        item = {
            'atoms_protein': protein['atoms'],
            'atoms_pocket': pocket['atoms'],
            'atoms_ligand': ligand['atoms'],
            'bonds': ligand['bonds'],
            'id': pdbcode,
            'seq': protein['seq'],
            'smiles': ligand['smiles'],
        }
        if self._transform:
            item = self._transform(item)
        return item


def split_lmdb_dataset(lmdb_path, train_txt, val_txt, test_txt, split_dir):
    print(f'Splitting indices, load data from {lmdb_path:}...')
    lmdb_ds = LMDBDataset(lmdb_path, 'lmdb')

    def _write_split_indices(split_txt, lmdb_ds, output_txt):
        with open(split_txt, 'r') as f:
            split_set = set([x.strip() for x in f.readlines()])
        # Check if the pdbcode in id is in the desired pdbcode split set
        split_ids = list(filter(lambda id: id in split_set, lmdb_ds.ids()))
        # Convert ids into lmdb numerical indices and write into txt file
        split_indices = lmdb_ds.ids_to_indices(split_ids)
        with open(output_txt, 'w') as f:
            f.write(str('\n'.join([str(i) for i in split_indices])))
        return split_indices

    print(f'Write results to {split_dir:}...')
    os.makedirs(os.path.join(split_dir, 'indices'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'data'), exist_ok=True)

    indices_train = _write_split_indices(train_txt, lmdb_ds, os.path.join(split_dir, 'indices/train_indices.txt'))
    indices_val = _write_split_indices(val_txt, lmdb_ds, os.path.join(split_dir, 'indices/val_indices.txt'))
    indices_test = _write_split_indices(test_txt, lmdb_ds, os.path.join(split_dir, 'indices/test_indices.txt'))

    train_dataset, val_dataset, test_dataset = spl.split(lmdb_ds, indices_train, indices_val, indices_test)
    da.make_lmdb_dataset(train_dataset, os.path.join(split_dir, 'data/train'))
    da.make_lmdb_dataset(val_dataset, os.path.join(split_dir, 'data/val'))
    da.make_lmdb_dataset(test_dataset, os.path.join(split_dir, 'data/test'))
    return


if __name__ == "__main__":
    data_dir = 'refined_set_2020'
    index_dir = 'index'
    out_dir = 'temp'

    if not os.path.exists(data_dir):
        raise Exception('Path not found. Please enter valid path to PDBBind dataset.')

    if not os.path.exists(index_dir):
        raise Exception('Path not found. Please enter valid path to PDBBind index.')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # ########## step 0
    # dist = 6
    # structures = process_files(data_dir)
    # produce_cleaned_dataset(structures, out_dir, dist)
    # generate_labels(index_dir, out_dir)

    # ########## step 1
    # scores = Scores('temp/pdbbind_refined_set_cleaned.csv')
    # pdbcodes = os.listdir('refined_set_2020')
    # print('len of pdbcodes: {}'.format(len(pdbcodes)))

    # output_root = 'temp_out'
    # lmdb_path = os.path.join(output_root, 'data')
    # os.makedirs(lmdb_path, exist_ok=True)
    # print(f'Creating lmdb dataset into {lmdb_path:}...')

    input_file_path = 'temp_out'
    # dataset = LBADataset(input_file_path, pdbcodes, transform=scores)
    # print('dataset: {}'.format(dataset))
    # da.make_lmdb_dataset(dataset, lmdb_path)

    # output_root = 'temp_out_02'
    # split_lmdb_dataset('temp_out/data', 'temp_out_02/train.txt', 'temp_out_02/val.txt', 'temp_out_02/test.txt', output_root)

    # ########## step 2
    # input_root = 'temp_out'
    # output_file_path = 'temp_out'
    # # Function for input and output path
    # out_path = lambda f: os.path.join(output_file_path, f)
    # inp_path = lambda f: os.path.join(input_root, f)
    # # Define which elements to drop
    # droph = True
    # drop = []
    # if droph:
    #     drop.append('H')
    
    # maxnumat = 500
    # split = True
    # split = False

    # if split: # use the split datasets 
    #     print(f'Processing datasets from {input_root:}.')
    #     # Training set
    #     print(f'Processing training dataset...')
    #     dataset = LMDBDataset(inp_path('train'), transform=UpdateTypes(['atoms_pocket','atoms_ligand']))
    #     indices = filter(dataset, maxnumat)
    #     write_npz(dataset, out_path('train.npz'), indices, drop)
    #     # Validation set
    #     print(f'Processing validation dataset...')
    #     dataset = LMDBDataset(inp_path('val'), transform=UpdateTypes(['atoms_pocket','atoms_ligand']))
    #     indices = filter(dataset, maxnumat)
    #     write_npz(dataset, out_path('valid.npz'), indices, drop)
    #     # Test set
    #     print(f'Processing test dataset...')
    #     dataset = LMDBDataset(inp_path('test'), transform=UpdateTypes(['atoms_pocket','atoms_ligand']))
    #     indices = filter(dataset, maxnumat)
    #     write_npz(dataset, out_path('test.npz'), indices, drop)

    # else: # use the full data set
    #     print(f'Processing full dataset from {input_root:}...')
    #     dataset = LMDBDataset(inp_path('all/data'), transform=UpdateTypes(['atoms_pocket','atoms_ligand']))
    #     indices = filter(dataset, maxnumat)
    #     write_npz(dataset, out_path('all.npz'), indices, drop)
    

    ##### for dataloader
    input_root = 'temp_out'
    inp_path = lambda f: os.path.join(input_root, f)
    dataset = LMDBDataset(inp_path('all/data'), transform=UpdateTypes(['atoms_pocket','atoms_ligand']))
    print(len(dataset))

    for i in range(10):
        print(i)
        print(dataset[i].keys())
        print(dataset[i]['id'])
        # print(dataset[i]['atoms_protein'])
        # print(dataset[i]['atoms_pocket'])
        # print(dataset[i]['atoms_ligand'])
        print('\n')

