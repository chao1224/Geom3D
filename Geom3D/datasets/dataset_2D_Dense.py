import os
import shutil
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


class MoleculeDataset2DDense(Dataset):
    def __init__(self, root, preprocessed_dataset, max_node_num):
        self.root = root
        self.preprocessed_dataset = preprocessed_dataset
        self.processed_dir = os.path.join(self.root, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        self.processed_file = os.path.join(self.processed_dir, "processed")
        print("processed_file", self.processed_file+".npz")
        self.max_node_num = max_node_num

        if not os.path.exists(self.processed_file+".npz"):
            self.process()

        data = np.load(self.processed_file+".npz")

        self.node_attr_matrix_list = data["node_attr_matrix_list"]
        self.adj_matrix_list = data["adj_matrix_list"]
        self.adj_attr_matrix_list = data["adj_attr_matrix_list"]
        self.node_num_list = data["node_num_list"]
        self.label_list = data["label_list"]

        return

    def __len__(self):
        return self.label_list.shape[0]

    def __getitem__(self, idx):
        node_attr_matrix = self.node_attr_matrix_list[idx]
        adj_matrix = self.adj_matrix_list[idx]
        adj_attr_matrix = self.adj_attr_matrix_list[idx]
        node_num = self.node_num_list[idx]
        label = self.label_list[idx]
        
        node_attr_matrix = torch.LongTensor(node_attr_matrix)
        adj_matrix = torch.LongTensor(adj_matrix)
        adj_attr_matrix = torch.LongTensor(adj_attr_matrix)
        node_num = torch.tensor([node_num])
        label = torch.FloatTensor(label)
        return node_attr_matrix, adj_matrix, adj_attr_matrix, node_num, label
        
    def process(self):
        print("Preprocessing with N-Gram Path ...")

        # TODO: will add in the future
        # preprocessed_smiles_path = os.path.join(self.preprocessed_dataset.processed_dir, "smiles.csv")
        # smiles_path = os.path.join(self.processed_dir, "smiles.csv")
        # shutil.copyfile(preprocessed_smiles_path, smiles_path)

        N = len(self.preprocessed_dataset)
        MAX_NODE = 0
        node_num_list = []
        label_list = []
        node_attr_matrix_list = []
        adj_matrix_list, adj_attr_matrix_list = [], []
        for idx in tqdm(range(N)):
            data = self.preprocessed_dataset[idx]
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr

            node_num = data.x.shape[0]
            node_num_list.append(node_num)
            MAX_NODE = max(MAX_NODE, node_num)
            if node_num > self.max_node_num:
                print("Invalid! The node num ({}) is larger than limit ({}).".format(node_num, self.max_node_num))
                continue

            label_list.append(data.y.numpy())

            node_attr_dim = data.x.shape[1]
            node_attr_matrix = np.zeros((self.max_node_num, node_attr_dim))
            for i, attr in enumerate(x):
                node_attr_matrix[i] = attr
            node_attr_matrix_list.append(node_attr_matrix)

            adj_matrix = np.zeros((self.max_node_num, self.max_node_num))
            adj_attr_dim = edge_attr.shape[1]
            adj_attr_matrix = np.zeros((self.max_node_num, self.max_node_num, adj_attr_dim))
            row_i, row_j = edge_index[0], edge_index[1]
            assert len(row_i) == len(row_j) == len(edge_attr)
            for i, j, attr in zip(row_i, row_j, edge_attr):
                adj_matrix[i][j] = 1
                adj_attr_matrix[i][j] = attr
            adj_matrix_list.append(adj_matrix)
            adj_attr_matrix_list.append(adj_attr_matrix)

        print("MAX_NODE: {}".format(MAX_NODE))

        node_num_list = np.asarray(node_num_list)
        label_list = np.array(label_list)
        node_attr_matrix_list = np.asarray(node_attr_matrix_list)
        adj_matrix_list = np.asarray(adj_matrix_list)
        adj_attr_matrix_list = np.asarray(adj_attr_matrix_list)

        print("node_num_list", node_num_list.shape)
        print("label_list", label_list.shape)
        print("node_attr_matrix_list", node_attr_matrix_list.shape)
        print("adj_matrix_list", adj_matrix_list.shape)
        print("adj_attr_matrix_list", adj_attr_matrix_list.shape)

        np.savez_compressed(
            self.processed_file,
            node_attr_matrix_list=node_attr_matrix_list,
            adj_matrix_list=adj_matrix_list,
            adj_attr_matrix_list=adj_attr_matrix_list,
            node_num_list=node_num_list,
            label_list=label_list,
        )
        return


if __name__ == "__main__":
    from Geom3D.datasets import MoleculeDatasetQM92D
    
    data_root = "../../data/molecule_datasets/qm9_2D"
    dataset = "qm9"
    task = "gap"
    print("start")
    preprocessed_dataset = MoleculeDatasetQM92D(
        data_root,
        dataset=dataset,
        task=task
    )
    print("done")
    task_id = preprocessed_dataset.task_id

    data_root = "../../data/molecule_datasets/qm9_2D_NGramPath"
    dataset = "qm9"
    task = "gap"
    dataset = MoleculeDataset2DDense(
        root=data_root,
        preprocessed_dataset=preprocessed_dataset,
        max_node_num=35
    )
