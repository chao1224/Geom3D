#Credit to  https://github.com/hehefan/Continuous-Discrete-Convolution/blob/main/datasets.py

import numpy as np
from sklearn.preprocessing import normalize
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import os

def orientation(pos):
    u = normalize(X=pos[1:,:] - pos[:-1,:], norm='l2', axis=1)
    u1 = u[1:,:]
    u2 = u[:-1, :]
    b = normalize(X=u2 - u1, norm='l2', axis=1)
    n = normalize(X=np.cross(u2, u1), norm='l2', axis=1)
    o = normalize(X=np.cross(b, n), norm='l2', axis=1)
    ori = np.stack([b, n, o], axis=1)
    return np.concatenate([np.expand_dims(ori[0], 0), ori, np.expand_dims(ori[-1], 0)], axis=0)

def fmax(probs, labels):
    thresholds = np.arange(0, 1, 0.01)
    f_max = 0.0

    for threshold in thresholds:
        precision = 0.0
        recall = 0.0
        precision_cnt = 0
        recall_cnt = 0
        for idx in range(probs.shape[0]):
            prob = probs[idx]
            label = labels[idx]
            pred = (prob > threshold).astype(np.int32)
            correct_sum = np.sum(label*pred)
            pred_sum = np.sum(pred)
            label_sum = np.sum(label)
            if pred_sum > 0:
                precision += correct_sum/pred_sum
                precision_cnt += 1
            if label_sum > 0:
                recall += correct_sum/label_sum
            recall_cnt += 1
        if recall_cnt > 0:
            recall = recall / recall_cnt
        else:
            recall = 0
        if precision_cnt > 0:
            precision = precision / precision_cnt
        else:
            precision = 0
        f = (2.*precision*recall)/max(precision+recall, 1e-8)
        f_max = max(f, f_max)

    return f_max


# AA Letter to id
aa = "ACDEFGHIKLMNPQRSTVWYX"
aa_to_id = {}
for i in range(0, 21):
    aa_to_id[aa[i]] = i

class DatasetFOLD_CDConv(Dataset):

    def __init__(self, root='/content/drive/MyDrive/proteinDT/fold', random_seed=0, split='training'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        npy_dir = os.path.join(root, 'coordinates', split)
        fasta_file = os.path.join(root, split+'.fasta')

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))

        fold_classes = {}
        with open(os.path.join(root, 'class_map.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                fold_classes[arr[0]] = int(arr[1])

        protein_folds = {}
        with open(os.path.join(root, split+'.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                protein_folds[arr[0]] = fold_classes[arr[-1]]

        self.data = []
        self.labels = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))

            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            ori = orientation(pos)

            self.data.append((pos, ori, amino_ids.astype(int)))

            self.labels.append(protein_folds[protein_name])

        self.num_classes = max(self.labels) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pos, ori, amino = self.data[idx]
        label = self.labels[idx]

        if self.split == "training":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos))    # [num_nodes, num_dimensions]

        return data
