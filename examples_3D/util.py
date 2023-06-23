import logGINg
import random
import numpy as np
from scipy import stats
from math import sqrt

from rdkit.Chem import AllChem

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logGINg.getLogger()
logGINg.basicConfig(level=logGINg.INFO)
cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def do_CL(X, Y, args):
    if args.CL_similarity_metric == 'InfoNCE_dot_prod':
        criterion = nn.CrossEntropyLoss()
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, args.T)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        CL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    elif args.CL_similarity_metric == 'EBM_dot_prod':
        criterion = nn.BCEWithLogitsLoss()
        neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)]
                           for i in range(args.CL_neg_samples)], dim=0)
        neg_X = X.repeat((args.CL_neg_samples, 1))

        pred_pos = torch.sum(X * Y, dim=1) / args.T
        pred_neg = torch.sum(neg_X * neg_Y, dim=1) / args.T

        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = loss_pos + args.CL_neg_samples * loss_neg

        CL_acc = (torch.sum(pred_pos > 0).float() +
                  torch.sum(pred_neg < 0).float()) / \
                 (len(pred_pos) + len(pred_neg))
        CL_acc = CL_acc.detach().cpu().item()

    elif args.CL_similarity_metric == 'EBM_node_dot_prod':
        criterion = nn.BCEWithLogitsLoss()
        
        neg_index = torch.randperm(len(Y))
        neg_Y = torch.cat([Y[neg_index]])

        pred_pos = torch.sum(X * Y, dim=1) / args.T
        pred_neg = torch.sum(X * neg_Y, dim=1) / args.T

        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = loss_pos + loss_neg

        CL_acc = (torch.sum(pred_pos > 0).float() +
                  torch.sum(pred_neg < 0).float()) / \
                 (len(pred_pos) + len(pred_neg))
        CL_acc = CL_acc.detach().cpu().item()

    else:
        raise Exception

    return CL_loss, CL_acc


def dual_CL(X, Y, args):
    CL_loss_1, CL_acc_1 = do_CL(X, Y, args)
    CL_loss_2, CL_acc_2 = do_CL(Y, X, args)
    return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item())/len(pred)


def do_AttrMasking(batch, criterion, node_repr, molecule_atom_masking_model):
    target = batch.mask_node_label[:, 0]
    node_pred = molecule_atom_masking_model(node_repr[batch.masked_atom_indices])
    attributemask_loss = criterion(node_pred.double(), target)
    attributemask_acc = compute_accuracy(node_pred, target)
    return attributemask_loss, attributemask_acc


def do_ContextPred(batch, criterion, args, molecule_substruct_model,
                   molecule_context_model, molecule_readout_func):

    # creating substructure representation
    substruct_repr = molecule_substruct_model(
        batch.x_substruct, batch.edge_index_substruct,
        batch.edge_attr_substruct)[batch.center_substruct_idx]

    # creating context representations
    overlapped_node_repr = molecule_context_model(
        batch.x_context, batch.edge_index_context,
        batch.edge_attr_context)[batch.overlap_context_substruct_idx]

    # positive context representation
    # readout -> global_mean_pool by default
    context_repr = molecule_readout_func(overlapped_node_repr,
                                         batch.batch_overlapped_context)

    # negative contexts are obtained by shifting
    # the indices of context embeddings
    neg_context_repr = torch.cat(
        [context_repr[cycle_index(len(context_repr), i + 1)]
         for i in range(args.contextpred_neg_samples)], dim=0)

    num_neg = args.contextpred_neg_samples
    pred_pos = torch.sum(substruct_repr * context_repr, dim=1)
    pred_neg = torch.sum(substruct_repr.repeat((num_neg, 1)) * neg_context_repr, dim=1)

    loss_pos = criterion(pred_pos.double(),
                         torch.ones(len(pred_pos)).to(pred_pos.device).double())
    loss_neg = criterion(pred_neg.double(),
                         torch.zeros(len(pred_neg)).to(pred_neg.device).double())

    contextpred_loss = loss_pos + num_neg * loss_neg

    num_pred = len(pred_pos) + len(pred_neg)
    contextpred_acc = (torch.sum(pred_pos > 0).float() +
                       torch.sum(pred_neg < 0).float()) / num_pred
    contextpred_acc = contextpred_acc.detach().cpu().item()

    return contextpred_loss, contextpred_acc


def check_same_molecules(s1, s2):
    mol1 = AllChem.MolFromSmiles(s1)
    mol2 = AllChem.MolFromSmiles(s2)
    return AllChem.MolToInchi(mol1) == AllChem.MolToInchi(mol2)


def rmse(y, f):
    return sqrt(((y - f) ** 2).mean(axis=0))


def mse(y, f):
    return ((y - f) ** 2).mean(axis=0)


def pearson(y, f):
    return np.corrcoef(y, f)[0, 1]


def spearman(y, f):
    return stats.spearmanr(y, f)[0]


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    # ci = S / z
    return S / z


def get_num_task(dataset):
    """ used in molecule_finetune.py """
    if dataset == 'tox21':
        return 12
    elif dataset in ['hiv', 'bace', 'bbbp', 'donor']:
        return 1
    elif dataset == 'pcba':
        return 92
    elif dataset == 'muv':
        return 17
    elif dataset == 'toxcast':
        return 617
    elif dataset == 'sider':
        return 27
    elif dataset == 'clintox':
        return 2
    raise ValueError('Invalid dataset name.')