import argparse
import os
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from Geom3D.datasets.dataset_QM9_2D_3D_Transformer import MoleculeDatasetQM9_2Dand3DTransformer
from Geom3D.models import Graphormer, TransformerM
from Geom3D.dataloaders.transformer_padding.collator import collate_fn_2D, collate_fn_3D
from splitters import qm9_random_customized_01, qm9_random_customized_02


def move_to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device)
  elif isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = move_to(v, device)
    return res


def mean_absolute_error(pred, target):
    return np.mean(np.abs(pred - target))


def split(dataset, data_root):
    if args.split == "customized_01" and "qm9" in args.dataset:
        train_dataset, valid_dataset, test_dataset = qm9_random_customized_01(
            dataset, null_value=0, seed=args.seed
        )
        print("customized random (01) on QM9")
    elif args.split == "customized_02" and "qm9" in args.dataset:
        train_dataset, valid_dataset, test_dataset = qm9_random_customized_02(
            dataset, null_value=0, seed=args.seed
        )
        print("customized random (02) on QM9")
    else:
        raise ValueError("Invalid split option on {}.".format(args.dataset))
    print(len(train_dataset), "\t", len(valid_dataset), "\t", len(test_dataset))
    return train_dataset, valid_dataset, test_dataset


def save_model(save_best):
    if not args.output_model_dir == "":
        if save_best:
            print("save model with optimal loss")
            output_model_path = os.path.join(args.output_model_dir, "model.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            torch.save(saved_model_dict, output_model_path)

        else:
            print("save model in the last epoch")
            output_model_path = os.path.join(args.output_model_dir, "model_final.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            torch.save(saved_model_dict, output_model_path)
    return


def train(epoch, device, loader, optimizer):
    model.train()

    loss_acc = 0
    num_iters = len(loader)

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        batch = move_to(batch, device)

        if args.model_3d == "Graphormer":
            x_pred = model(batch)  # [B, num_node, 1]
            x_pred = x_pred.squeeze() # [B, num_node]
            pred = x_pred.mean(dim=1)

        elif args.model_3d == "TransformerM":
            x_pred = model(batch)[0]  # [B, num_node, 1]
            x_pred = x_pred.squeeze() # [B, num_node]
            pred = x_pred.mean(dim=1)

        B = pred.size()[0]
        y = batch["y"].view(B, -1)[:, task_id]
        # normalize
        y = (y - TRAIN_mean) / TRAIN_std

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_acc += loss.cpu().detach().item()

        if args.lr_scheduler in ["CosineAnnealingWarmRestarts"]:
            lr_scheduler.step(epoch - 1 + step / num_iters)

    loss_acc /= len(loader)
    if args.lr_scheduler in ["StepLR", "CosineAnnealingLR"]:
        lr_scheduler.step()
    elif args.lr_scheduler in [ "ReduceLROnPlateau"]:
        lr_scheduler.step(loss_acc)

    return loss_acc


@torch.no_grad()
def eval(device, loader):
    model.eval()
    y_true = []
    y_scores = []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for batch in L:
        batch = move_to(batch, device)

        if args.model_3d == "graphormer":
            x_pred = model(batch)  # [B, num_node, 1]
            x_pred = x_pred.squeeze() # [B, num_node]
            pred = x_pred.mean(dim=1)

        elif args.model_3d == "transformerM":
            x_pred = model(batch)[0]  # [B, num_node, 1]
            x_pred = x_pred.squeeze() # [B, num_node]
            pred = x_pred.mean(dim=1)

        B = pred.size()[0]
        y = batch["y"].view(B, -1)[:, task_id]
        # denormalize
        pred = pred * TRAIN_std + TRAIN_mean

        y_true.append(y)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    mae = mean_absolute_error(y_scores, y_true)
    return mae, y_true, y_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="qm9")
    parser.add_argument("--task", type=str, default="gap")
    parser.add_argument("--split", type=str, default="customized_01")
    parser.add_argument("--model_3d", type=str, default="graphormer", choices=["graphormer", "transformerM"])

    # base hypers
    parser.add_argument("--num_atoms", type=int, default=119)
    parser.add_argument("--num_in_degree", type=int, default=10)
    parser.add_argument("--num_out_degree", type=int, default=10)
    parser.add_argument("--num_edges", type=int, default=5)
    parser.add_argument("--num_spatial", type=int, default=512)
    parser.add_argument("--num_edge_dis", type=int, default=128)
    parser.add_argument("--edge_type", type=str, default="multi_hop")
    parser.add_argument("--multi_hop_max_dist", type=int, default=5)
    parser.add_argument("--encoder_embed_dim", type=int, default=300)
    parser.add_argument("--encoder_ffn_embed_dim", type=int, default=300)
    parser.add_argument("--encoder_attention_heads", type=int, default=4)
    parser.add_argument("--encoder_layers", type=int, default=5)
    parser.add_argument("--activation_fn", type=str, default='relu')
    parser.add_argument("--max_nodes", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--act_dropout", type=float, default=0.1)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--num_segment", type=int, default=2)
    parser.add_argument("--num_3d_bias_kernel", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=1)
    # share_input_output_embed or share_encoder_input_output_embed
    parser.add_argument("--share_input_output_embed", dest="share_input_output_embed", action="store_true")
    parser.add_argument("--no_share_input_output_embed", dest="share_input_output_embed", action="store_false")
    parser.set_defaults(share_input_output_embed=False)
    #
    parser.add_argument("--encoder_learned_pos", dest="encoder_learned_pos", action="store_true")
    parser.add_argument("--no_encoder_learned_pos", dest="encoder_learned_pos", action="store_false")
    parser.set_defaults(encoder_learned_pos=True)
    #
    parser.add_argument("--encoder_normalize_before", dest="encoder_normalize_before", action="store_true")
    parser.add_argument("--no_encoder_normalize_before", dest="encoder_normalize_before", action="store_false")
    parser.set_defaults(encoder_normalize_before=True)
    #
    parser.add_argument("--apply_init", dest="apply_init", action="store_true")
    parser.add_argument("--no_apply_init", dest="apply_init", action="store_false")
    parser.set_defaults(apply_init=True)

    # hypers for Graphormer
    parser.add_argument("--pre_layernorm", dest="pre_layernorm", action="store_true")
    parser.add_argument("--no_pre_layernorm", dest="pre_layernorm", action="store_false")
    parser.set_defaults(pre_layernorm=False)

    # hypers for TransformerM
    parser.add_argument("--mode_prob", type=str, default="0.2,0.2,0.6", help="probability of {2D+3D, 2D, 3D} mode for joint training")
    parser.add_argument("--add_3d", action='store_true', help="add 3D attention bias")
    parser.add_argument("--no_2d", action='store_false', help="remove 2D encodings")
    parser.add_argument("--droppath_prob", type=float, default=0.0)
    parser.add_argument("--sandwich_ln", type=int, default=0)
    #
    parser.add_argument("--token_positional_embeddings", dest="token_positional_embeddings", action="store_true")
    parser.add_argument("--no_token_positional_embeddings", dest="token_positional_embeddings", action="store_false")
    parser.set_defaults(token_positional_embeddings=False)

    # for optimization
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--loss", type=str, default="mae", choices=["mse", "mae"])
    parser.add_argument("--print_every_epoch", type=int, default=1)
    parser.add_argument("--eval_train", type=int, default=0)
    parser.add_argument("--output_model_dir", type=str, default="")

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--no_verbose", dest="verbose", action="store_false")
    parser.set_defaults(verbose=False)
    
    args = parser.parse_args()

    node_class = args.num_atoms + 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_root = "../data/molecule_datasets/{}_Transformer".format(args.dataset)
    dataset = MoleculeDatasetQM9_2Dand3DTransformer(data_root, dataset=args.dataset, task=args.task)
    task_id = dataset.task_id

    train_dataset, valid_dataset, test_dataset = split(dataset, data_root)
    TRAIN_mean, TRAIN_std = (
        train_dataset.mean()[task_id].item(),
        train_dataset.std()[task_id].item(),
    )
    print("Train mean: {}\tTrain std: {}".format(TRAIN_mean, TRAIN_std))

    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    else:
        raise ValueError("Loss {} not included.".format(args.loss))

    if args.model_3d == "graphormer":
        collate_fn = collate_fn_2D
        model = Graphormer(args)
    elif args.model_3d == "transformerM":
        collate_fn = collate_fn_3D
        model = TransformerM(args)
    model = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda data_list: collate_fn(data_list, max_node=args.max_nodes, multi_hop_max_dist=args.multi_hop_max_dist)
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda data_list: collate_fn(data_list, max_node=args.max_nodes, multi_hop_max_dist=args.multi_hop_max_dist)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda data_list: collate_fn(data_list, max_node=args.max_nodes, multi_hop_max_dist=args.multi_hop_max_dist)
    )

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = [{"params": model.parameters(), "lr": args.lr}]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    lr_scheduler = None
    if args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs
        )
        print("Apply lr scheduler CosineAnnealingLR")
    elif args.lr_scheduler == "CosineAnnealingWarmRestarts":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, args.epochs, eta_min=1e-4
        )
        print("Apply lr scheduler CosineAnnealingWarmRestarts")
    elif args.lr_scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor
        )
        print("Apply lr scheduler StepLR")
    elif args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.lr_decay_factor, patience=args.lr_decay_patience, min_lr=args.min_lr
        )
        print("Apply lr scheduler ReduceLROnPlateau")
    else:
        print("lr scheduler {} is not included.".format(args.lr_scheduler))

    train_mae_list, val_mae_list, test_mae_list = [], [], []
    best_val_mae, best_val_idx = 1e10, 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_acc = train(epoch, device, train_loader, optimizer)
        print("Epoch: {}\nLoss: {}".format(epoch, loss_acc))

        if epoch % args.print_every_epoch == 0:
            if args.eval_train:
                train_mae, train_target, train_pred = eval(device, train_loader)
            else:
                train_mae = 0
            val_mae, val_target, val_pred = eval(device, val_loader)
            test_mae, test_target, test_pred = eval(device, test_loader)

            train_mae_list.append(train_mae)
            val_mae_list.append(val_mae)
            test_mae_list.append(test_mae)
            print(
                "train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
                    train_mae, val_mae, test_mae
                )
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_idx = len(train_mae_list) - 1
                if not args.output_model_dir == "":
                    save_model(save_best=True)

                    filename = os.path.join(
                        args.output_model_dir, "evaluation_best.pth"
                    )
                    np.savez(
                        filename,
                        val_target=val_target,
                        val_pred=val_pred,
                        test_target=test_target,
                        test_pred=test_pred,
                    )
        print("Took\t{}\n".format(time.time() - start_time))

    print(
        "best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
            train_mae_list[best_val_idx],
            val_mae_list[best_val_idx],
            test_mae_list[best_val_idx],
        )
    )

    save_model(save_best=False)
