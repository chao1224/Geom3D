import argparse
import os
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Geom3D.datasets import MoleculeDatasetQM92D, MoleculeDataset2DDense
from Geom3D.models import AWARE
from utils_NGram import qm9_random_customized_01, qm9_random_customized_02, split, extract_data_dict, AtomEncoder, BondEncoder


def mean_absolute_error(pred, target):
    return np.mean(np.abs(pred - target))


def run_N_Gram_Graph_RF(train_data_dict, test_data_dict, args):
    from sklearn.ensemble import RandomForestRegressor

    X_train, y_train = train_data_dict["ngram_graph_matrix_list"], train_data_dict["label_list"][:, task_id]
    y_train = (y_train - TRAIN_mean) / TRAIN_std
    X_test, y_test = test_data_dict["ngram_graph_matrix_list"], test_data_dict["label_list"][:, task_id]

    model = RandomForestRegressor(
        n_estimators=args.RF_n_estimators,
        max_features=args.RF_max_features,
        min_samples_leaf=args.RF_min_samples_leaf,
        n_jobs=8,
        random_state=args.RF_random_seed,
        oob_score=False,
        verbose=args.verbose
    )
    model.fit(X_train, y_train)

    y_pred_on_train = model.predict(X_train)
    y_pred_on_train = y_pred_on_train * TRAIN_std + TRAIN_mean
    y_train = y_train * TRAIN_std + TRAIN_mean
    train_mae = mean_absolute_error(y_pred_on_train, y_train)

    y_pred_on_test = model.predict(X_test)
    y_pred_on_test = y_pred_on_test * TRAIN_std + TRAIN_mean
    test_mae = mean_absolute_error(y_pred_on_test, y_test)

    print("best train: {:.6f}\ttest: {:.6f}".format(train_mae, test_mae))
    
    if not args.output_model_dir == "":
        import joblib
        output_model_path = os.path.join(args.output_model_dir, "model.pth")
        joblib.dump(model, output_model_path, compress=3)
    return


def run_N_Gram_Graph_XGB(train_data_dict, test_data_dict, args):
    from xgboost import XGBRegressor

    X_train, y_train = train_data_dict["ngram_graph_matrix_list"], train_data_dict["label_list"][:, task_id]
    y_train = (y_train - TRAIN_mean) / TRAIN_std
    X_test, y_test = test_data_dict["ngram_graph_matrix_list"], test_data_dict["label_list"][:, task_id]

    model = XGBRegressor(
        max_depth=args.XGB_max_depth,
        learning_rate=args.XGB_learning_rate,
        n_estimators=args.XGB_n_estimators,
        objective=args.XGB_objective,
        booster=args.XGB_booster,
        subsample=args.XGB_subsample,
        colsample_bylevel=args.XGB_colsample_bylevel,
        colsample_bytree=args.XGB_colsample_bytree,
        min_child_weight=args.XGB_min_child_weight,
        reg_alpha=args.XGB_reg_alpha,
        reg_lambda=args.XGB_reg_lambda,
        n_jobs=8,
        random_state=args.XGB_random_seed,
        verbosity=int(args.verbose),
    )
    model.fit(X_train, y_train)

    y_pred_on_train = model.predict(X_train)
    y_pred_on_train = y_pred_on_train * TRAIN_std + TRAIN_mean
    y_train = y_train * TRAIN_std + TRAIN_mean
    train_mae = mean_absolute_error(y_pred_on_train, y_train)

    y_pred_on_test = model.predict(X_test)
    y_pred_on_test = y_pred_on_test * TRAIN_std + TRAIN_mean
    test_mae = mean_absolute_error(y_pred_on_test, y_test)

    print("best train: {:.6f}\ttest: {:.6f}".format(train_mae, test_mae))
    if not args.output_model_dir == "":
        import joblib
        output_model_path = os.path.join(args.output_model_dir, "model.pth")
        joblib.dump(model, output_model_path, compress=3)
    return


def train_AWARE(epoch, device, loader, optimizer):
    atom_encoder.train()
    bond_encoder.train()
    model.train()

    loss_acc = 0
    num_iters = len(loader)

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        node_attr_matrix, adj_matrix, adj_attr_matrix, node_num, label = batch
        node_attr_matrix = node_attr_matrix.to(device)
        adj_matrix = adj_matrix.to(device)
        label = label.to(device)

        node_attr_matrix = atom_encoder(node_attr_matrix)
        node_mask = node_attr_matrix.sum(dim=2, keepdim=True)
        node_attr_matrix = node_attr_matrix * node_mask

        if args.use_bond:
            adj_attr_matrix = adj_attr_matrix.to(device)
            adj_attr_matrix = bond_encoder(adj_attr_matrix)
            adj_mask = adj_matrix.unsqueeze(dim=3)
            adj_attr_matrix = adj_attr_matrix * adj_mask
        else:
            adj_attr_matrix = None

        pred = model(node_attr_matrix, adj_matrix, adj_attr_matrix).squeeze(dim=1)
        
        B = pred.size()[0]
        y = label[:, task_id]
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
def eval_AWARE(device, loader):
    atom_encoder.eval()
    bond_encoder.train()
    model.train()
    y_true = []
    y_scores = []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for batch in L:
        node_attr_matrix, adj_matrix, adj_attr_matrix, node_num, label = batch
        node_attr_matrix = node_attr_matrix.to(device)
        adj_matrix = adj_matrix.to(device)
        label = label.to(device)

        node_attr_matrix = atom_encoder(node_attr_matrix)
        mask = node_attr_matrix.sum(dim=2, keepdim=True)
        node_attr_matrix = node_attr_matrix * mask

        if args.use_bond:
            adj_attr_matrix = adj_attr_matrix.to(device)
            adj_attr_matrix = bond_encoder(adj_attr_matrix)
            adj_mask = adj_matrix.unsqueeze(dim=3)
            adj_attr_matrix = adj_attr_matrix * adj_mask
        else:
            adj_attr_matrix = None

        pred = model(node_attr_matrix, adj_matrix, adj_attr_matrix).squeeze(dim=1)

        B = pred.size()[0]
        y = label[:, task_id]
        # denormalize
        pred = pred * TRAIN_std + TRAIN_mean

        y_true.append(y)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    mae = mean_absolute_error(y_scores, y_true)
    return mae, y_true, y_scores


def save_model(save_best):
    if not args.output_model_dir == "":
        if save_best:
            print("save model with optimal loss")
            output_model_path = os.path.join(args.output_model_dir, "model.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            saved_model_dict["atom_encoder"] = atom_encoder.state_dict()
            saved_model_dict["bond_encoder"] = bond_encoder.state_dict()
            torch.save(saved_model_dict, output_model_path)

        else:
            print("save model in the last epoch")
            output_model_path = os.path.join(args.output_model_dir, "model_final.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            saved_model_dict["atom_encoder"] = atom_encoder.state_dict()
            saved_model_dict["bond_encoder"] = bond_encoder.state_dict()
            torch.save(saved_model_dict, output_model_path)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="N_Gram_Graph", choices=["N_Gram_Graph", "AWARE"])
    parser.add_argument("--dataset", type=str, default="qm9")
    parser.add_argument("--task", type=str, default="alpha")
    parser.add_argument("--split", type=str, default="customized_01")
    parser.add_argument("--emb_dim", type=int, default=100)

    parser.add_argument("--max_node_num", type=int, default=35)

    parser.add_argument("--use_bond", dest="use_bond", action="store_true")
    parser.add_argument("--no_bond", dest="use_bond", action="store_false")
    parser.set_defaults(use_bond=True)

    parser.add_argument("--use_3D_coordinates", dest="use_3D_coordinates", action="store_true")
    parser.add_argument("--no_3D_coordinates", dest="use_3D_coordinates", action="store_false")
    parser.set_defaults(use_3D_coordinates=False)

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--no_verbose", dest="verbose", action="store_false")
    parser.set_defaults(verbose=True)

    # for N-Gram Graph
    parser.add_argument("--N_Gram_Graph_predictor_type", type=str, default="RF", choices=["RF", "XGB"])
    parser.add_argument("--N_Gram_Graph_normalize", dest="N_Gram_Graph_normalize", action="store_true")
    parser.add_argument("--no_N_Gram_Graph_normalize", dest="N_Gram_Graph_normalize", action="store_false")
    parser.set_defaults(N_Gram_Graph_normalize=False)

    # for N-Gram Graph (RF)
    parser.add_argument("--RF_random_seed", type=int, default=42)
    parser.add_argument("--RF_min_samples_leaf", type=int, default=1)
    parser.add_argument("--RF_n_estimators", type=int, default=10)
    parser.add_argument("--RF_max_features", type=str, default=None)

    # for N-Gram Graph (XGB)
    parser.add_argument("--XGB_random_seed", type=int, default=42)
    parser.add_argument("--XGB_n_estimators", type=int, default=10)
    parser.add_argument("--XGB_max_depth", type=int, default=50)
    parser.add_argument("--XGB_min_child_weight", type=float, default=1)
    parser.add_argument("--XGB_subsample", type=float, default=0.5)
    parser.add_argument("--XGB_colsample_bylevel", type=float, default=0.5)
    parser.add_argument("--XGB_colsample_bytree", type=float, default=0.5)
    parser.add_argument("--XGB_reg_alpha", type=float, default=0)
    parser.add_argument("--XGB_reg_lambda", type=float, default=0.7)
    parser.add_argument("--XGB_learning_rate", type=float, default=0.03)
    parser.add_argument("--XGB_objective", type=str, default="reg:squarederror")
    parser.add_argument("--XGB_booster", type=str, default="gbtree")

    # for AWARE
    parser.add_argument("--r_prime", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--loss", type=str, default="mae", choices=["mse", "mae"])
    parser.add_argument("--lr_scheduler", type=str, default='CosineAnnealingLR')
    parser.add_argument("--print_every_epoch", type=int, default=1)
    parser.add_argument("--eval_train", type=int, default=0)
    parser.add_argument("--max_walk_len", type=int, default=6)
    parser.add_argument("--num_layers", type=int, default=5)

    # for dadtaloader
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_model_dir", type=str, default="")
    args = parser.parse_args()
    print(args)

    assert args.dataset == "qm9"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # TODO: only for preprocessing
    data_root = "../data/molecule_datasets/qm9_2D"
    preprocessed_dataset = MoleculeDatasetQM92D(data_root, dataset=args.dataset, task=args.task)
    task_id = preprocessed_dataset.task_id

    data_root = "../data/molecule_datasets/qm9_2D_NGramPath"
    dataset = MoleculeDataset2DDense(
        root=data_root,
        preprocessed_dataset=preprocessed_dataset,
        max_node_num=args.max_node_num,
    )
    
    train_dataset, valid_dataset, test_dataset = split(args, dataset, data_root)

    labels_list = []
    for data in train_dataset:
        labels_list.append(data[-1])
    labels_list = torch.stack(labels_list, dim=0)
    print("labels_list", labels_list.size())

    TRAIN_mean, TRAIN_std = (
        labels_list.mean(dim=0)[task_id].item(),
        labels_list.std(dim=0)[task_id].item(),
    )
    print("Train mean: {}\tTrain std: {}".format(TRAIN_mean, TRAIN_std))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    if args.model == "N_Gram_Graph":
        train_data_dict = extract_data_dict([train_loader, valid_loader], emb_dim=args.emb_dim, use_bond=args.use_bond, use_3D_coordinates=args.use_3D_coordinates, normalize=args.N_Gram_Graph_normalize)
        test_data_dict = extract_data_dict([test_loader], emb_dim=args.emb_dim, use_bond=args.use_bond, use_3D_coordinates=args.use_3D_coordinates, normalize=args.N_Gram_Graph_normalize)

        if args.N_Gram_Graph_predictor_type == "RF":
            run_N_Gram_Graph_RF(train_data_dict, test_data_dict, args)

        elif args.N_Gram_Graph_predictor_type == "XGB":
            run_N_Gram_Graph_XGB(train_data_dict, test_data_dict, args)
    
    elif args.model == "AWARE":
        
        device = (
            torch.device("cuda:" + str(args.device))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        model = AWARE(
            emb_dim=args.emb_dim,
            r_prime=args.r_prime,
            max_walk_len=args.max_walk_len,
            num_layers=args.num_layers,
            out_dim=1,
            use_bond=args.use_bond
        ).to(device)

        atom_encoder = AtomEncoder(args.emb_dim).to(device)
        bond_encoder = BondEncoder(args.emb_dim).to(device)

        model_param_group = [{"params": atom_encoder.parameters(), "lr": args.lr}]
        model_param_group += [{"params": bond_encoder.parameters(), "lr": args.lr}]
        model_param_group += [{"params": model.parameters(), "lr": args.lr}]

        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

        if args.loss == "mse":
            criterion = nn.MSELoss()
        elif args.loss == "mae":
            criterion = nn.L1Loss()
        else:
            raise ValueError("Loss {} not included.".format(args.loss))

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
        for epoch in range(1, 1+args.epochs):
            start_time = time.time()
            loss_acc = train_AWARE(epoch, device, train_loader, optimizer)
            print("Epoch: {}\nLoss: {}".format(epoch, loss_acc))

            if epoch % args.print_every_epoch == 0:
                if args.eval_train:
                    train_mae, train_target, train_pred = eval_AWARE(device, train_loader)
                else:
                    train_mae = 0
                val_mae, val_target, val_pred = eval_AWARE(device, valid_loader)
                test_mae, test_target, test_pred = eval_AWARE(device, test_loader)

                train_mae_list.append(train_mae)
                val_mae_list.append(val_mae)
                test_mae_list.append(test_mae)
                print("train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_mae, val_mae, test_mae))

                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_val_idx = len(train_mae_list) - 1
                    if not args.output_model_dir == "":
                        save_model(save_best=True)

                        filename = os.path.join(args.output_model_dir, "evaluation_best.pth")
                        np.savez(
                            filename,
                            val_target=val_target,
                            val_pred=val_pred,
                            test_target=test_target,
                            test_pred=test_pred,
                        )
            print("Took\t{}\n".format(time.time() - start_time))

        print("best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
            train_mae_list[best_val_idx],
            val_mae_list[best_val_idx],
            test_mae_list[best_val_idx],
        ))
