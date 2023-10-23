import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_max_pool, global_mean_pool
from tqdm import tqdm
from torch.utils.data import DataLoader as TorchDataLoader

from config import args
from Geom3D.datasets import DatasetECSingle, DatasetGVP, DatasetECSingleGearNet
from Geom3D.models import ProNet, GearNetIEConv, MQAModel, CD_Convolution
import Geom3D.models.GearNet_layer as GearNet_layer


def model_setup():
    num_class = 384

    if args.model_3d == "GVP":
        node_in_dim = (6, 3)
        node_h_dim = (100, 16)
        edge_in_dim = (32, 1)
        edge_h_dim = (32, 1)
        model = MQAModel(node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, out_channels=num_class)
        graph_pred_linear = None

    elif args.model_3d == "GearNet":
        input_dim = 21
        model = GearNetIEConv(
            input_dim=input_dim, embedding_dim=512, hidden_dims=[512, 512, 512, 512, 512, 512], num_relation=args.num_relation,
            batch_norm=True, concat_hidden=True, short_cut=True, readout=args.GearNet_readout, layer_norm=True, dropout=args.GearNet_dropout,
            edge_input_dim=args.GearNet_edge_input_dim, num_angle_bin=args.GearNet_num_angle_bin)

        num_mlp_layer = 3
        hidden_dims = [model.output_dim] * (num_mlp_layer - 1)
        graph_pred_linear = GearNet_layer.MultiLayerPerceptron(
            model.output_dim, hidden_dims + [num_class], batch_norm=True, dropout=0.5)
        
    elif args.model_3d == "GearNet_IEConv":
        input_dim = 21
        model = GearNetIEConv(
            input_dim=input_dim, embedding_dim=512, hidden_dims=[512, 512, 512, 512, 512, 512], num_relation=7,
            batch_norm=True, concat_hidden=True, short_cut=True, readout="sum", layer_norm=True, dropout=0.2, use_ieconv=True)

        num_mlp_layer = 3
        hidden_dims = [model.output_dim] * (num_mlp_layer - 1)
        graph_pred_linear = GearNet_layer.MultiLayerPerceptron(
            model.output_dim, hidden_dims + [num_class], batch_norm=True, dropout=0.5)

    elif args.model_3d == "ProNet":
        model = ProNet(
            level=args.ProNet_level,
            dropout=args.ProNet_dropout,
            out_channels=num_class,
            euler_noise=args.euler_noise,
        )
        graph_pred_linear = None

    elif args.model_3d == "CDConv":
        geometric_radii = [x * args.CDConv_radius for x in args.CDConv_geometric_raddi_coeff]
        model = CD_Convolution(
            geometric_radii=geometric_radii,
            sequential_kernel_size=args.CDConv_kernel_size,
            kernel_channels=args.CDConv_kernel_channels, channels=args.CDConv_channels, base_width=args.CDConv_base_width,
            num_classes=num_class)
        graph_pred_linear = None
        
    else:
        raise Exception("3D model {} not included.".format(args.model_3d))
    return model, graph_pred_linear


def load_model(model, graph_pred_linear, model_weight_file):
    print("Loading from {}".format(model_weight_file))
    if "MoleculeSDE" in model_weight_file:
        model_weight = torch.load(model_weight_file)
        model.load_state_dict(model_weight["model_3D"])
        if (graph_pred_linear is not None) and ("graph_pred_linear" in model_weight):
            graph_pred_linear.load_state_dict(model_weight["graph_pred_linear"])

    else:
        model_weight = torch.load(model_weight_file)
        model.load_state_dict(model_weight["model"])
        if (graph_pred_linear is not None) and ("graph_pred_linear" in model_weight):
            graph_pred_linear.load_state_dict(model_weight["graph_pred_linear"])
    return


def save_model(save_best):
    if not args.output_model_dir == "":
        if save_best:
            print("save model with optimal loss")
            output_model_path = os.path.join(args.output_model_dir, "model.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            if graph_pred_linear is not None:
                saved_model_dict["graph_pred_linear"] = graph_pred_linear.state_dict()
            torch.save(saved_model_dict, output_model_path)

        else:
            print("save model in the last epoch")
            output_model_path = os.path.join(args.output_model_dir, "model_final.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            if graph_pred_linear is not None:
                saved_model_dict["graph_pred_linear"] = graph_pred_linear.state_dict()
            torch.save(saved_model_dict, output_model_path)
    return


def train(epoch, device, loader, optimizer):
    model.train()
    if graph_pred_linear is not None:
        graph_pred_linear.train()

    loss_acc = 0
    num_iters = len(loader)

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        if args.model_3d == "ProNet":
            if args.mask:
                # random mask node aatype
                mask_indice = torch.tensor(np.random.choice(batch.num_nodes, int(batch.num_nodes * args.mask_aatype), replace=False))
                batch.x[:, 0][mask_indice] = 25
            if args.noise:
                # add gaussian noise to atom coords
                gaussian_noise = torch.clip(torch.normal(mean=0.0, std=0.1, size=batch.coords_ca.shape), min=-0.3, max=0.3)
                batch.coords_ca += gaussian_noise
                if args.ProNet_level != 'aminoacid':
                    batch.coords_n += gaussian_noise
                    batch.coords_c += gaussian_noise
            if args.deform:
                # Anisotropic scale
                deform = torch.clip(torch.normal(mean=1.0, std=0.1, size=(1, 3)), min=0.9, max=1.1)
                batch.coords_ca *= deform
                if args.ProNet_level != 'aminoacid':
                    batch.coords_n *= deform
                    batch.coords_c *= deform

        batch = batch.to(device)

        if args.model_3d == "GVP":
            molecule_3D_repr = model(batch=batch)
        elif args.model_3d in ["GearNet", "GearNet_IEConv"]:
            molecule_3D_repr = model(batch, batch.node_feature.float())["graph_feature"]
        elif args.model_3d == "ProNet":
            molecule_3D_repr = model(batch)
        elif args.model_3d == "CDConv":
            molecule_3D_repr = model(batch, split="training")
        elif args.model_3d == "FrameNetProtein":
            molecule_3D_repr = model(batch.coords_n, batch.coords_ca, batch.coords_c, batch.seq, batch.batch)

        if graph_pred_linear is not None:
            pred = graph_pred_linear(molecule_3D_repr).squeeze(1)
        else:
            pred = molecule_3D_repr.squeeze(1)

        y = batch.y

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
    if graph_pred_linear is not None:
        graph_pred_linear.eval()
    y_true = []
    y_scores = []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for batch in L:
        if args.model_3d == "ProNet":
            if args.mask:
                # random mask node aatype
                mask_indice = torch.tensor(np.random.choice(batch.num_nodes, int(batch.num_nodes * args.mask_aatype), replace=False))
                batch.x[:, 0][mask_indice] = 25
            if args.noise:
                # add gaussian noise to atom coords
                gaussian_noise = torch.clip(torch.normal(mean=0.0, std=0.1, size=batch.coords_ca.shape), min=-0.3, max=0.3)
                batch.coords_ca += gaussian_noise
                if args.ProNet_level != 'aminoacid':
                    batch.coords_n += gaussian_noise
                    batch.coords_c += gaussian_noise
            if args.deform:
                # Anisotropic scale
                deform = torch.clip(torch.normal(mean=1.0, std=0.1, size=(1, 3)), min=0.9, max=1.1)
                batch.coords_ca *= deform
                if args.ProNet_level != 'aminoacid':
                    batch.coords_n *= deform
                    batch.coords_c *= deform

        batch = batch.to(device)
        
        if args.model_3d == "GVP":
            molecule_3D_repr = model(batch=batch)
        elif args.model_3d in ["GearNet", "GearNet_IEConv"]:
            molecule_3D_repr = model(batch, batch.node_feature.float())["graph_feature"]
        elif args.model_3d == "ProNet":
            molecule_3D_repr = model(batch)
        elif args.model_3d == "CDConv":
            molecule_3D_repr = model(batch)
        elif args.model_3d == "FrameNetProtein":
            molecule_3D_repr = model(batch.coords_n, batch.coords_ca, batch.coords_c, batch.seq, batch.batch)

        if graph_pred_linear is not None:
            pred = graph_pred_linear(molecule_3D_repr).squeeze(1)
        else:
            pred = molecule_3D_repr.squeeze(1)
        pred = pred.argmax(dim=-1)

        y = batch.y

        y_true.append(y)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    L = len(y_true)
    acc =  sum(y_true == y_scores) * 1. / L
    return acc


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_root = args.data_root
    dataset_class = DatasetECSingle
    if args.model_3d == "GearNet":
        dataset_class = DatasetECSingleGearNet

    train_dataset = dataset_class(root=data_root, split='Train')
    valid_dataset = dataset_class(root=data_root, split='Val')
    test_dataset = dataset_class(root=data_root, split='Test')

    if args.model_3d == "GVP":
        data_root = "../data/ECSingle_GVP"
        train_dataset = DatasetGVP(
            data_root, train_dataset, split='Train', num_positional_embeddings=args.num_positional_embeddings, top_k=args.top_k, num_rbf=args.num_rbf)
        valid_dataset = DatasetGVP(
            data_root, valid_dataset, split='Val', num_positional_embeddings=args.num_positional_embeddings, top_k=args.top_k, num_rbf=args.num_rbf)
        test_dataset = DatasetGVP(
            data_root, test_dataset, split='Test', num_positional_embeddings=args.num_positional_embeddings, top_k=args.top_k, num_rbf=args.num_rbf)
    
    criterion = nn.CrossEntropyLoss()

    DataLoaderClass = PyGDataLoader
    dataloader_kwargs = {}

    if args.model_3d in ["GearNet", "GearNet_IEConv"]:
        dataloader_kwargs["collate_fn"] = DatasetECSingleGearNet.collate_fn
        DataLoaderClass = TorchDataLoader
    
    train_loader = DataLoaderClass(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        **dataloader_kwargs
    )
    val_loader = DataLoaderClass(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        **dataloader_kwargs
    )
    test_loader = DataLoaderClass(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        **dataloader_kwargs
    )

    model, graph_pred_linear = model_setup()

    if args.input_model_file is not "":
        load_model(model, graph_pred_linear, args.input_model_file)
    model.to(device)
    print(model)
    if graph_pred_linear is not None:
        graph_pred_linear.to(device)
    print(graph_pred_linear)

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = [{"params": model.parameters(), "lr": args.lr}]
    if graph_pred_linear is not None:
        model_param_group.append(
            {"params": graph_pred_linear.parameters(), "lr": args.lr}
        )
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model_param_group, lr=args.lr, weight_decay=5e-4, momentum=0.9)

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
    elif args.lr_scheduler == "StepLRCustomized":
        print("Will decay with {}, at epochs {}".format(args.lr_decay_factor, args.StepLRCustomized_scheduler))
        print("Apply lr scheduler StepLR (customized)")
    else:
        print("lr scheduler {} is not included.".format(args.lr_scheduler))
    global_learning_rate = args.lr

    train_acc_list, val_acc_list, test_acc_list = [], [], []
    best_val_acc, best_val_idx = -1e10, 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_acc = train(epoch, device, train_loader, optimizer)
        print("Epoch: {}\nLoss: {}".format(epoch, loss_acc))

        if epoch % args.print_every_epoch == 0:
            if args.eval_train:
                train_acc, train_target, train_pred = eval(device, train_loader)
            else:
                train_acc = 0
            val_acc = eval(device, val_loader)
            test_acc = eval(device, test_loader)

            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)
            print(
                "train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
                    train_acc, val_acc, test_acc
                )
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_idx = len(train_acc_list) - 1
                if not args.output_model_dir == "":
                    save_model(save_best=True)
        
        if args.lr_scheduler == "StepLRCustomized" and epoch in args.StepLRCustomized_scheduler:
            print('ChanGINg learning rate, from {} to {}'.format(global_learning_rate, global_learning_rate * args.lr_decay_factor)),
            global_learning_rate *= args.lr_decay_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = global_learning_rate
        print("Took\t{}\n".format(time.time() - start_time))

    print(
        "best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
            train_acc_list[best_val_idx],
            val_acc_list[best_val_idx],
            test_acc_list[best_val_idx],
        )
    )

    save_model(save_best=False)
