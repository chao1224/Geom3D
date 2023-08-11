import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from config import args
from Geom3D.datasets import DatasetMD17, DatasetrMD17, MoleculeDataset3DRadius
from Geom3D.dataloaders import DataLoaderGemNet
from Geom3D.models import SchNet, DimeNetPlusPlus, EGNN, SphereNet, PaiNN, GemNet, SEGNN, EquiformerEnergyForce
from Geom3D.models.NequIP.model import model_from_config

from torch.autograd import grad


def preprocess_input(one_hot, charges, charge_power, charge_scale):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1.0, device=device, dtype=torch.float32)
    )  # (-1, 3)
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (
        one_hot.unsqueeze(-1) * charge_tensor
    )  # (N, charge_scale, charge_power + 1)
    atom_scalars = atom_scalars.view(
        charges.shape[:1] + (-1,)
    )  # (N, charge_scale * (charge_power + 1) )
    return atom_scalars


def model_setup():
    if args.model_3d == "SchNet":
        model = SchNet(
            hidden_channels=args.emb_dim,
            num_filters=args.SchNet_num_filters,
            num_interactions=args.SchNet_num_interactions,
            num_gaussians=args.SchNet_num_gaussians,
            cutoff=args.SchNet_cutoff,
            readout=args.SchNet_readout,
            node_class=node_class,
        )
        graph_pred_linear = torch.nn.Linear(args.emb_dim, num_tasks)

    elif args.model_3d == "DimeNetPlusPlus":
        model = DimeNetPlusPlus(
            node_class=node_class,
            hidden_channels=args.emb_dim,
            out_channels=num_tasks,
            num_blocks=args.DimeNetPlusPlus_num_blocks,
            int_emb_size=args.DimeNetPlusPlus_int_emb_size,
            basis_emb_size=args.DimeNetPlusPlus_basis_emb_size,
            out_emb_channels=args.DimeNetPlusPlus_out_emb_channels,
            num_spherical=args.DimeNetPlusPlus_num_spherical,
            num_radial=args.DimeNetPlusPlus_num_radial,
            cutoff=args.DimeNetPlusPlus_cutoff,
            envelope_exponent=args.DimeNetPlusPlus_envelope_exponent,
            num_before_skip=args.DimeNetPlusPlus_num_before_skip,
            num_after_skip=args.DimeNetPlusPlus_num_after_skip,
            num_output_layers=args.DimeNetPlusPlus_num_output_layers,
        )
        graph_pred_linear = None

    elif args.model_3d == "EGNN":
        in_node_nf = node_class * (1 + args.EGNN_charge_power)
        model = EGNN(
            in_node_nf=in_node_nf,
            in_edge_nf=0,
            hidden_nf=args.emb_dim,
            n_layers=args.EGNN_n_layers,
            positions_weight=args.EGNN_positions_weight,
            attention=args.EGNN_attention,
            node_attr=args.EGNN_node_attr,
        )
        graph_pred_linear = nn.Sequential(
            nn.Linear(args.emb_dim, args.emb_dim),
            nn.SiLU(),
            nn.Linear(args.emb_dim, num_tasks),
        )

    elif args.model_3d == "SEGNN":
        model = SEGNN(
            node_class,
            num_tasks,
            hidden_features=args.emb_dim,
            N=args.SEGNN_N,
            lmax_h=args.SEGNN_lmax_h,
            lmax_pos=args.SEGNN_lmax_pos,
            norm=args.SEGNN_norm,
            pool=args.SEGNN_pool,
            edge_inference=args.SEGNN_edge_inference
        )
        graph_pred_linear = None

    elif args.model_3d == "SphereNet":
        model = SphereNet(
            hidden_channels=args.emb_dim,
            out_channels=num_tasks,
            energy_and_force=True,
            cutoff=args.SphereNet_cutoff,
            num_layers=args.SphereNet_num_layers,
            int_emb_size=args.SphereNet_int_emb_size,
            basis_emb_size_dist=args.SphereNet_basis_emb_size_dist,
            basis_emb_size_angle=args.SphereNet_basis_emb_size_angle,
            basis_emb_size_torsion=args.SphereNet_basis_emb_size_torsion,
            out_emb_channels=args.SphereNet_out_emb_channels,
            num_spherical=args.SphereNet_num_spherical,
            num_radial=args.SphereNet_num_radial,
            envelope_exponent=args.SphereNet_envelope_exponent,
            num_before_skip=args.SphereNet_num_before_skip,
            num_after_skip=args.SphereNet_num_after_skip,
            num_output_layers=args.SphereNet_num_output_layers,
        )
        graph_pred_linear = None

    elif args.model_3d == "PaiNN":
        model = PaiNN(
            n_atom_basis=args.emb_dim,  # default is 64
            n_interactions=args.PaiNN_n_interactions,
            n_rbf=args.PaiNN_n_rbf,
            cutoff=args.PaiNN_radius_cutoff,
            max_z=node_class,
            n_out=num_tasks,
            readout=args.PaiNN_readout,
        )
        graph_pred_linear = model.create_output_layers()

    elif args.model_3d == "GemNet":
        model = GemNet(
            # node_class=93,
            node_class=node_class,
            num_spherical=args.GemNet_num_spherical,
            num_radial=args.GemNet_num_radial,
            num_blocks=args.GemNet_num_blocks,
            emb_size_atom=args.emb_dim,
            emb_size_edge=args.emb_dim,
            emb_size_trip=args.GemNet_emb_size_trip,
            emb_size_quad=args.GemNet_emb_size_quad,
            emb_size_rbf=args.GemNet_emb_size_rbf,
            emb_size_cbf=args.GemNet_emb_size_cbf,
            emb_size_sbf=args.GemNet_emb_size_sbf,
            emb_size_bil_quad=args.GemNet_emb_size_bil_quad,
            emb_size_bil_trip=args.GemNet_emb_size_bil_trip,
            num_before_skip=args.GemNet_num_before_skip,
            num_after_skip=args.GemNet_num_after_skip,
            num_concat=args.GemNet_num_concat,
            num_atom=args.GemNet_num_atom,
            cutoff=args.GemNet_cutoff,
            int_cutoff=args.GemNet_int_cutoff,
            triplets_only=args.GemNet_triplets_only,
            direct_forces=args.GemNet_direct_forces,
            envelope_exponent=args.GemNet_envelope_exponent,
            extensive=args.GemNet_extensive,
            forces_coupled=args.GemNet_forces_coupled,
            output_init=args.GemNet_output_init,
            activation=args.GemNet_activation,
            scale_file=args.GemNet_scale_file,
            num_targets=num_tasks,
        )
        graph_pred_linear = None

    elif args.model_3d == "NequIP":
        # reference to https://github.com/mir-group/NequIP/discussions/131
        config = dict(
            model_builders=[
                "SimpleIrrepsConfig",
                "EnergyModel",
            ],
            dataset_statistics_stride=1,
            chemical_symbols=["H", "C", "N", "O", "F", "Na", "Cl", "He", "P", "B"],
            r_max=args.NequIP_radius_cutoff,
            num_layers=5,
            chemical_embedding_irreps_out="64x0e",
            l_max=1,
            parity=True,
            num_features=64,
            nonlinearity_type="gate",
            nonlinearity_scalars={'e': 'silu', 'o': 'tanh'},
            nonlinearity_gates={'e': 'silu', 'o': 'tanh'},
            resnet=False,
            num_basis=8,
            BesselBasis_trainable=True,
            PolynomialCutoff_p=6,
            invariant_layers=3,
            invariant_neurons=64,
            avg_num_neighbors=8,
            use_sc=True,
            compile_model=False,
        )
        model = model_from_config(config=config, initialize=True)
        graph_pred_linear = None

    elif args.model_3d == "Allegro":
        config = dict(
            model_builders=[
                "Geom3D.models.Allegro.model.Allegro",
            ],
            dataset_statistics_stride=1,
            chemical_symbols=["H", "C", "N", "O", "F", "Na", "Cl", "He", "P", "B"],
            default_dtype="float32",
            allow_tf32=False,
            model_debug_mode=False,
            equivariance_test=False,
            grad_anomaly_mode=False,
            _jit_bailout_depth=2,
            _jit_fusion_strategy=[("DYNAMIC", 3)],
            r_max=args.NequIP_radius_cutoff,
            num_layers=5,
            l_max=1,
            num_features=64,
            nonlinearity_type="gate",
            nonlinearity_scalars={'e': 'silu', 'o': 'tanh'},
            nonlinearity_gates={'e': 'silu', 'o': 'tanh'},
            num_basis=8,
            BesselBasis_trainable=True,
            PolynomialCutoff_p=6,
            invariant_layers=3,
            invariant_neurons=64,
            avg_num_neighbors=8,
            use_sc=True,
            
            parity="o3_full",
            mlp_latent_dimensions=[512],
        )
        model = model_from_config(config=config, initialize=True)
        graph_pred_linear = None

    elif args.model_3d == "Equiformer":
        model = EquiformerEnergyForce(
            irreps_in=args.Equiformer_irreps_in,
            max_radius=args.Equiformer_radius,
            node_class=node_class,
            number_of_basis=args.Equiformer_num_basis, 
            irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
            irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
            fc_neurons=[64, 64], basis_type='exp',
            irreps_feature='512x0e',
            irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
            rescale_degree=False, nonlinear_message=True,
            irreps_mlp_mid='384x0e+192x1e+96x2e',
            norm_layer='layer',
            alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0)
        graph_pred_linear = None

    else:
        raise Exception("3D model {} not included.".format(args.model_3d))

    return

def train(device, loader):
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
        batch = batch.to(device)
        positions = batch.positions
        positions.requires_grad_()
        x = batch.x

        if args.model_3d == "SchNet":
            molecule_3D_repr = model(x, positions, batch.batch)
        elif args.model_3d == "DimeNetPlusPlus":
            molecule_3D_repr = model(x, positions, batch.batch)
        elif args.model_3d == "SphereNet":
            molecule_3D_repr = model(x, positions, batch.batch)
        elif args.model_3d == "PaiNN":
            molecule_3D_repr = model(x, positions, batch.radius_edge_index, batch.batch)
        elif args.model_3d == "GemNet":
            molecule_3D_repr = model(x, positions, batch)
        elif args.model_3d == "EGNN":
            x_one_hot = F.one_hot(x, num_classes=node_class)
            x = preprocess_input(
                x_one_hot,
                x,
                charge_power=args.EGNN_charge_power,
                charge_scale=node_class,
            )
            node_3D_repr = model(
                x=x,
                positions=positions,
                edge_index=batch.radius_edge_index,
                edge_attr=None,
            )
            molecule_3D_repr = global_mean_pool(node_3D_repr, batch.batch)
        elif args.model_3d == "SEGNN":
            molecule_3D_repr = model(batch)
        elif args.model_3d in ["NequIP", "Allegro"]:
            data = {
                "atom_types": x,
                "pos": positions,
                "edge_index": batch.radius_edge_index,
                "batch": batch.batch,
            }
            out = model(data)
            molecule_3D_repr = out["total_energy"]
        elif args.model_3d == "Equiformer":
            molecule_3D_repr = model(node_atom=x, pos=positions, batch=batch.batch)

        if graph_pred_linear is not None:
            pred_energy = graph_pred_linear(molecule_3D_repr).squeeze(1)
        else:
            pred_energy = molecule_3D_repr.squeeze(1)

        if args.energy_force_with_normalization:
            pred_energy = pred_energy * FORCE_MEAN_TOTAL + ENERGY_MEAN_TOTAL * NUM_ATOM

        pred_force = -grad(outputs=pred_energy, inputs=positions, grad_outputs=torch.ones_like(pred_energy), create_graph=True, retain_graph=True)[0]

        actual_energy = batch.y
        actual_force = batch.force

        loss = args.md17_energy_coeff * criterion(pred_energy, actual_energy) + args.md17_force_coeff * criterion(pred_force, actual_force)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_acc += loss.cpu().detach().item()

        if args.lr_scheduler in ["CosineAnnealingWarmRestarts"]:
            lr_scheduler.step(epoch - 1 + step / num_iters)

    loss_acc /= len(loader)
    if args.lr_scheduler in ["StepLR", "CosineAnnealingLR"]:
        lr_scheduler.step()
    elif args.lr_scheduler in ["ReduceLROnPlateau"]:
        lr_scheduler.step(loss_acc)
    return loss_acc


def eval(device, loader):
    model.eval()
    if graph_pred_linear is not None:
        graph_pred_linear.eval()
    pred_energy_list, actual_energy_list = [], []
    pred_force_list = torch.Tensor([]).to(device)
    actual_force_list = torch.Tensor([]).to(device)

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader

    for batch in L:
        batch = batch.to(device)
        positions = batch.positions
        positions.requires_grad_()
        x = batch.x

        if args.model_3d == "SchNet":
            molecule_3D_repr = model(x, positions, batch.batch)
        elif args.model_3d == "DimeNetPlusPlus":
            molecule_3D_repr = model(x, positions, batch.batch)
        elif args.model_3d == "SphereNet":
            molecule_3D_repr = model(x, positions, batch.batch)
        elif args.model_3d == "PaiNN":
            molecule_3D_repr = model(x, positions, batch.radius_edge_index, batch.batch)
        elif args.model_3d == "GemNet":
            molecule_3D_repr = model(x, positions, batch)
        elif args.model_3d == "EGNN":
            x_one_hot = F.one_hot(x, num_classes=node_class)
            x = preprocess_input(
                x_one_hot,
                x,
                charge_power=args.EGNN_charge_power,
                charge_scale=node_class,
            )
            node_3D_repr = model(
                x=x,
                positions=positions,
                edge_index=batch.radius_edge_index,
                edge_attr=None,
            )
            molecule_3D_repr = global_mean_pool(node_3D_repr, batch.batch)
        elif args.model_3d == "SEGNN":
            molecule_3D_repr = model(batch)
        elif args.model_3d in ["NequIP", "Allegro"]:
            data = {
                "atom_types": x,
                "pos": positions,
                "edge_index": batch.radius_edge_index,
                "batch": batch.batch,
            }
            out = model(data)
            molecule_3D_repr = out["total_energy"]
        elif args.model_3d == "Equiformer":
            molecule_3D_repr = model(node_atom=x, pos=positions, batch=batch.batch)

        if graph_pred_linear is not None:
            pred_energy = graph_pred_linear(molecule_3D_repr).squeeze(1)
        else:
            pred_energy = molecule_3D_repr.squeeze(1)

        if args.energy_force_with_normalization:
            pred_energy = pred_energy * FORCE_MEAN_TOTAL + ENERGY_MEAN_TOTAL * NUM_ATOM

        force = -grad(outputs=pred_energy, inputs=positions, grad_outputs=torch.ones_like(pred_energy), create_graph=True, retain_graph=True)[0].detach_()

        if torch.sum(torch.isnan(force)) != 0:
            mask = torch.isnan(force)
            force = force[~mask].reshape((-1, 3))
            batch.force = batch.force[~mask].reshape((-1, 3))

        pred_energy_list.append(pred_energy.cpu().detach())
        actual_energy_list.append(batch.y.cpu())
        pred_force_list = torch.cat([pred_force_list, force], dim=0)
        actual_force_list = torch.cat([actual_force_list, batch.force], dim=0)

    pred_energy_list = torch.cat(pred_energy_list, dim=0)
    actual_energy_list = torch.cat(actual_energy_list, dim=0)
    energy_mae = torch.mean(torch.abs(pred_energy_list - actual_energy_list)).cpu().item()
    force_mae = torch.mean(torch.abs(pred_force_list - actual_force_list)).cpu().item()

    return energy_mae, force_mae


def load_model(model, graph_pred_linear, model_weight_file, load_latest=False):
    print("Loading from {}".format(model_weight_file))

    if load_latest:
        model_weight = torch.load(model_weight_file)
        model.load_state_dict(model_weight["model"])
        if (graph_pred_linear is not None) and ("graph_pred_linear" in model_weight):
            graph_pred_linear.load_state_dict(model_weight["graph_pred_linear"])

    elif "MoleculeSDE" in model_weight_file:
        model_weight = torch.load(model_weight_file)
        if "model_3D" in model_weight:
            model.load_state_dict(model_weight["model_3D"])
        else:
            model.load_state_dict(model_weight["model"])
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

    if args.dataset == "MD17":
        data_root = "../data/MD17"
        dataset = DatasetMD17(data_root, task=args.task)
        split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=args.seed)
    elif args.dataset == "rMD17":
        data_root = "../data/rMD17"
        dataset = DatasetrMD17(data_root, task=args.task, split_id=args.rMD17_split_id)
        split_idx = dataset.get_idx_split()
    print("train:", len(split_idx["train"]), split_idx["train"][:5])
    print("valid:", len(split_idx["valid"]), split_idx["valid"][:5])
    print("test:", len(split_idx["test"]), split_idx["test"][:5])

    if args.model_3d == "PaiNN":
        data_root = "../data/{}_{}/{}".format(args.dataset, args.PaiNN_radius_cutoff, args.task)
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.PaiNN_radius_cutoff
        )
    elif args.model_3d in ["NequIP", "Allegro"]:
        # Will update this
        data_root = "../data/{}_{}/{}".format(args.dataset, args.NequIP_radius_cutoff, args.task)
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.NequIP_radius_cutoff
        )
    elif args.model_3d == "EGNN":
        data_root = "../data/{}_{}/{}".format(args.dataset, args.EGNN_radius_cutoff, args.task)
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.EGNN_radius_cutoff
        )
    elif args.model_3d == "SEGNN":
        data_root = "../data/{}_{}/{}".format(args.dataset, args.SEGNN_radius, args.task)
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.SEGNN_radius
        )
    train_dataset, val_dataset, test_dataset = \
        dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

    # Remove energy mean.
    ENERGY_MEAN_TOTAL = 0
    FORCE_MEAN_TOTAL = 0
    NUM_ATOM = None
    for data in train_dataset:
        energy = data.y
        force = data.force
        NUM_ATOM = force.size()[0]
        energy_mean = energy / NUM_ATOM
        ENERGY_MEAN_TOTAL += energy_mean
        force_rms = torch.sqrt(torch.mean(force.square()))
        FORCE_MEAN_TOTAL += force_rms
    ENERGY_MEAN_TOTAL /= len(train_dataset)
    FORCE_MEAN_TOTAL /= len(train_dataset)
    ENERGY_MEAN_TOTAL = ENERGY_MEAN_TOTAL.to(device)
    FORCE_MEAN_TOTAL = FORCE_MEAN_TOTAL.to(device)

    DataLoaderClass = DataLoader
    dataloader_kwargs = {}
    if args.model_3d == "GemNet":
        DataLoaderClass = DataLoaderGemNet
        dataloader_kwargs = {"cutoff": args.GemNet_cutoff, "int_cutoff": args.GemNet_int_cutoff, "triplets_only": args.GemNet_triplets_only}

    train_loader = DataLoaderClass(train_dataset, args.MD17_train_batch_size, shuffle=True, num_workers=args.num_workers, **dataloader_kwargs)
    val_loader = DataLoaderClass(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, **dataloader_kwargs)
    test_loader = DataLoaderClass(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, **dataloader_kwargs)

    node_class = 119
    num_tasks = 1

    # set up model
    model, graph_pred_linear = model_setup()

    if args.input_model_file is not "":
        load_model(model, graph_pred_linear, args.input_model_file)
    model.to(device)
    print(model)
    if graph_pred_linear is not None:
        graph_pred_linear.to(device)
        print(graph_pred_linear)

    criterion = torch.nn.L1Loss()
    model_param_group = [{"params": model.parameters(), "lr": args.lr}]
    if graph_pred_linear is not None:
        model_param_group.append(
            {"params": graph_pred_linear.parameters(), "lr": args.lr}
        )
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

    train_energy_mae_list, train_force_mae_list = [], []
    val_energy_mae_list, val_force_mae_list = [], []
    test_energy_mae_list, test_force_mae_list = [], []
    best_val_force_mae = 1e10
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_acc = train(device, train_loader)
        print("Epoch: {}\nLoss: {}".format(epoch, loss_acc))

        if epoch % args.print_every_epoch == 0:
            if args.eval_train:
                train_energy_mae, train_force_mae = eval(device, train_loader)
            else:
                train_energy_mae = train_force_mae = 0
            val_energy_mae, val_force_mae = eval(device, val_loader)
            if args.eval_test:
                test_energy_mae, test_force_mae = eval(device, test_loader)
            else:
                test_energy_mae = test_force_mae = 0

            train_energy_mae_list.append(train_energy_mae)
            train_force_mae_list.append(train_force_mae)
            val_energy_mae_list.append(val_energy_mae)
            val_force_mae_list.append(val_force_mae)
            test_energy_mae_list.append(test_energy_mae)
            test_force_mae_list.append(test_force_mae)
            print("Energy\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_energy_mae, val_energy_mae, test_energy_mae))
            print("Force\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_force_mae, val_force_mae, test_force_mae))

            if val_force_mae < best_val_force_mae:
                best_val_force_mae = val_force_mae
                best_val_idx = len(train_energy_mae_list) - 1
                if not args.output_model_dir == "":
                    save_model(save_best=True)
        print("Took\t{}\n".format(time.time() - start_time))
    
    save_model(save_best=False)

    if args.eval_test:
        optimal_test_energy, optimal_test_force = test_energy_mae_list[best_val_idx], test_force_mae_list[best_val_idx]
    else:
        optimal_model_weight = os.path.join(args.output_model_dir, "model.pth")
        load_model(model, graph_pred_linear, optimal_model_weight, load_latest=True)
        optimal_test_energy, optimal_test_force = eval(device, test_loader)

    print("best Energy\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_energy_mae_list[best_val_idx], val_energy_mae_list[best_val_idx], optimal_test_energy))
    print("best Force\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_force_mae_list[best_val_idx], val_force_mae_list[best_val_idx], optimal_test_force))
