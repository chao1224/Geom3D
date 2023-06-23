import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool, global_mean_pool
from tqdm import tqdm


from config import args
from Geom3D.models import EGNN, SEGNN, TFN, DimeNet, DimeNetPlusPlus, SchNet, SE3Transformer, SphereNet, PaiNN, GemNet, EquiformerEnergy
from Geom3D.datasets import MoleculeDatasetQM9, MoleculeDataset3DRadius, MoleculeDataset3DFull, MoleculeDatasetOneAtom, MoleculeDataset3DRemoveCenter
from Geom3D.dataloaders import DataLoaderGemNet
from splitters import qm9_random_customized_01, qm9_random_customized_02


def mean_absolute_error(pred, target):
    return np.mean(np.abs(pred - target))


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


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return (x @ Q).float()


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
        graph_pred_linear = torch.nn.Linear(intermediate_dim, num_tasks)

    elif args.model_3d == "DimeNet":
        model = DimeNet(
            node_class=node_class,
            hidden_channels=args.emb_dim,
            out_channels=num_tasks,
            num_blocks=6,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6,
            cutoff=10.0,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
        )
        graph_pred_linear = None

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

    elif args.model_3d == "TFN":
        # This follows the dataset construction in oriGINal implementation
        # https://github.com/FabianFuchsML/se3-transformer-public/blob/master/experiments/qm9/QM9.py#L187
        atom_feature_size = node_class + 1
        model = TFN(
            atom_feature_size=atom_feature_size,
            edge_dim=edge_class,
            num_layers=args.TFN_num_layers,
            num_channels=args.TFN_num_channels,
            num_degrees=args.TFN_num_degrees,
            num_nlayers=args.TFN_num_nlayers,
        )
        latent_dim = args.TFN_num_channels * args.TFN_num_degrees
        graph_pred_linear = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_tasks),
        )

    elif args.model_3d == "SE3_Transformer":
        # This follows the dataset construction in oriGINal implementation
        # https://github.com/FabianFuchsML/se3-transformer-public/blob/master/experiments/qm9/QM9.py#L187
        atom_feature_size = node_class + 1
        model = SE3Transformer(
            atom_feature_size=atom_feature_size,
            edge_dim=edge_class,
            num_layers=args.SE3_Transformer_num_layers,
            num_channels=args.SE3_Transformer_num_channels,
            num_degrees=args.SE3_Transformer_num_degrees,
            num_nlayers=args.SE3_Transformer_num_nlayers,
            div=args.SE3_Transformer_div,
            n_heads=args.SE3_Transformer_n_heads,
        )
        latent_dim = (
            args.SE3_Transformer_num_channels * args.SE3_Transformer_num_degrees
        )
        graph_pred_linear = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_tasks),
        )

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

    elif args.model_3d == "SphereNet":
        model = SphereNet(
            hidden_channels=args.emb_dim,
            out_channels=num_tasks,
            energy_and_force=False,
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

    elif args.model_3d == "SEGNN":
        model = SEGNN(
            node_class,
            num_tasks,
            hidden_features=args.emb_dim,
            N=args.SEGNN_radius,
            lmax_h=args.SEGNN_N,
            lmax_pos=args.SEGNN_lmax_pos,
            norm=args.SEGNN_norm,
            pool=args.SEGNN_pool,
            edge_inference=args.SEGNN_edge_inference
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

    elif args.model_3d == "NequIP":
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
    
    elif args.model_3d == "Equiformer":
        if args.Equiformer_hyperparameter == 0:
            # This follows the hyper in Equiformer_l2
            model = EquiformerEnergy(
                irreps_in=args.Equiformer_irreps_in,
                max_radius=args.Equiformer_radius,
                node_class=node_class,
                number_of_basis=args.Equiformer_num_basis, 
                irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
                irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
                fc_neurons=[64, 64], 
                irreps_feature='512x0e',
                irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
                rescale_degree=False, nonlinear_message=False,
                irreps_mlp_mid='384x0e+192x1e+96x2e',
                norm_layer='layer',
                alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0)
        elif args.Equiformer_hyperparameter == 1:
            # This follows the hyper in Equiformer_nonlinear_bessel_l2_drop00
            model = EquiformerEnergy(
                irreps_in=args.Equiformer_irreps_in,
                max_radius=args.Equiformer_radius,
                node_class=node_class,
                number_of_basis=args.Equiformer_num_basis, 
                irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
                irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
                fc_neurons=[64, 64], basis_type='bessel',
                irreps_feature='512x0e',
                irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
                rescale_degree=False, nonlinear_message=True,
                irreps_mlp_mid='384x0e+192x1e+96x2e',
                norm_layer='layer',
                alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0)
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

    elif "JAE" in model_weight_file:
        if "SchNet_02" in args.output_model_dir:
            tag = "3D_model_02"
        else:
            tag = "3D_model_01"
        print("Loading model from {} ...".format(tag))
        model_weight = torch.load(model_weight_file)

        model.load_state_dict(model_weight[tag])
        
        # pretrained_dict = {}
        # for key, value in model_weight["AE_2D_3D_model"].items():
        #     if not key.startswith("model.{}.".format(tag)):
        #         continue
        #     neo_key = key.replace("model.{}.".format(tag), "")
        #     pretrained_dict[neo_key] = value
        # model.load_state_dict(pretrained_dict)

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
        batch = batch.to(device)

        if args.model_3d == "SchNet":
            molecule_3D_repr = model(batch.x, batch.positions, batch.batch)

        elif args.model_3d == "DimeNet":
            molecule_3D_repr = model(batch.x, batch.positions, batch.batch)

        elif args.model_3d == "DimeNetPlusPlus":
            molecule_3D_repr = model(batch.x, batch.positions, batch.batch)

        elif args.model_3d == "TFN":
            x_one_hot = F.one_hot(batch.x, num_classes=node_class).float()
            x = torch.cat([x_one_hot, batch.x.unsqueeze(1)], dim=1).float()
            edge_attr_one_hot = F.one_hot(batch.edge_attr[:, 0], num_classes=edge_class)
            node_3D_repr = model(
                x=x,
                positions=batch.positions,
                edge_index=batch.edge_index,
                edge_feat=edge_attr_one_hot,
            )
            molecule_3D_repr = global_max_pool(node_3D_repr, batch.batch)

        elif args.model_3d == "SE3_Transformer":
            x_one_hot = F.one_hot(batch.x, num_classes=node_class).float()
            x = torch.cat([x_one_hot, batch.x.unsqueeze(1)], dim=1).float()
            edge_attr_one_hot = F.one_hot(batch.edge_attr[:, 0], num_classes=edge_class)
            node_3D_repr = model(
                x=x,
                positions=batch.positions,
                edge_index=batch.edge_index,
                edge_feat=edge_attr_one_hot,
            )
            molecule_3D_repr = global_max_pool(node_3D_repr, batch.batch)

        elif args.model_3d == "EGNN":
            x_one_hot = F.one_hot(batch.x, num_classes=node_class)
            x = preprocess_input(
                x_one_hot,
                batch.x,
                charge_power=args.EGNN_charge_power,
                charge_scale=node_class,
            )
            node_3D_repr = model(
                x=x,
                positions=batch.positions,
                edge_index=batch.full_edge_index,
                edge_attr=None,
            )
            molecule_3D_repr = global_mean_pool(node_3D_repr, batch.batch)

        elif args.model_3d == "SphereNet":
            molecule_3D_repr = model(batch.x, batch.positions, batch.batch)

        elif args.model_3d == "SEGNN":
            molecule_3D_repr = model(batch)

        elif args.model_3d == "PaiNN":
            molecule_3D_repr = model(batch.x, batch.positions, batch.radius_edge_index, batch.batch)

        elif args.model_3d == "GemNet":
            molecule_3D_repr = model(batch.x, batch.positions, batch)

        elif args.model_3d in ["NequIP", "Allegro"]:
            # # TODO: will check how edge_index is constructured.
            data = {
                "edge_index": batch.radius_edge_index,
                "pos": batch.positions,
                "atom_types": batch.x,
                "batch": batch.batch,
            }
            out = model(data)
            molecule_3D_repr = out["total_energy"].squeeze()

        elif args.model_3d == "Equiformer":
            molecule_3D_repr = model(node_atom=batch.x, pos=batch.positions, batch=batch.batch)

        if graph_pred_linear is not None:
            pred = graph_pred_linear(molecule_3D_repr).squeeze()
        else:
            pred = molecule_3D_repr.squeeze()

        B = pred.size()[0]
        y = batch.y.view(B, -1)[:, task_id]
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
    if graph_pred_linear is not None:
        graph_pred_linear.eval()
    y_true = []
    y_scores = []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for batch in L:
        batch = batch.to(device)

        if args.model_3d == "SchNet":
            molecule_3D_repr = model(batch.x, batch.positions, batch.batch)

        elif args.model_3d == "DimeNet":
            molecule_3D_repr = model(batch.x, batch.positions, batch.batch)

        elif args.model_3d == "DimeNetPlusPlus":
            molecule_3D_repr = model(batch.x, batch.positions, batch.batch)

        elif args.model_3d == "TFN":
            x_one_hot = F.one_hot(batch.x, num_classes=node_class).float()
            x = torch.cat([x_one_hot, batch.x.unsqueeze(1)], dim=1).float()
            edge_attr_one_hot = F.one_hot(batch.edge_attr[:, 0], num_classes=edge_class)
            node_3D_repr = model(
                x=x,
                positions=batch.positions,
                edge_index=batch.edge_index,
                edge_feat=edge_attr_one_hot,
            )
            molecule_3D_repr = global_max_pool(node_3D_repr, batch.batch)

        elif args.model_3d == "SE3_Transformer":
            x_one_hot = F.one_hot(batch.x, num_classes=node_class).float()
            x = torch.cat([x_one_hot, batch.x.unsqueeze(1)], dim=1).float()
            edge_attr_one_hot = F.one_hot(batch.edge_attr[:, 0], num_classes=edge_class)
            node_3D_repr = model(
                x=x,
                positions=batch.positions,
                edge_index=batch.edge_index,
                edge_feat=edge_attr_one_hot,
            )
            molecule_3D_repr = global_max_pool(node_3D_repr, batch.batch)

        elif args.model_3d == "EGNN":
            x_one_hot = F.one_hot(batch.x, num_classes=node_class).float()
            x = preprocess_input(
                x_one_hot,
                batch.x,
                charge_power=args.EGNN_charge_power,
                charge_scale=node_class,
            )
            node_3D_repr = model(
                x=x,
                positions=batch.positions,
                edge_index=batch.full_edge_index,
                edge_attr=None,
            )
            molecule_3D_repr = global_mean_pool(node_3D_repr, batch.batch)

        elif args.model_3d == "SphereNet":
            molecule_3D_repr = model(batch.x, batch.positions, batch.batch)

        elif args.model_3d == "SEGNN":
            molecule_3D_repr = model(batch)

        elif args.model_3d == "PaiNN":
            molecule_3D_repr = model(batch.x, batch.positions, batch.radius_edge_index, batch.batch)

        elif args.model_3d == "GemNet":
            molecule_3D_repr = model(batch.x, batch.positions, batch)

        elif args.model_3d in ["NequIP", "Allegro"]:
            data = {
                "edge_index": batch.radius_edge_index,
                "pos": batch.positions,
                "atom_types": batch.x,
                "batch": batch.batch,
            }
            out = model(data)
            molecule_3D_repr = out["total_energy"].squeeze()

        elif args.model_3d == "Equiformer":
            molecule_3D_repr = model(node_atom=batch.x, pos=batch.positions, batch=batch.batch)

        if graph_pred_linear is not None:
            pred = graph_pred_linear(molecule_3D_repr).squeeze()
        else:
            pred = molecule_3D_repr.squeeze()

        B = pred.size()[0]
        y = batch.y.view(B, -1)[:, task_id]
        # denormalize
        pred = pred * TRAIN_std + TRAIN_mean

        y_true.append(y)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    mae = mean_absolute_error(y_scores, y_true)
    return mae, y_true, y_scores


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

    rotation_transform = None
    if args.use_rotation_transform:
        rotation_transform = RandomRotation()

    num_tasks = 1
    assert args.dataset == "qm9"
    data_root = "../data/molecule_datasets/{}".format(args.dataset)
    dataset = MoleculeDatasetQM9(
        data_root,
        dataset=args.dataset,
        task=args.task,
        rotation_transform=rotation_transform,
    )
    task_id = dataset.task_id

    ##### Dataset wrapper for graph with radius. #####
    if args.model_3d == "EGNN":
        data_root = "../data/molecule_datasets/{}_full".format(args.dataset)
        dataset = MoleculeDataset3DFull(
            data_root,
            preprcessed_dataset=dataset
        )
    elif args.model_3d == "SEGNN":
        data_root = "../data/molecule_datasets/{}_{}".format(args.dataset, args.SEGNN_radius)
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.SEGNN_radius
        )
    elif args.model_3d == "PaiNN":
        data_root = "../data/molecule_datasets/{}_{}".format(args.dataset, args.PaiNN_radius_cutoff)
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.PaiNN_radius_cutoff
        )
    elif args.model_3d in ["NequIP", "Allegro"]:
        # Will update this
        data_root = "../data/molecule_datasets/{}_{}".format(args.dataset, args.NequIP_radius_cutoff)
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.NequIP_radius_cutoff
        )
    
    if args.only_one_atom_type:
        data_root = "{}_one_atom".format(dataset.root)
        print("neo root", data_root)
        dataset = MoleculeDatasetOneAtom(
            data_root,
            preprcessed_dataset=dataset
        )

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

    DataLoaderClass = DataLoader
    dataloader_kwargs = {}
    if args.model_3d == "GemNet":
        DataLoaderClass = DataLoaderGemNet
        dataloader_kwargs = {"cutoff": args.GemNet_cutoff, "int_cutoff": args.GemNet_int_cutoff, "triplets_only": args.GemNet_triplets_only}

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

    # set up model
    if args.JK == "concat":
        intermediate_dim = (args.num_layer + 1) * args.emb_dim
    else:
        intermediate_dim = args.emb_dim

    node_class, edge_class = 119, 5
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
