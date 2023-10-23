import argparse
from email.policy import default

parser = argparse.ArgumentParser()

# about seed and basic info
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--data_root")

parser.add_argument(
    "--model_3d",
    type=str,
    default="SchNet",
    choices=[
        "SchNet",
        "DimeNet",
        "DimeNetPlusPlus",
        "TFN",
        "SE3_Transformer",
        "EGNN",
        "SphereNet",
        "SEGNN",
        "PaiNN",
        "GemNet",
        "NequIP",
        "Allegro",
        "Equiformer",
        "GVP",
        "GearNet",
        "ProNet",
        "CDConv",
    ],
)
parser.add_argument(
    "--model_2d",
    type=str,
    default="GIN",
    choices=[
        "GIN",
        "SchNet",
        "DimeNet",
        "DimeNetPlusPlus",
        "TFN",
        "SE3_Transformer",
        "EGNN",
        "SphereNet",
        "SEGNN",
        "PaiNN",
        "GemNet",
        "NequIP",
        "Allegro",
    ],
)

# about dataset and dataloader
parser.add_argument("--dataset", type=str, default="qm9")
parser.add_argument("--task", type=str, default="alpha")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--only_one_atom_type", dest="only_one_atom_type", action="store_true")
parser.set_defaults(only_one_atom_type=False)

# for MD17
# The default hyper from here: https://github.com/divelab/DIG_storage/tree/main/3dgraph/md17
parser.add_argument("--md17_energy_coeff", type=float, default=0.05)
parser.add_argument("--md17_force_coeff", type=float, default=0.95)
parser.add_argument("--energy_force_with_normalization", dest="energy_force_with_normalization", action="store_true")
parser.add_argument("--energy_force_no_normalization", dest="energy_force_with_normalization", action="store_false")
parser.set_defaults(energy_force_with_normalization=False)

# for rMD17
parser.add_argument("--rMD17_split_id", type=str, default="01", choices=["01", "02", "03", "04", "05"])

# for MatBench
parser.add_argument("--MatBench_cutoff", type=float, default=5)
parser.add_argument("--MatBench_max_neighbours", type=int, default=None)
parser.add_argument("--MatBench_split_mode", type=str, default="standard", choices=["ablation", "standard"])
# for QMOF
parser.add_argument("--QMOF_cutoff", type=float, default=5)
parser.add_argument("--QMOF_max_neighbours", type=int, default=None)
# for MatBench and QMOF
parser.add_argument("--periodic_data_augmentation", type=str, default="image_gathered", choices=["image_gathered", "image_expanded"])

# for COLL
# The default hyper from here: https://github.com/divelab/DIG_storage/tree/main/3dgraph/md17
parser.add_argument("--coll_energy_coeff", type=float, default=0.05)
parser.add_argument("--coll_force_coeff", type=float, default=0.95)

# for LBA
# The default hyper from here: https://github.com/drorlab/atom3d/blob/master/examples/lep/enn/utils.py#L37-L43
parser.add_argument("--LBA_year", type=int, default=2020)
parser.add_argument("--LBA_dist", type=float, default=6)
parser.add_argument("--LBA_maxnum", type=int, default=500)
parser.add_argument("--LBA_use_complex", dest="LBA_use_complex", action="store_true")
parser.add_argument("--LBA_no_complex", dest="LBA_use_complex", action="store_false")
parser.set_defaults(LBA_use_complex=False)

# for LEP
# The default hyper from here: https://github.com/drorlab/atom3d/blob/master/examples/lep/enn/utils.py#L48-L55
parser.add_argument("--LEP_dist", type=float, default=6)
parser.add_argument("--LEP_maxnum", type=float, default=400)
parser.add_argument("--LEP_droph", dest="LEP_droph", action="store_true")
parser.add_argument("--LEP_useh", dest="LEP_droph", action="store_false")
parser.set_defaults(LEP_droph=False)

# for GeneOntology
parser.add_argument("--GO_level", default="mf", choices=["mf", "bp", "cc"])

# for MoleculeNet
parser.add_argument("--moleculenet_num_conformers", type=int, default=10)

# about training strategies
parser.add_argument("--split", type=str, default="customized_01",
                    choices=["customized_01", "customized_02", "random", "atom3d_lba_split30"])
parser.add_argument("--MD17_train_batch_size", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lr_scale", type=float, default=1)
parser.add_argument("--decay", type=float, default=0)
parser.add_argument("--print_every_epoch", type=int, default=1)
parser.add_argument("--loss", type=str, default="mae", choices=["mse", "mae"])
parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"])
parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")
parser.add_argument("--lr_decay_factor", type=float, default=0.5)
parser.add_argument("--lr_decay_step_size", type=int, default=100)
parser.add_argument("--lr_decay_patience", type=int, default=50)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--StepLRCustomized_scheduler", type=int, nargs='+', default=[150])
parser.add_argument("--verbose", dest="verbose", action="store_true")
parser.add_argument("--no_verbose", dest="verbose", action="store_false")
parser.set_defaults(verbose=False)
parser.add_argument("--use_rotation_transform", dest="use_rotation_transform", action="store_true")
parser.add_argument("--no_rotation_transform", dest="use_rotation_transform", action="store_false")
parser.set_defaults(use_rotation_transform=False)

# for SchNet
parser.add_argument("--SchNet_num_filters", type=int, default=128)
parser.add_argument("--SchNet_num_interactions", type=int, default=6)
parser.add_argument("--SchNet_num_gaussians", type=int, default=51)
parser.add_argument("--SchNet_cutoff", type=float, default=10)
parser.add_argument("--SchNet_readout", type=str, default="mean", choices=["mean", "add"])
parser.add_argument("--SchNet_gamma", type=float, default=None)

# for TFN
parser.add_argument("--TFN_num_layers", type=int, default=7)
parser.add_argument("--TFN_num_channels", type=int, default=32)
parser.add_argument("--TFN_num_degrees", type=int, default=4)
parser.add_argument("--TFN_num_nlayers", type=int, default=1)

# for SE(3)-Transformer
parser.add_argument("--SE3_Transformer_num_layers", type=int, default=7)
parser.add_argument("--SE3_Transformer_num_channels", type=int, default=32)
parser.add_argument("--SE3_Transformer_num_degrees", type=int, default=4)
parser.add_argument("--SE3_Transformer_num_nlayers", type=int, default=1)
parser.add_argument("--SE3_Transformer_div", type=int, default=2)
parser.add_argument("--SE3_Transformer_n_heads", type=int, default=8)

# for EGNN
parser.add_argument("--EGNN_n_layers", type=int, default=7)
parser.add_argument("--EGNN_attention", type=int, default=1)
parser.add_argument("--EGNN_node_attr", type=int, default=0)
parser.add_argument("--EGNN_positions_weight", type=float, default=1.0)
parser.add_argument("--EGNN_charge_power", type=int, default=2)
parser.add_argument("--EGNN_radius_cutoff", type=float, default=5.0)

# for DimeNet++
parser.add_argument("--DimeNetPlusPlus_num_blocks", type=int, default=4)
parser.add_argument("--DimeNetPlusPlus_int_emb_size", type=int, default=64)
parser.add_argument("--DimeNetPlusPlus_basis_emb_size", type=int, default=8)
parser.add_argument("--DimeNetPlusPlus_out_emb_channels", type=int, default=128)
parser.add_argument("--DimeNetPlusPlus_num_spherical", type=int, default=7)
parser.add_argument("--DimeNetPlusPlus_num_radial", type=int, default=6)
parser.add_argument("--DimeNetPlusPlus_cutoff", type=float, default=5.0)
parser.add_argument("--DimeNetPlusPlus_envelope_exponent", type=int, default=5)
parser.add_argument("--DimeNetPlusPlus_num_before_skip", type=int, default=1)
parser.add_argument("--DimeNetPlusPlus_num_after_skip", type=int, default=2)
parser.add_argument("--DimeNetPlusPlus_num_output_layers", type=int, default=3)
parser.add_argument("--DimeNetPlusPlus_readout", type=str, default="add", choices=["mean", "add"])

# for SphereNet
parser.add_argument("--SphereNet_cutoff", type=float, default=5.0)
parser.add_argument("--SphereNet_num_layers", type=int, default=4)
parser.add_argument("--SphereNet_int_emb_size", type=int, default=64)
parser.add_argument("--SphereNet_basis_emb_size_dist", type=int, default=8)
parser.add_argument("--SphereNet_basis_emb_size_angle", type=int, default=8)
parser.add_argument("--SphereNet_basis_emb_size_torsion", type=int, default=8)
parser.add_argument("--SphereNet_out_emb_channels", type=int, default=256)
parser.add_argument("--SphereNet_num_spherical", type=int, default=3)
parser.add_argument("--SphereNet_num_radial", type=int, default=6)
parser.add_argument("--SphereNet_envelope_exponent", type=int, default=5)
parser.add_argument("--SphereNet_num_before_skip", type=int, default=1)
parser.add_argument("--SphereNet_num_after_skip", type=int, default=2)
parser.add_argument("--SphereNet_num_output_layers", type=int, default=3)

# for SEGNN
parser.add_argument("--SEGNN_radius", type=float, default=2)
parser.add_argument("--SEGNN_N", type=int, default=7)
parser.add_argument("--SEGNN_lmax_h", type=int, default=2)
parser.add_argument("--SEGNN_lmax_pos", type=int, default=3)
parser.add_argument("--SEGNN_norm", type=str, default="instance")
parser.add_argument("--SEGNN_pool", type=str, default="avg")
parser.add_argument("--SEGNN_edge_inference", type=int, default=0)

# for PaiNN
parser.add_argument("--PaiNN_radius_cutoff", type=float, default=5.0)
parser.add_argument("--PaiNN_n_interactions", type=int, default=3)
parser.add_argument("--PaiNN_n_rbf", type=int, default=20)
parser.add_argument("--PaiNN_readout", type=str, default="add", choices=["mean", "add"])
parser.add_argument("--PaiNN_gamma", type=float, default=None)

# for GemNet
parser.add_argument("--GemNet_num_spherical", type=int, default=7)
parser.add_argument("--GemNet_num_radial", type=int, default=6)
parser.add_argument("--GemNet_num_blocks", type=int, default=4)
parser.add_argument("--GemNet_emb_size_trip", type=int, default=64)
parser.add_argument("--GemNet_emb_size_quad", type=int, default=32)
parser.add_argument("--GemNet_emb_size_rbf", type=int, default=16)
parser.add_argument("--GemNet_emb_size_cbf", type=int, default=16)
parser.add_argument("--GemNet_emb_size_sbf", type=int, default=32)
parser.add_argument("--GemNet_emb_size_bil_trip", type=int, default=64)
parser.add_argument("--GemNet_emb_size_bil_quad", type=int, default=32)
parser.add_argument("--GemNet_num_before_skip", type=int, default=1)
parser.add_argument("--GemNet_num_after_skip", type=int, default=1)
parser.add_argument("--GemNet_num_concat", type=int, default=1)
parser.add_argument("--GemNet_num_atom", type=int, default=2)
parser.add_argument("--GemNet_cutoff", type=float, default=5.)
parser.add_argument("--GemNet_int_cutoff", type=float, default=10.)
parser.add_argument("--GemNet_triplets_only", type=int, default=1, choices=[0, 1])
parser.add_argument("--GemNet_direct_forces", type=int, default=0, choices=[0, 1])
parser.add_argument("--GemNet_envelope_exponent", type=int, default=5)
parser.add_argument("--GemNet_extensive", type=int, default=1, choices=[0, 1])
parser.add_argument("--GemNet_forces_coupled", type=int, default=0, choices=[0, 1])
parser.add_argument("--GemNet_output_init", type=str, default="HeOrthogonal")
parser.add_argument("--GemNet_activation", type=str, default="swish")
parser.add_argument("--GemNet_scale_file", type=str, default="scaling_factors.json")

# for NequIP and Allegro
parser.add_argument("--NequIP_radius_cutoff", type=float, default=4.)

# For Equiformer
parser.add_argument("--Equiformer_radius", type=float, default=5)
parser.add_argument("--Equiformer_irreps_in", type=str, default="5x0e")
parser.add_argument("--Equiformer_num_basis", type=int, default=128)
parser.add_argument("--Equiformer_hyperparameter", type=int, default=0)

# for GVP
parser.add_argument("--num_positional_embeddings", type=int, default=16)
parser.add_argument("--top_k", type=int, default=30)
parser.add_argument("--num_rbf", type=int, default=16)

# for ProNet
parser.add_argument("--ProNet_level", type=str, default="aminoacid", choices=["aminoacid", "backbone", "allatom"])
parser.add_argument("--ProNet_dropout", type=float, default=0.3)

# for CDConv
parser.add_argument("--CDConv_radius", type=float, default=4)
parser.add_argument("--CDConv_kernel_size", type=int, default=21)
parser.add_argument("--CDConv_kernel_channels", type=int, nargs="+", default=[24])
parser.add_argument("--CDConv_geometric_raddi_coeff", type=int, nargs="+", default=[2, 3, 4, 5])
parser.add_argument("--CDConv_channels", type=int, nargs="+", default=[256, 512, 1024, 2048])
parser.add_argument("--CDConv_base_width", type=int, default=64)

# for GearNet
parser.add_argument("--num_relation", type=int, default=7)
parser.add_argument("--GearNet_readout", type=str, default="sum")
parser.add_argument("--GearNet_dropout", type=float, default=0)
parser.add_argument("--GearNet_edge_input_dim", type=int)
parser.add_argument("--GearNet_num_angle_bin", type=int)

# data augmentation tricks, see appendix E in the paper (https://openreview.net/pdf?id=9X-hgLDLYkQ)
parser.add_argument('--mask', action='store_true')
parser.add_argument('--noise', action='store_true')
parser.add_argument('--deform', action='store_true')
parser.add_argument('--data_augment_eachlayer', action='store_true')
parser.add_argument('--euler_noise', action='store_true')
parser.add_argument('--mask_aatype', type=float, default=0.1)

######################### for Charge Prediction SSL #########################
parser.add_argument("--charge_masking_ratio", type=float, default=0.3)

######################### for Distance Perturbation SSL #########################
parser.add_argument("--distance_sample_ratio", type=float, default=1)

######################### for Torsion Angle Perturbation SSL #########################
parser.add_argument("--torsion_angle_sample_ratio", type=float, default=0.001)

######################### for Position Perturbation SSL #########################
parser.add_argument("--PP_mu", type=float, default=0)
parser.add_argument("--PP_sigma", type=float, default=0.3)


######################### for GraphMVP SSL #########################
### for 2D GNN
parser.add_argument("--gnn_type", type=str, default="GIN")
parser.add_argument("--num_layer", type=int, default=5)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument("--dropout_ratio", type=float, default=0.5)
parser.add_argument("--graph_pooling", type=str, default="mean")
parser.add_argument("--JK", type=str, default="last")
parser.add_argument("--gnn_2d_lr_scale", type=float, default=1)


######################### for 3D-EMGP #########################
parser.add_argument('--EMGP_noise_type', type=str, default="gaussian", choices=["riemann", "gaussian"])
parser.add_argument('--EMGP_equivariant_noise_scale_coeff', type=float, default=1)
parser.add_argument('--EMGP_invariant_noise_scale_coeff', type=float, default=0.2)

######################### for GeoSSL #########################
parser.add_argument("--GeoSSL_mu", type=float, default=0)
parser.add_argument("--GeoSSL_sigma", type=float, default=0.3)
parser.add_argument("--GeoSSL_atom_masking_ratio", type=float, default=0.3)
parser.add_argument("--GeoSSL_option", type=str, default="EBM_NCE", choices=["EBM_NCE", "InfoNCE", "RR", "EBM_SM", "DDM"])

parser.add_argument("--GeoSSL_sigma_beGIN", type=float, default=10)
parser.add_argument("--GeoSSL_sigma_end", type=float, default=0.01)
parser.add_argument("--GeoSSL_num_noise_level", type=int, default=50)
parser.add_argument("--GeoSSL_noise_type", type=str, default="symmetry", choices=["symmetry", "random"])
parser.add_argument("--GeoSSL_anneal_power", type=float, default=2)

######################### for GraphMVP SSL #########################
### for 3D GNN
parser.add_argument("--gnn_3d_lr_scale", type=float, default=1)

### for masking
parser.add_argument("--SSL_masking_ratio", type=float, default=0.15)

### for 2D-3D Contrastive SSL
parser.add_argument("--CL_neg_samples", type=int, default=1)
parser.add_argument("--CL_similarity_metric", type=str, default="InfoNCE_dot_prod",
                    choices=["InfoNCE_dot_prod", "EBM_dot_prod", "EBM_node_dot_prod"])
parser.add_argument("--T", type=float, default=0.1)
parser.add_argument("--normalize", dest="normalize", action="store_true")
parser.add_argument("--no_normalize", dest="normalize", action="store_false")
parser.add_argument("--alpha_1", type=float, default=1)

### for 2D-3D Generative SSL
parser.add_argument("--GraphMVP_AE_model", type=str, default="VAE")
parser.add_argument("--detach_target", dest="detach_target", action="store_true")
parser.add_argument("--no_detach_target", dest="detach_target", action="store_false")
parser.set_defaults(detach_target=True)
parser.add_argument("--AE_loss", type=str, default="l2", choices=["l1", "l2", "cosine"])
parser.add_argument("--beta", type=float, default=1)
parser.add_argument("--alpha_2", type=float, default=1)


### for MoleculeSDE
parser.add_argument("--SDE_type_2Dto3D", type=str, default="VE")
parser.add_argument("--SDE_type_3Dto2D", type=str, default="VE")
parser.add_argument("--SDE_2Dto3D_model", type=str, default="SDEModel2Dto3D_01")
parser.add_argument("--SDE_3Dto2D_model", type=str, default="SDEModel3Dto2D_node_adj_dense")
parser.add_argument("--SDE_coeff_contrastive", type=float, default=1)
parser.add_argument("--SDE_coeff_contrastive_skip_epochs", type=int, default=0)
parser.add_argument("--SDE_coeff_generative_2Dto3D", type=float, default=1)
parser.add_argument("--SDE_coeff_generative_3Dto2D", type=float, default=1)
parser.add_argument("--SDE_coeff_2D_masking", type=float, default=0)
parser.add_argument("--SDE_coeff_2D_masking_ratio", type=float, default=0)
parser.add_argument("--SDE_coeff_3D_masking", type=float, default=0)
# This is only for 3D to 2D
parser.add_argument("--use_extend_graph", dest="use_extend_graph", action="store_true")
parser.add_argument("--no_extend_graph", dest="use_extend_graph", action="store_false")
parser.set_defaults(use_extend_graph=True)
# This is only for 2D to 3D
parser.add_argument("--noise_on_one_hot", dest="noise_on_one_hot", action="store_true")
parser.add_argument("--no_noise_on_one_hot", dest="noise_on_one_hot", action="store_false")
parser.set_defaults(noise_on_one_hot=True)
parser.add_argument("--SDE_anneal_power", type=float, default=0)
# This is only for 2D to 3D to MoleculeNet property
parser.add_argument("--molecule_property_SDE_2D", type=float, default=1)

### for MoleculeSDE inference
parser.add_argument('--generator', type=str, help='type of generator [MultiScaleLD, PC]', default='MultiScaleLD')
parser.add_argument('--eval_epoch', type=int, default=None, help='evaluation epoch')
parser.add_argument('--start', type=int, default=0, help='start idx of test generation')
parser.add_argument('--end', type=int, default=100, help='end idx of test generation')
parser.add_argument('--num_repeat_SDE_inference', type=int, default=10, help='number of conformers')
parser.add_argument('--num_repeat_SDE_predict', type=int, default=1, help='number of conformers for prediction')
parser.add_argument("--min_sigma", type=float, default=0.0)
parser.add_argument('--steps_pos', type=int, default=100, help='MCMC')
parser.add_argument("--step_lr_pos", type=float, default=0.0000015)
parser.add_argument("--clip", type=float, default=1000)
parser.add_argument("--num_diffusion_timesteps_2Dto3D_inference", type=int, default=20)
parser.add_argument("--num_diffusion_timesteps_3Dto2D_inference", type=int, default=20)
parser.add_argument("--visualization_timesteps_interval", type=int, default=20)
parser.add_argument("--data_path_2D_SDE", type=str, default="")


### for 2D SSL
parser.add_argument("--GraphMVP_2D_mode", type=str, default="AM", choices=["AM", "CP"])
parser.add_argument("--alpha_3", type=float, default=1)
### for AttributeMask
parser.add_argument("--mask_rate", type=float, default=0.15)
### for ContextPred
parser.add_argument("--csize", type=int, default=3)
parser.add_argument("--contextpred_neg_samples", type=int, default=1)
#######################################################################

parser.add_argument("--corrector_steps", type=int, default=1)

##### about if we would print out eval metric for training data
parser.add_argument("--eval_train", dest="eval_train", action="store_true")
parser.add_argument("--no_eval_train", dest="eval_train", action="store_false")
parser.set_defaults(eval_train=False)
##### about if we would print out eval metric for training data
##### this is only for COLL
parser.add_argument("--eval_test", dest="eval_test", action="store_true")
parser.add_argument("--no_eval_test", dest="eval_test", action="store_false")
parser.set_defaults(eval_test=True)

parser.add_argument("--input_data_dir", type=str, default="")

# about loading and saving
parser.add_argument("--input_model_file", type=str, default="")
parser.add_argument("--output_model_dir", type=str, default="")

parser.add_argument("--threshold", type=float, default=0)

parser.add_argument("--use_predictor", dest="use_predictor", action="store_true")
parser.add_argument("--no_predictor", dest="use_predictor", action="store_false")
parser.set_defaults(use_predictor=True)
parser.add_argument('--FF', action='store_true', help='only for rdkit')

args = parser.parse_args()
print("arguments\t", args)
