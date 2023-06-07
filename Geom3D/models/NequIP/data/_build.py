import inspect
from importlib import import_module

from Geom3D.models.NequIP import data
from Geom3D.models.NequIP.data.transforms import TypeMapper
from Geom3D.models.NequIP.data import AtomicDataset, register_fields
from Geom3D.models.NequIP.utils import instantiate, get_w_prefix


def dataset_from_config(config, prefix: str = "dataset") -> AtomicDataset:
    """initialize database based on a config instance

    It needs dataset type name (case insensitive),
    and all the parameters needed in the constructor.

    Examples see tests/data/test_dataset.py TestFromConfig
    and tests/datasets/test_simplest.py

    Args:

    config (dict, NequIP.utils.Config): dict/object that store all the parameters
    prefix (str): Optional. The prefix of all dataset parameters

    Return:

    dataset (NequIP.data.AtomicDataset)
    """

    config_dataset = config.get(prefix, None) # npz
    if config_dataset is None:
        raise KeyError(f"Dataset with prefix `{prefix}` isn't present in this config!")

    if inspect.isclass(config_dataset):
        # user define class
        class_name = config_dataset
    else:
        try:
            module_name = ".".join(config_dataset.split(".")[:-1])
            class_name = ".".join(config_dataset.split(".")[-1:])
            class_name = getattr(import_module(module_name), class_name)
        except Exception:
            # ^ TODO: don't catch all Exception
            # default class defined in NequIP.data or NequIP.dataset
            dataset_name = config_dataset.lower()

            class_name = None
            for k, v in inspect.getmembers(data, inspect.isclass):
                if k.endswith("Dataset"):
                    if k.lower() == dataset_name:
                        class_name = v
                    if k[:-7].lower() == dataset_name:
                        class_name = v
                elif k.lower() == dataset_name:
                    class_name = v

    if class_name is None:
        raise NameError(f"dataset type {dataset_name} does not exists")

    # if dataset r_max is not found, use the universal r_max
    eff_key = "extra_fixed_fields"
    prefixed_eff_key = f"{prefix}_{eff_key}"
    config[prefixed_eff_key] = get_w_prefix(
        eff_key, {}, prefix=prefix, arg_dicts=config
    )
    config[prefixed_eff_key]["r_max"] = get_w_prefix(
        "r_max",
        prefix=prefix,
        arg_dicts=[config[prefixed_eff_key], config],
    )

    # print("config in dataset", config)  # config in dataset {'_jit_bailout_depth': 2, '_jit_fusion_strategy': [('DYNAMIC', 3)], 'root': 'results/toluene', 'run_name': 'example-run-toluene', 'append': True, 'wandb': False, 'wandb_project': 'toluene-example', 'model_builders': ['SimpleIrrepsConfig', 'EnergyModel', 'PerSpeciesRescale', 'ForceOutput', 'RescaleEnergyEtc'], 'dataset_statistics_stride': 1, 'chemical_symbols': ['H', 'C'], 'default_dtype': 'float32', 'allow_tf32': False, 'verbose': 'info', 'model_debug_mode': False, 'equivariance_test': False, 'grad_anomaly_mode': False, 'dataset': 'npz', 'dataset_url': 'http://quantum-machine.org/gdml/data/npz/toluene_ccsd_t.zip', 'dataset_file_name': './benchmark_data/toluene_ccsd_t-train.npz', 'seed': 123, 'dataset_seed': 456, 'r_max': 4.0, 'num_layers': 4, 'l_max': 1, 'parity': True, 'num_features': 32, 'nonlinearity_type': 'gate', 'nonlinearity_scalars': {'e': 'silu', 'o': 'tanh'}, 'nonlinearity_gates': {'e': 'silu', 'o': 'tanh'}, 'num_basis': 8, 'BesselBasis_trainable': True, 'PolynomialCutoff_p': 6, 'invariant_layers': 2, 'invariant_neurons': 64, 'avg_num_neighbors': 'auto', 'use_sc': True, 'key_mapping': {'z': 'atomic_numbers', 'E': 'total_energy', 'F': 'forces', 'R': 'pos'}, 'npz_fixed_field_keys': ['atomic_numbers'], 'log_batch_freq': 10, 'log_epoch_freq': 1, 'save_checkpoint_freq': -1, 'save_ema_checkpoint_freq': -1, 'n_train': 100, 'n_val': 50, 'learning_rate': 0.005, 'batch_size': 5, 'validation_batch_size': 10, 'max_epochs': 100000, 'train_val_split': 'random', 'shuffle': True, 'metrics_key': 'validation_loss', 'use_ema': True, 'ema_decay': 0.99, 'ema_use_num_updates': True, 'report_init_validation': True, 'early_stopping_patiences': {'validation_loss': 50}, 'early_stopping_lower_bounds': {'LR': 1e-05}, 'loss_coeffs': {'forces': 1, 'total_energy': [1, 'PerAtomMSELoss']}, 'metrics_components': [['forces', 'mae'], ['forces', 'rmse'], ['forces', 'mae', {'PerSpecies': True, 'report_per_component': False}], ['forces', 'rmse', {'PerSpecies': True, 'report_per_component': False}], ['total_energy', 'mae'], ['total_energy', 'mae', {'PerAtom': True}]], 'optimizer_name': 'Adam', 'optimizer_amsgrad': False, 'lr_scheduler_name': 'ReduceLROnPlateau', 'lr_scheduler_patience': 100, 'lr_scheduler_factor': 0.5, 'per_species_rescale_shifts_trainable': False, 'per_species_rescale_scales_trainable': False, 'per_species_rescale_shifts': 'dataset_per_atom_total_energy_mean', 'per_species_rescale_scales': 'dataset_forces_rms', 'dataset_extra_fixed_fields': {'r_max': 4.0}}
    # print("prefix", prefix) # prefix dataset
    # Build a TypeMapper from the config
    type_mapper, _ = instantiate(TypeMapper, prefix=prefix, optional_args=config)

    # Register fields:
    # This might reregister fields, but that's OK:
    instantiate(register_fields, all_args=config)

    instance, _ = instantiate(
        class_name,
        prefix=prefix,
        positional_args={"type_mapper": type_mapper},
        optional_args=config,
    )

    return instance
