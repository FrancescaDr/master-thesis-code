import argparse
import torch
import numpy as np
from datetime import datetime

def set_hparams_(seed, parser):
    """
    Set hyper-parameters to (i) default values if `seed=0`, (ii) random
    values if `seed != 0`, or (iii) values fixed by user for those
    hyper-parameters specified in the JSON string `hparams`.
    """
    default = (seed == 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    hparams = {
        "vae_n_hidden": 128 if default else
        int(np.random.choice([128, 256, 512])),
        #"vae_n_latent": 20 if default else
        #"vae_n_latent": int(np.random.choice([20, 30, 40])),
        "vae_n_layers": 2 if default else
        int(np.random.choice([1, 2, 3, 4, 5])),
        "vae_dropout": 0.1 if default else
        int(np.random.uniform(low=0.1, high=0.4)),
        "batch_size": 128 if default else
        int(np.random.choice([64, 128, 256, 512])),
    }
    return hparams

def parse_arguments_train(seed=0):
    """
    Read arguments if this script is called from a terminal.
    """
    parser = argparse.ArgumentParser(
        usage="python main.py --func {function_name} [--option]",
        description='Model arguments')
    # dataset arguments
    parser.add_argument('--func', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--cat_cov_keys', type=str, nargs='+', default=None)
    parser.add_argument('--test_split', type=int, default=None) # set aside data for testing, percentage what amount of each condition combination should be left out
    parser.add_argument('--ood_condition', type=str, default=None)
    parser.add_argument('--train_size', type=float, default=0.8)
    # Defined 'subset_data' variables:
    # 'PG': includes only specified population group
    # 'PG_AD' excludes all perturbed individuals from specified population group
    parser.add_argument('--subset_data', type=str, default=None)
    parser.add_argument('--subset_PG', type=str, default=None)
    parser.add_argument('--subset_AD', type=str, default=None)
    parser.add_argument('--n_hvg', type=int, default=5000)
    parser.add_argument('--n_latent', type=int, default=20)

    # ComPert arguments (see set_hparams_() in compert.model.ComPert)
    parser.add_argument('--seed', type=int, default=seed)
    hparams = set_hparams_(seed, parser)
    parser.add_argument('--hparams', type=dict, default=hparams)

    # training arguments
    parser.add_argument('--max_epochs', type=int, default=400)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=20)

    # output folder
    parser.add_argument('--save_dir', type=str, default='..')
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--save_anndata', type=bool, default=False)
    # number of trials when executing compert.sweep
    parser.add_argument('--sweep_seeds', type=int, default=200)
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.add_argument('--model_name', type=str, default=time)


    print('Argument parser set up.')

    return dict(vars(parser.parse_args()))

def parse_arguments_eval():
    """
    Read arguments if this script is called from a terminal.
    """
    parser = argparse.ArgumentParser(
        usage="python main.py --func {function_name} [--option]",
        description='Model arguments')
    # dataset arguments
    parser.add_argument('--func', type=str, required=True)
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--training_scenario', type=int, required=True)
    parser.add_argument('--folder', type=str, default='')
    parser.add_argument('--n_sample', type=int, default=1)
    parser.add_argument('--sample_dist', type=str, nargs='+', default='posterior')
    parser.add_argument('--modality', type=str, default='RNA')
    parser.add_argument('--ls_constant_genes', type=int, default=7)
    parser.add_argument('--ls_constant_protein', type=int, default=7)
    parser.add_argument('--eval_hvg', type=bool, default=False)


    return dict(vars(parser.parse_args()))