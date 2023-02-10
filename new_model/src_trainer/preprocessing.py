from random import random

import numpy as np
import scanpy as sc
import pandas as pd

def data_preprocessing(adata, args):
    """
    1. Subsetting data according to training scenario
    2. Preprocessing data

    :param adata: unprocessed data
    :param args: arguments from training parse_arguments_train()
    :return: adata for model training
    """

    if args['subset_data'] is not None:
        if args['subset_data'] == 'PG':
            # excludes data from only one population group
            bool = (adata.obs['group'] == args['subset_PG'])
            adata = adata[~bool]
        elif args['subset_data'] == 'PG_AD':
            # excludes all perturbed data from a specific population group
            bool = ((adata.obs['group'] == args['subset_PG']) &
                        (adata.obs['condition'] == args['subset_AD']))
            adata = adata[~bool]


    # Seperate data for testing:
    # The testing argument defines the percentage of data to be left out
    adata.obs['split'] = 'nan'
    if args['test_split'] is not None:
        # Right now only selection on one covariate at a time but we want to select on the covariate combinations
        for cat_cov in args['cat_cov_keys']:
            for cond in np.unique(adata.obs[cat_cov]):
                idx_list = np.where(adata.obs[cat_cov] == cond)[0].tolist()
                k = len(idx_list) * args['test_split'] // 100
                indicies = random.sample(range(len(idx_list)), k)
    data_idx = adata.obs_names[adata.obs.split != 'ood']
    adata.obs['split'].loc[data_idx] = 'train/val'
    adata = adata[adata.obs.split == 'train/val'].copy()

    adata.raw = adata

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # Filter genes with low counts out
    sc.pp.filter_genes(adata, min_counts=3)

    # Select highly-variable genes
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=args['n_hvg'],
        subset=True,
        layer="counts",
        flavor="seurat_v3",
    )

    # add gene captured as ADT as well!
    if "czi_1" in args["dataset_path"]:
        adt_meta = pd.read_csv("../data/adt_metadata.csv")
        adata.var['present_in_adt'] = adata.var.ID.isin(adt_meta.ensembl)
        hvg = set(adata.var['features'][adata.var['highly_variable'].values])
        gene_as_adt = set(adata.var['Symbol'][adata.var['present_in_adt']].values)

        hvg_comb = hvg.union(gene_as_adt)
        print(f"Number of HVGs = {len(hvg)}")
        print(f"Number of combined HVGs = {len(hvg_comb)}")

        adata.var['highly_variable'] = adata.var_names.isin(hvg_comb)

    adata.X = adata.raw[:, adata.var_names].X

    return adata




