import sys
sys.path.append('/tudelft.net/staff-bulk/ewi/insy/DBL/fdrummer/master_thesis')
from src_trainer.arguments import parse_arguments_train, parse_arguments_eval
from src_trainer.SCVI_model import SCVI
from src_trainer.cellPMVI_model import cellPMVI_model
from src_trainer.cellPMVI_citeseq_model import cellPMVI_CITESEQ
import anndata as ad
from src_trainer.TOTALVI_model import TOTALVI
from src_trainer.plotting import ModelEvaluation
from src_trainer.preprocessing import data_preprocessing
import pandas as pd
import sys
import warnings


def train_model():
    # read arguments and prepare data
    args = parse_arguments_train()
    #adata = ad.read(args["dataset_path"])[:100,]
    adata = ad.read(args["dataset_path"])
    print(adata)

    if "czi" in args["dataset_path"]:
        adata.obsm['adt_raw'] = adata.obsm['adt_raw'].iloc[:, 1:]
    if "pilot" in args["dataset_path"]:
        print("pilot data")
        czi = ad.read("czi.h5ad")
        protein_data = pd.DataFrame(adata.obsm['adt_raw'].copy().toarray(), columns=czi.obsm['adt_raw'].columns)
        adata.obsm['protein_expression'] = protein_data

    adata = data_preprocessing(adata, args)

    # Initialize models
    if args['model_type'] == 'SCVI_RNA':
        SCVI.setup_anndata(adata,
                           layer='counts',
                           categorical_covariate_keys=args["cat_cov_keys"])
        model = SCVI(adata, args)
    elif args['model_type'] == 'TOTALVI':
        TOTALVI.setup_anndata(adata,
                                layer='counts',
                                categorical_covariate_keys=args["cat_cov_keys"],
                                protein_expression_obsm_key='adt_raw')
        # Set up model and train
        model = TOTALVI(adata, args)
    elif "cellPMVI" in args['model_type']:
        # MMVI model using scvi decoder with gaussian distribution prior
        cellPMVI_model.setup_anndata(adata,
                                     layer='counts',
                                     categorical_covariate_keys=args["cat_cov_keys"],
                                     protein_expression_obsm_key='adt_raw')
        # Set up model and train
        if args['model_type'] == 'cellPMVI_lp':
            # laplace distribution prior'
            model = cellPMVI_model(adata,
                                   args,
                                   latent_distribution='laplace')
        else:
            model = cellPMVI_model(adata, args)
    elif args['model_type'] == 'MMVI_CITESEQ':
        # MMVI model using TotalVI decoder
        cellPMVI_CITESEQ.setup_anndata(adata,
                                       layer='counts',
                                       categorical_covariate_keys=args["cat_cov_keys"],
                                       protein_expression_obsm_key='adt_raw')
        # Set up model and train
        model = cellPMVI_CITESEQ(adata,
                                 args)
    else:
        print("No valid model type argument has been parsed.")

    print(model)
    # training
    model.train(max_epochs=args['max_epochs'],
                check_val_every_n_epoch=args['check_val_every_n_epoch'], )

    # save
    if args['save_model']:
        model.save("{}/{}_{}/".format(args['save_dir'], model.args['model_type'], model.model_name),
                   save_anndata=args['save_anndata'])


def evaluate_model(args,
                   save_df = True):
    """
    sets up model evaluation class
    """

    model_evaluation = ModelEvaluation(args,
                                       training_scenario=args['training_scenario'])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(model_evaluation.my_model._model_summary_string)

        for sample_dist in model_evaluation.sample_dist_list:
            model_evaluation.evaluate_corr(sample_dist=sample_dist, save_df=save_df, modality=args['modality'], eval_hvg = args['eval_hvg'])

if __name__ == '__main__':
    # run function using: python myscript.py --func myfunction [--args]
    # Example command: --func train_model --dataset_path ../data/czi_data.h5ad --model_type MMVAE
    if str(sys.argv[2]) == 'train_model':
        train_model()
    # Example command: --func evaluate_model --filename 20220518-222446 --model_type TOTALVI
    elif str(sys.argv[2]) == 'evaluate_model':
        args = parse_arguments_eval()
        evaluate_model(args)
