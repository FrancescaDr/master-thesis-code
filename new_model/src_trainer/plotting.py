import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr
from scipy.sparse import csc_matrix
from scvi._compat import Literal
import matplotlib.gridspec as gridspec
from scipy import stats
import seaborn as sns

from src_trainer.TOTALVI_model import TOTALVI
from src_trainer.SCVI_model import SCVI
from src_trainer.cellPMVI_model import cellPMVI_model
from src_trainer.cellPMVI_citeseq_model import cellPMVI_CITESEQ
import pickle

class ModelEvaluation:

    def __init__(self,
                 args):

        model_type = args['model_type']
        filename = args['filename']
        folder = args['folder']
        self.n_samples = args['n_sample']
        self.sample_dist_list = args['sample_dist']
        self.modality_dict = {'RNA': 0, 'protein': 1}
        self.args = args

        if model_type == 'SCVI_RNA':
            self.my_model = SCVI.load("../input/{}{}_{}".format(folder, model_type, filename))
        elif model_type == 'cellPMVI' or model_type == 'cellPMVI_lp':
            self.my_model = cellPMVI_model.load("../input/{}{}_{}".format(folder, model_type, filename))
        elif model_type == 'MMVI_CITESEQ':
            self.my_model = cellPMVI_CITESEQ.load("../input/{}{}_{}".format(folder, model_type, filename))
        elif model_type == 'TOTALVI':
            self.my_model = TOTALVI.load("../input/{}{}_{}".format(folder, model_type, filename))
        else:
            print("Invalid model type.")

        # list with position: 0 = RNA, 1 = protein
        self.all_markers = []
        self.data_org = []
        self.all_markers.append(self.my_model.adata.var_names)
        data = ad.read(self.my_model.args["dataset_path"])
        data.obsm['adt_raw'] = data.obsm['adt_raw'].iloc[:, 1:]
        self.data_org.append(data.copy())

        # multi-modality models: add protein data
        if model_type == 'cellPMVI' or model_type == 'cellPMVI_lp' or model_type == 'TOTALVI':
            self.all_markers.append(self.my_model.adata.obsm['adt_raw'].columns)
            self.data_org.append(ad.AnnData(csc_matrix(data.obsm['adt_raw']), obs=data.obs))
            self.data_org[1].var_names = self.all_markers[1]

        # preprocess original data
        self.data_org[0].raw = self.data_org[0]

        sc.pp.normalize_total(self.data_org[0])
        sc.pp.log1p(self.data_org[0])

        # Filter genes with low counts out
        sc.pp.filter_genes(self.data_org[0], min_counts=3)

        # Select highly-variable genes
        sc.pp.highly_variable_genes(
            self.data_org[0],
            n_top_genes=5000,
            subset=True,
            layer="counts",
            flavor="seurat_v3",
        )


        if args['training_scenario'] == 3:
            self.data_org[0].obsm['adt_raw'] = self.data_org[0].obsm['adt_raw'].iloc[:, 1:]
            self.my_model.adata = self.data_org[0]

    def get_cat_covs(self,
                     cond_dict,
                     adata=None):
        """
        Finds the covariate indices from the conditions in cond_dict.
        If adata is None, then uses the adata associated with my_model;
        Otherwise, uses adata that is passed.

        Parameters:
            my_model: 'scvi-tools model'
                Works with model from SCVI, TOTALVI, MMVAE

            cond_dict: 'dict'
                Name of condition as key and covariate specification as value,
                e.g. group: LD

            adata: 'AnnDAta'
                Only necessary if combination of cond_dict is not in adata from model

        Return:
            cat_covs: 'list'
                List with indices for covariates.
            indices: 'List'
                List of length n_cells is the specific condition from cond_dict.
                Indicating position indices of conditions in data.
        """
        if adata is None:
            adata = self.my_model.adata
        # condition indicated should either be in dataset or in categorical condition of model
        if self.my_model.args['cat_cov_keys'] is not None:
            cat_covs = []
            bool_list = []
            for cat_key in self.my_model.args['cat_cov_keys']:
                try:
                    if cat_key in cond_dict.keys():
                        covariates_names = np.array(adata.obs[cat_key])
                        unique, _ = np.unique(covariates_names, return_inverse=True)
                        cat_covs.append(np.where(unique == cond_dict[cat_key])[0][0].tolist())
                        bool_list.append(np.where(adata.obs[cat_key] == cond_dict[cat_key])[0].tolist())
                except ValueError:
                    print('Invalid combination for reconstruction because cat_key {} is not in data'.format(cat_key))
            indices = list(set.intersection(*[set(list) for list in bool_list]))
        else:
            indices = np.arange(adata.n_obs)
            cat_covs = None
        return cat_covs, indices

    def get_reconstruction(self,
                           cond_dict,
                           sample_dist):
        """
        Samples from trained model and returns the sampled output with same cell count as in original data

        Parameters:
            cond_dict: 'dict'
                Contains value for each categorical covariate to be conditioned on
            sample_dist:  'str'
                distribution to sample from one of
                * ``'posterior'``
                * ``'prior'``
                * ``transfer``
            n_sample: 'int': default = 1
                Number of sampled reconstructions to return

        Returns:
            x_new: 'numpy array'
                sampled reconstruction of size n_cells x n_genes
            indices:
                indices of cells that are included in the condition
            use_model_data: 'bool'
                indicates if categorical covariate combination is in model data or not
                True: If data combination was included during training
        """
        use_model_data = True
        if sample_dist == 'transfer':
            # when data is excluded from data set use original data
            data = self.data_org[0]
            _, indices = self.get_cat_covs(cond_dict, adata=data)
            """
            if self.my_model.args['subset_PG'] == cond_dict['group'] and self.my_model.args['subset_AD'] == cond_dict['condition']:
                print("original data: group {} and adjuvant {}".format(cond_dict['group'], cond_dict['condition']))
                data = self.data_org[0]
                _, indices = self.get_cat_covs(cond_dict, adata=data)
                use_model_data = False
            else:
                _, indices = self.get_cat_covs(cond_dict)
                data = self.my_model.adata
            """
            # transfer conditions
            if cond_dict['condition'] == 'medium':
                cond_dict['condition'] = 'PI'
            else:
                cond_dict['condition'] = 'medium'
            cat_covs, indices_tr = self.get_cat_covs(cond_dict, adata=data)
            x_new = self.my_model.transfer_predictive_sample(adata=data,
                                                        indices=indices,
                                                        cat_covs=cat_covs)
            return x_new, indices_tr, False

        else:
            cat_covs, indices = self.get_cat_covs(cond_dict)

            # Training scenario 1: All data included
            if sample_dist == 'posterior':
                if len(indices) < 1:
                    return None, None, False

                x_new = self.my_model.posterior_predictive_sample(adata=self.my_model.adata,
                                                             indices=indices,
                                                             n_samples=self.n_samples)
            elif sample_dist == 'prior':
                if len(indices) < 1:
                    org_data = self.data_org[0]
                    cat_covs, indices = self.get_cat_covs(cond_dict, adata=org_data)
                    n_sample = len(indices)
                    x_new = self.my_model.prior_predictive_sample(cat_covs=cat_covs,
                                                                n_samples=n_sample,
                                                                  ls_constant_genes=self.args['ls_constant_genes'],
                                                                  ls_constant_protein=self.args['ls_constant_protein'])
                    return x_new, indices, False

                n_sample = len(indices)
                x_new = self.my_model.prior_predictive_sample(cat_covs=cat_covs,
                                                         n_samples=n_sample,
                                                              ls_constant_genes=self.args['ls_constant_genes'],
                                                              ls_constant_protein=self.args['ls_constant_protein'])

        return x_new, indices, use_model_data

    def evaluate_corr(self,
                      sample_dist: str="posterior",
                      plot_ranks: bool=False,
                      eval_hvg: bool=False,
                      save_df: bool=False,
                      save_y_pred: bool=True,
                      modality: str='RNA'
                    ):
        """
        Calculates the R2 score about means and variances for all genes, as
        well as R2 score about means and variances about differentially expressed
        (_de) genes.

        Parameters:
            org_x:  `AnnData`
                Contains count values of original data
            pred_x: n_cells x n_genes `numpy array`
                Contains predicted count values
            plot_corr: 'bool'
                If True returns scatterplot for the gene mean and variance comparison
            plot_ranks: 'bool'
                If True: ranks genes according to correlation scores and plots the correlation between true and predicted rank
            eval_hvg: 'bool'
                If True evaluate common hvg,
                Otherwise print None.
            eval_hvg: 'bool'
                If true returns boxplots a given condition for correlations.
            save_df: 'bool'
                If True, saves the score df with name: my_model.args['model_name] + '_' + sample_dist
            modality: 'str'
                evaluate one of
                    * ``'RNA'``
                    * ``'protein'``

        Returns:
            scores: 'pandas.Dataframe'
                Scores of correlation values.
        """
        scores = pd.DataFrame(
            columns=[
                "PG",  # Population group
                "CT",  # Cell type
                "ADJ",  # Adjuvant
                "corr_mean",
                "corr_mean_DE",
                "corr_var",
                "corr_var_DE",
                "common_hvg",
                "model",
                "num_cells",
            ]
        )

        icond = 0
        y_pred_dict = {}
        y_expr_dict = {} # save original and reconstructed matrix expression

        for pg in np.unique(self.my_model.adata.obs['group']):
            for ct in np.unique(self.my_model.adata.obs['annotation_L1']):
                de_genes = differentially_expressed_genes(ct)
                de_idx = np.where(self.my_model.adata.var_names.isin(np.array(de_genes)))[0]
                for adj in np.unique(self.my_model.adata.obs['condition']):
                    cond_dict = {'group': pg,
                                 'annotation_L1': ct,
                                 'condition': adj}

                    y_pred, indices, use_model_data = self.get_reconstruction(cond_dict, sample_dist)

                    if y_pred is None and indices is None:
                        print("Invalid combination: {}, {} and {}".format(pg, ct, adj))
                        continue

                    if len(y_pred) == 2:  # in case of multi-modality prediction
                        y_pred = y_pred[self.modality_dict[modality]]

                    if use_model_data:
                        data = self.my_model.adata.copy()
                        if modality == 'RNA':
                            #X = self.my_model.adata.X.copy()
                            X = self.my_model.adata.layers['counts'].copy()
                            y_true = X[indices, :].toarray()
                        elif modality == 'protein':
                            X = self.my_model.adata.obsm['adt_raw'].copy()
                            y_true = X.iloc[indices, :].to_numpy()
                    else:
                        data = self.data_org[self.modality_dict[modality]].copy()
                        X = data.layers['counts'].copy()
                        y_true = X[indices, :].toarray()

                    # true means and variances
                    yt_m = y_true.mean(axis=0)
                    yt_v = y_true.var(axis=0)
                    # true means and variances
                    yp_m = y_pred.mean(axis=0)
                    yp_v = y_pred.var(axis=0)

                    mean_score = round(spearmanr(yt_m, yp_m)[0], 3)
                    var_score = round(spearmanr(yt_v, yp_v)[0], 3)

                    # FIX INDEX FOR DE
                    if modality == 'RNA':
                        mean_score_de = round(spearmanr(yt_m[de_idx], yp_m[de_idx])[0], 3)
                        var_score_de = round(spearmanr(yt_v[de_idx], yp_v[de_idx])[0], 3)
                    else:
                        mean_score_de = None
                        var_score_de = None

                    # rank genes and plot correspondance
                    if plot_ranks:
                        yt_m_r = stats.rankdata(yt_m)
                        yt_v_r = stats.rankdata(yt_v)
                        yp_m_r = stats.rankdata(yp_m)
                        yp_v_r = stats.rankdata(yp_v)
                        txt = "Correlation of ranks for cell type {} in population group {} given adjuvant {}".format(
                            ct, pg, adj)
                        scatterplot_corr(yt_m_r, yt_v_r, yp_m_r, yp_v_r, txt)
                        idx = np.argmax(yt_m)

                        print("The gene with the highest rank is in true is: {} and pred: {}".format(
                            self.all_markers[np.argmax(yt_m)],
                            self.all_markers[np.argmax(yp_m)]
                        ))

                    if eval_hvg:
                        print(len(y_pred))
                        print(len(self.all_markers[self.modality_dict[modality]]))
                        recon_adata = ad.AnnData(csc_matrix(pd.DataFrame(y_pred, columns=self.all_markers[self.modality_dict[modality]])))
                        recon_adata.var_names = self.all_markers[self.modality_dict[modality]]
                        common_hvg = len(compare_hvg(recon_adata=recon_adata, org_adata=data, n_top_genes=1000))
                        # de_recon_data = recon_adata[de_idx]
                        # common_hvg_de = len(compare_hvg(recon_adata=recon_adata[de_idx], org_adata=data[de_idx], n_top_genes=1000))
                    else:
                        common_hvg = None
                        common_hvg_de = None

                    scores.loc[icond] = [
                        pg,
                        ct,
                        adj,
                        mean_score,
                        mean_score_de,
                        var_score,
                        var_score_de,
                        common_hvg,
                        self.my_model.model_name,
                        len(indices),
                    ]
                    icond += 1

                    if save_y_pred:
                        key ='{}_{}_{}'.format(pg, ct, adj)
                        y_pred_dict[key] = [yt_m, yt_v, yp_m, yp_v]
                        expr_key = '{}_{}'.format(pg, adj)
                        if expr_key not in y_expr_dict:
                            if sample_dist == 'transfer':
                                if adj == 'medium':
                                    cond_dict['condition'] = 'PI'
                                else:
                                    cond_dict['condition'] = 'medium'
                                _, indices = self.get_cat_covs(cond_dict)
                                #y_true_transfer = X[indices, :].toarray()
                                y_expr_dict[expr_key] = [y_true, y_pred] #[y_true_adj y_true_transfer, y_pred]
                            else:
                                y_expr_dict[expr_key] = [y_true, y_pred]

        print(scores)
        if save_df:
            print("save df")
            scores.to_csv(
                '../results/csv/{}_{}_{}_{}.csv'.format(self.my_model.args['model_type'],
                                                             self.my_model.args['model_name'],
                                                     sample_dist, modality))
        if save_y_pred:
            with open('../results/y_pred/{}_{}_{}_{}.pickle'.format(self.my_model.args['model_type'], self.my_model.args['model_name'],
                                                     sample_dist, modality), 'wb') as handle:
                pickle.dump(y_pred_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('../results/y_expr/{}_{}_{}_{}.pickle'.format(self.my_model.args['model_type'], self.my_model.args['model_name'],
                                                     sample_dist, modality), 'wb') as handle:
                pickle.dump(y_expr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return scores

def set_font_size():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plot_single_value(model, args, condition, pdf=None):
    """
    Plots one value from single condition
    :param model:
    :param model_name:
    :param args:
    :param condition: string: 'train' or 'validation'
    :return:
    """
    plt.figure()
    value = model.history[args["var"]]
    x = np.linspace(0, (len(value)), num=len(value))
    plt.plot(x, value, label=args["var_label"])
    if args["ylim"] is not None:
        plt.ylim(args["ylim"][0], args["ylim"][1])
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(args["var_label"])
    plt.title('{} {} of {} data'.format(condition, args["var_label"], args['modality']))
    #plt.savefig("../{}_{}.png".format(model_name, args["var_label"]))
    if pdf is not None:
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
    else:
        plt.show()

def plot_train_val(model, plot_args, pdf=None):
    plt.figure()
    plt.plot(model.history['{}_train'.format(plot_args['var'])]['{}_train'.format(plot_args['var'])], label='train');
    plt.plot(model.history['{}_validation'.format(plot_args['var'])]['{}_validation'.format(plot_args['var'])], label='validation');
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(plot_args['var_label'])
    if plot_args["ylim"] is not None:
        plt.ylim(plot_args["ylim"][0], plot_args["ylim"][1])
    plt.title('Train and validation {} loss of {} data'.format(plot_args['var_label'], plot_args['modality']))
    #plt.savefig("../{}_trainval_{}.png".format(model_name, plot_args['var_label']))
    if pdf is not None:
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
    else:
        plt.show()

def plot_library_size(folder, model_type_dict, library_sizes, color_list, modality, model_labels):
    """Plots the mean correlation values for prior predictive sampling
     for the given models over different library sizes
     library_sizes: np.array with library sizes measured
     """
    fig, ax = plt.subplots(1)
    for i, (model_type, model_name) in enumerate(model_type_dict.items()):
        mu = []
        sigma = []
        for ls in library_sizes:
            df = pd.read_csv('../../results/{}/{}_ls{}_{}_{}_{}.csv'.format(folder, model_type, ls, model_name, 'prior', modality))
            mu.append(df['corr_mean'].mean())
            sigma.append(df['corr_mean'].std())
        print("Mu: {}, sigma: {}".format(mu, sigma))
        #ax.scatter(library_sizes, np.array(mu))
        #ax.plot(library_sizes, np.array(mu), lw=2, label=model_type)
        #ax.fill_between(library_sizes, np.array(mu)+np.array(sigma), np.array(mu)-np.array(sigma), alpha=0.2)
        plt.errorbar(library_sizes, np.array(mu), np.array(sigma), linestyle='None', marker='X', markersize=10, color=color_list[i])
        ax.plot(library_sizes, np.array(mu), lw=2, label=model_labels[i], color=color_list[i])
    #ax.set_title('Influence of library size on correlation mean for reconstruction')
    ax.legend().remove()
    ax.set_xlabel('Library size', fontsize=14)
    ax.set_ylabel('Correlation mean', fontsize=14)
    ax.grid(axis='y', zorder=0)
    #ax.grid()
    plt.show()


def plot_gene_embeddings(org_adata, org_x, new_adata, txt, pdf, marker_genes=None):
    """
    :param adata:
    :param my_model:
    :param filename:
    :param x_new: reconstructed data
    :param marker_genes:
    :return:
    """
    plt.figure()
    fig, axes = plt.subplots(1, 2)
    #fig.suptitle('Original and Reconstructed {} embeddings of cell type {} for markers: {}'.format('RNA', cell_type, marker_genes))
    fig.suptitle(txt)
    # Embedding original data
    create_embedding(org_adata, org_x, 'Original', axes[0], marker_genes)
    # Embedding for reconstructed data
    create_embedding(new_adata, new_adata.X, 'Reconstruction',  axes[1], marker_genes)
    #plt.savefig("../{}_umap_{}.png".format(filename, my_model.args['modality']))
    #fig.text(0, 0, txt, ha='center')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    """
    if pdf is not None:
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
    """
    plt.show()


def create_embedding(adata, x, label, axes, markers=None):
    adata.obsm["X_umap"] = x
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, use_rep="X_umap")
    #sc.tl.leiden(adata)
    sc.tl.umap(adata)
    sc.pl.umap(
        adata,
        color=markers,
        frameon=False,
        ncols=1,
        title=label,
        #save="umap_{}_{}".format(model_name, label),
        show=False,
        ax=axes,
    )

def figure_caption(modality, marker, cell_type, pop_group, pert_cond):
    return "{} expression of marker {} for \n " \
           "cell type {}, pop group {} and pert cond {}."\
        .format(modality, marker, cell_type, pop_group, pert_cond)

def compare_hvg(recon_adata, org_adata, n_top_genes=10):
    """
    Compares the highly variable genes of the original and predicted data

    Parameters:
        recon_adata:  `AnnData`
            sampled data
        org_adata:  `AnnData`
            original data
        n_top_genes: `integer`
            number of genes to sample from

    :return:
        common_hvg: `Index`
            common hvg between original and predicted data
    """

    sc.pp.highly_variable_genes(
        recon_adata,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
        span=1.0,
    )
    # Print what are the highly variable genes
    hvg_recon = recon_adata.var.highly_variable[recon_adata.var.highly_variable].index
    df = org_adata.var.sort_values(by=['highly_variable_rank'])
    #hvg_org = org_adata.var.highly_variable[org_adata.var.highly_variable].index
    hvg_org = df.highly_variable[df.highly_variable].index[:n_top_genes]

    common_hvg = hvg_recon.intersection(hvg_org)

    #print("The difference for the {} hvg between the reconstructed and orignal data is: {}".format(n_top_genes, len(common_hvg)))
    #print("The hvg that both detected: {}".format(common_hvg))

    return common_hvg

def plot_latentspace(my_model):
    adata = my_model.adata
    adata.obsm["X_scVI"] = my_model.get_latent_representation()
    # use scVI latent space for UMAP generation
    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.umap(adata, min_dist=0.3)

    sc.pl.umap(
        adata,
        color=["annotation_L1"],
        frameon=False,
        show=True,
    )
    sc.pl.umap(
        adata,
        color=["group", "condition"],
        ncols=2,
        frameon=False,
        show=True,
    )


def save_loss(my_model, model_args, model_name):
    with PdfPages('../results/{}_loss.pdf'.format(model_name)) as pdf:
        args_elbo = {'var': "elbo",
                     'var_label': "elbo",
                     'ylim': None,
                     'modality': model_args['modality']
                     }
        args_rl = {'var': "reconstruction_loss",
                   'var_label': "rl",
                   'ylim': None,
                   'modality': model_args['modality']
                   }
        args_kld = {'var': "kl_local",
                    'var_label': "kld",
                    'ylim': None,
                    'modality': model_args['modality']
                    }

        list_args = [args_elbo, args_rl, args_kld]

        for plot_args in list_args:
            if model_args['check_val_every_n_epoch'] is not None:
                plot_train_val(my_model, plot_args, pdf)
            else:
                plot_single_value(my_model, plot_args, 'train', pdf)

        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = 'Loss for model {}'.format(model_name)

def histogram_expressionChange(x, marker_idx, medium_idx, pert_idx):
    # marker selection (ndarray: [n_cells, n_marker])
    org_x_medium = x[medium_idx, marker_idx]
    org_x_pert = x[pert_idx, marker_idx]
    print(org_x_medium)
    # append missing 0s to array to make same size
    #org_x_medium = np.pad(org_x_medium, pad_width=(0,len(medium_idx)-len(org_x_medium)), mode='constant', constant_values=0)
    #org_x_pert = np.pad(org_x_pert, pad_width=(0,len(pert_idx)-len(org_x_pert)), mode='constant', constant_values=0)
    plt.figure()
    bins = 20
    bins = np.histogram(np.hstack((org_x_medium, org_x_pert)), bins=bins)[1]
    plt.hist(org_x_medium, bins, edgecolor='black', alpha=0.5, label='Original')
    plt.hist(org_x_pert, bins, edgecolor='black', alpha=0.5, label='Reconstructed')
    plt.legend()
    #plt.set(xlabel="Counts", ylabel="Percentage")
    plt.show()

def plot_reconstruction(data, recon_adata, marker_mask, txt, all_markers, indices, pdf=None):
    """
    Plots the histogram and UMAP for the original and reconstructed data samples

    Parameters:
        data:  `AnnData`
            original gene data

        recon_adata: `AnnData`
            reconstucted data

        marker_mask:
            indices of marker to include in the analysis

        txt: string
            Figure description

        all_markers:
    """
    # all marker data for the given covariate conditions (ndarray: [n_cells, n_genes])
    org_x = data[indices].X
    # marker selection (ndarray: [n_cells, n_marker])
    org_x_gene = data[indices, marker_mask].X.data

    # marker selection (ndarray: [n_cells, n_marker])
    recon_x = recon_adata.X
    recon_x_gene = recon_adata[:, marker_mask].X.data

    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=2, ncols=2)

    ax11 = fig.add_subplot(gs[0, :2])
    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])
    fig.tight_layout()
    # Plot histogram
    # append missing 0s to array to make same size
    org_x_gene = np.pad(org_x_gene, pad_width=(0,len(indices)-len(org_x_gene)), mode='constant', constant_values=0)
    recon_x_gene = np.pad(recon_x_gene, pad_width=(0,len(indices) - len(recon_x_gene)), mode='constant', constant_values=0)
    #df = pd.DataFrame({'org': org_x_gene, 'recon': recon_x_gene})
    #sns.histplot(data=[df['org'], df['recon']], ax=ax11, color=['r', 'b'])
    #sns.histplot(data=df, x="org", color="blue", label="Original", ax=ax11, stat="percent")
    #sns.histplot(data=df, x="recon", color="red", label="Reconstructed", ax=ax11, stat="percent")
    bins = 20
    bins = np.histogram(np.hstack((org_x_gene, recon_x_gene)), bins=bins)[1]
    ax11.hist(org_x_gene, bins, edgecolor='black', alpha=0.5, label='Original')
    ax11.hist(recon_x_gene, bins, edgecolor='black', alpha=0.5, label='Reconstructed')
    ax11.legend()
    ax11.set(xlabel="Counts", ylabel="Percentage")
    #sns.histplot(data=[recon_x_gene, org_x_gene], stat="percent", ax=ax11,  color=['r', 'b'], bins = 20)
    #ax11.legend(['recon', 'org'], title='Data')
    marker = [all_markers[marker_i] for marker_i in marker_mask]

    # Embedding original data
    create_embedding(data[indices], org_x, 'Original', ax21, marker)
    # Embedding for reconstructed data
    create_embedding(recon_adata, recon_x, 'Reconstruction', ax22, marker)
    fig.suptitle(txt)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    if pdf is not None:
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
    else:
        plt.show()



def differentially_expressed_genes(cell_type):
    """
    Finds all differentially expressed genes (adj. p value < 0.05) of the cell type.

    Parameters:
        cell_type: 'str'
            Name of the cell type to find differentially expressed genes

    Return:
        de_genes: 'list'
            List with de gene names for cell type
    """
    df = pd.read_csv('../diffexp/{}.csv'.format(cell_type))
    df.index = df.iloc[:, 0]
    de_genes = df.index[df['adj.P.Val'] < 0.05].tolist()
    return de_genes

def boxplot_grouped_sampleDist(model_type_dict,
                         sample_dist_list,
                         cond_list,
                         score = "corr_mean",
                         folder="csv",
                         modality = "RNA"):
    df_list = []
    mean_dict = {}
    std_dict = {}
    for model_type, model_name in model_type_dict.items():
        for sample_dist in sample_dist_list:
            df = pd.read_csv('../results/{}/{}_{}_{}_{}.csv'.format(folder, model_type, model_name, sample_dist, modality))
            for cond_dict in cond_list:
                idx_list = []
                for col, cond in cond_dict.items():
                    if cond is not None:
                        idx_list.append(set(np.where([df[col] == cond])[1]))
                bool_idx = list(set.intersection(*map(set, idx_list)))
                df["box_hue"] = sample_dist
                df["model_type"] = model_type
                df_list.append(df.iloc[bool_idx, :][["box_hue", score, "model_type", "ADJ"]])
                mean_dict[model_type + '_' + sample_dist] = df.iloc[bool_idx, :][score].mean()
                std_dict[model_type + '_' + sample_dist] = df.iloc[bool_idx, :][score].std()
    # merge dataframes
    dataframe = pd.concat(df_list)
    print(dataframe)

    # plot seperate figures for each distribution
    for sample_dist in sample_dist_list:
        plt.figure()
        sd_df = dataframe[dataframe["box_hue"] == sample_dist]
        sns.boxplot(y=score, x='model_type',
                    data=sd_df,
                    hue='ADJ',
                    palette="colorblind")
        plt.title("{} predictive sampling (PG: {})".format(sample_dist, cond_dict['PG']))
        plt.show()
    return dataframe, mean_dict, std_dict

def boxplot_grouped_corr(model_type_dict,
                        sample_dist_list,
                         cond_dict,
                         score = "corr_mean",
                         folder="csv",
                         modality="RNA"):
    df_list = []
    mean_dict = {}
    std_dict = {}
    model_names = ['scVI', 'TotalVI', 'cellPMVI']
    for i, (model_type, model_name) in enumerate(model_type_dict.items()):
        for sample_dist in sample_dist_list:
            df = pd.read_csv('../results/{}/{}_{}_{}_{}.csv'.format(folder, model_type, model_name, sample_dist, modality))
            idx_list = []
            for col, cond in cond_dict.items():
                if cond is not None:
                    idx_list.append(set(np.where([df[col] == cond])[1]))
            bool_idx = list(set.intersection(*map(set, idx_list)))
            df["box_hue"] = sample_dist
            df["model_type"] = model_names[i]
            df_list.append(df.iloc[bool_idx, :][["box_hue", score, "model_type"]])
            mean_dict[model_type + '_' + sample_dist] = df.iloc[bool_idx, :][score].mean()
            std_dict[model_type + '_' + sample_dist] = df.iloc[bool_idx, :][score].std()
    # merge dataframes
    dataframe = pd.concat(df_list)

    p = sns.boxplot(y=score, x='box_hue',
                data=dataframe,
                palette="colorblind",
                hue='model_type')
    plt.title("PG: {}, ADJ: {}, CT: {}".format(cond_dict['PG'], cond_dict['ADJ'], cond_dict['CT']))
    p.set_xlabel("Sample distribution", fontsize=14)
    p.set_ylabel("Correlation mean", fontsize=14)
    p.legend(title='Model', fontsize=12)
    plt.show()
    return dataframe, mean_dict, std_dict

def average_corr_mean_condition(folder, model_type, model_name, sample_dist_list, condition,
                                score = "corr_mean",
                                modality="RNA"):
    results_dict = {}
    for sample_dist in sample_dist_list:
        df = pd.read_csv('../results/{}/{}_{}_{}_{}.csv'.format(folder, model_type, model_name, sample_dist, modality))
        row = []
        for key in df[condition].unique():
            idx = np.where(df[condition] == key)[0]
            avg = np.mean(df[score][idx])
            row.append(np.round(avg,2))
            #print("Avergae for {}: {}".format(sample_dist, avg))
        results_dict[sample_dist] = row
    results_dict['Condition'] = df[condition].unique().tolist()
    results = pd.DataFrame(results_dict)
    return results

def swarmplot_cond_corr(model_type_list,
                        cat_cond_list,
                        cond_corr_dict,
                         score = "prior",
                        fig_name=None):
    """
    Swarmplot for each categorical condition indicated
    """
    fig, axs = plt.subplots(1, len(cat_cond_list), figsize=(15,8))
    for i, cat_cond in enumerate(cat_cond_list):
        results_dict = cond_corr_dict[cat_cond]
        df_list = []
        for model_type in model_type_list:
            df = results_dict[model_type]
            df["model_type"] = model_type
            df_list.append(df)
        # merge dataframes
        dataframe = pd.concat(df_list)
        print(dataframe)
        p = sns.swarmplot(y=score, x="model_type",
                      data=dataframe,
                      palette="colorblind",
                      hue='Condition',
                      ax=axs[i],
                      size = 10
                      )
        p.set_xlabel("Model type", fontsize=14)
        p.set_ylabel("Correlation mean", fontsize=14)
        p.set_xticklabels(['scVI', 'totalVI', 'cellPMVI'])
        plt.title(cat_cond)
    plt.rc('legend', loc="upper right")
    if fig_name is not None:
        fig.savefig('../figures/{}.png'.format(fig_name))
    plt.show()


def swarmplot_categories(model_type,
                        cat_cond_list,
                        cond_corr_dict,
                         score = "prior",
                        fig_name=None):
    """
    Swarmplot for each categorical condition indicated for one model
    """
    x_label_title = ['Population group', 'Adjuvant', 'Cell type']
    fig, axs = plt.subplots(1, len(cat_cond_list), sharey=True, figsize=(5,8))
    for i, cat_cond in enumerate(cat_cond_list):
        df_list = []
        results_dict = cond_corr_dict[cat_cond]
        df = results_dict[model_type]
        df["cat_cond"] = x_label_title[i]
        df_list.append(df)
        # merge dataframes
        dataframe = pd.concat(df_list)
        print(dataframe)
        p = sns.swarmplot(y=score, x='cat_cond',
                          data=dataframe,
                          palette="colorblind",
                          hue='Condition',
                          size = 8,
                          ax=axs[i]
                      )
        #if i == 2:
            #p.legend(loc='lower left',bbox_to_anchor = (1, 0.5))
        p.legend(loc='lower center', bbox_to_anchor=(1.5, 1.5),
                  ncol=3, fancybox=True, shadow=True)
        set_font_size()
        #axs[i].set_xlabel(x_label_title[i], fontsize=14)
    axs[0].set_ylabel("Correlation mean", fontsize=14)
    axs[1].set_ylabel('')
    axs[2].set_ylabel('')
    axs[0].set_xlabel('')
    axs[1].set_xlabel('Categorical covariates', fontsize=14)
    axs[2].set_xlabel('')


    if fig_name is not None:
        fig.savefig('../figures/{}.png'.format(fig_name))
    plt.show()

def swarmplot_grouped_corr(model_type,
                           model_name,
                           sample_dist_list,
                           pg_list,
                           adj_list,
                         score = "corr_mean",
                         hue = "CT",
                         folder="csv_1"):
    fig, axs = plt.subplots(len(adj_list), len(pg_list), figsize=(15,15))
    for adj_i, adj in enumerate(adj_list):
        for pg_i, pg in enumerate(pg_list):
            df_list = []
            for sample_dist in sample_dist_list:
                df = pd.read_csv('../results/{}/{}_{}_{}.csv'.format(folder, model_type, model_name, sample_dist))
                idx_list = []
                idx_list.append(set(np.where([df["PG"] == pg])[1]))
                idx_list.append(set(np.where([df["ADJ"] == adj])[1]))
                bool_idx = list(set.intersection(*map(set, idx_list)))
                df["box_hue"] = sample_dist
                df_list.append(df.iloc[bool_idx, :][["box_hue", score, "CT", "PG"]])
            # merge dataframes
            dataframe = pd.concat(df_list)
            sns.swarmplot(y=score, x="box_hue",
                          data=dataframe,
                          palette="colorblind",
                          hue=hue,
                          ax=axs[adj_i, pg_i],
                          size = 10
                          )
            plt.title("PG: {}, ADJ: {}".format(pg, adj))
    #plt.rc('legend', loc="upper right")
    axs.flatten()[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    for ax in axs.flatten()[:-1]:
        ax.get_legend().remove()

    plt.show()

def boxplot_corr(filename,
                 col_conds,
                 score = "corr_mean"):
    """
    Creates boxplots for a dataframe given the column and condition speficiations.
    Makes a figure for each column comparing the given conditions.

    Parameters:

        filename: 'str'
            path to csv file containing a pandas dataframe with correlation values
        column_cond: 'list'
            List of dictionaries containing the conditions for boxplots in format:
            [{col_1: cond_1, ..., col_i:cond_i}, ..., {col_1: cond_1, ..., col_i:cond_i}]

        score: 'str'
            Score to plot, either 'corr_mean', 'corr_mean_DE', 'corr_var' and 'corr_var_DE'
    """
    df = pd.read_csv('{}.csv'.format(filename))
    labels = []
    data_list = []
    for d in col_conds:
        label = ""
        idx_list = []
        for col, cond in d.items():
            idx_list.append(set(np.where([df[col] == cond])[1]))
            label += cond + "_"
        labels.append(label[:-1])
        bool_idx = list(set.intersection(*map(set, idx_list)))
        data_list.append(df.iloc[bool_idx, :][score])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(data_list, labels=labels)
    plt.show()

def scatterplot_corr(yt_m, yt_v, yp_m, yp_v, fig_txt, color_markers=None):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    df_mean = pd.DataFrame({'org': yt_m, 'recon': yp_m})
    df_var = pd.DataFrame({'org': yt_v, 'recon': yp_v})
    for ax, df, txt in [(ax1, df_mean, "Mean"), (ax2, df_var, "Variance")]:
        sns.scatterplot(x=df['org'], y=df['recon'], ax=ax)
        ax.plot((0, np.max(df['org'])), (0, np.max(df['org'])), '-r', linewidth=1)
        ax.set_title(txt)
        ax.set_xlabel("Original")
        ax.set_ylabel("Reconstructed")
        spearm_r = round(spearmanr(df['org'], df['recon'])[0], 2)
        ax.text(0.025, 0.95, spearm_r, fontsize=14, transform=ax.transAxes)
    if color_markers is not None:
        print('color markers')
    fig.suptitle(fig_txt)
    plt.savefig('result.png')
    plt.show()
    plt.close()

def scatterplot_corr_mean(yt_m, yp_m, fig_txt, color_markers=None):
    fig, ax = plt.subplots()
    df_mean = pd.DataFrame({'org': yt_m, 'recon': yp_m})
    sns.scatterplot(x=df_mean['org'], y=df_mean['recon'], ax=ax)
    ax.plot((0, np.max(df_mean['org'])), (0, np.max(df_mean['org'])), '-r', linewidth=1)
    ax.set_xlabel("Original", fontsize=14)
    ax.set_ylabel("Reconstructed", fontsize=14)
    spearm_r = round(spearmanr(df_mean['org'], df_mean['recon'])[0], 2)
    ax.text(0.025, 0.95,'r: ' + str(spearm_r), fontsize=14, transform=ax.transAxes)
    if color_markers is not None:
        print('color markers')
    #fig.suptitle(fig_txt)
    plt.savefig('result.png')
    plt.show()
    plt.close()

def get_feature_correlation_matrix(adata,
                                   pred_x,
                                   pdf,
                                   correlation_type: Literal["spearman", "pearson"] = "spearman",
                                   plot_corr = False,
                                   cell_type=None,):
    """
    org_x:  `AnnData`
        Contains count values of original data
    pred_x: n_cells x n_genes `numpy array`
        Contains predicted count values

    Inspired by scvi toolbox get_feature_correlation_matrix().
    :return: correlation value between original and predicted data
    """
    # mean values of cells, returns n_genes x 1 matrix
    x = np.mean(adata.X.toarray(), axis=0)
    y = np.mean(pred_x, axis=0)
    # check if values are higher than 1
    print(((x >= 1).sum()).astype(np.int))
    print(((y >= 1).sum()).astype(np.int))

    m, b, pearson_r, p_value, std_err = stats.linregress(y, x)
    r2 = pearson_r ** 2

    if correlation_type == "pearson":
        corr = np.corrcoef(x,y, rowvar=False)
    else:
        #corr, p = spearmanr(adata.X.toarray(),pred_x, axis=1)
        corr, p = spearmanr(x, y, axis=0)

    if plot_corr:
        df = pd.DataFrame({'org': x, 'recon': y})
        fig = sns.scatterplot(x=df['org'], y=df['recon'])
        fig.plot((0,np.max(x)), (0, np.max(x)), '-r', linewidth=2)
        fig.set(xlabel="Original", ylabel="Reconstructed", title="Mean expression for {}".format(cell_type))
        pdf.savefig()  # saves the current figure into a pdf page
        plt.show()
        plt.close()

    return corr, p
