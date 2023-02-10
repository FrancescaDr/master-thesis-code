from typing import Optional, Sequence, List

from anndata import AnnData
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    LayerField,
    CategoricalObsField,
    NumericalObsField,
    CategoricalJointObsField,
    NumericalJointObsField,
    ProteinObsmField,
)

from src_trainer.cellPMVI import cellPMVAE
from src_trainer.cellPMVI_lp import cellPMVI_lp

import numpy as np
import torch
from torch.distributions import Normal
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.utils._docstrings import setup_anndata_dsp


class cellPMVI_model(UnsupervisedTrainingMixin, BaseModelClass):
    """
    scVI based MMVI model
    """

    def __init__(
        self,
        adata: AnnData,
        args: dict,
        latent_distribution: str = 'normal',
        **model_kwargs,):

        super(cellPMVI_model, self).__init__(adata)

        self.hparams = args['hparams']
        self.args = args
        print(args)

        cat_cov = []
        if self.args['cat_cov_keys'] is not None:
            encode_covariates = True
            for cov in self.args['cat_cov_keys']:
                covariates_names = np.array(adata.obs[cov])
                unique, _ = np.unique(covariates_names, return_inverse=True)
                cat_cov.append(len(set(unique)))
        else:
            encode_covariates = False

        if latent_distribution == 'laplace':
            self.module = cellPMVI_lp(n_input_rna=self.summary_stats["n_vars"],
                                      n_input_pro=self.summary_stats["n_proteins"],
                                      n_batch=self.summary_stats["n_batch"],
                                      n_hidden=self.hparams['vae_n_hidden'],
                                      n_latent=self.args['n_latent'],
                                      n_layers=self.hparams['vae_n_layers'],
                                      gene_likelihood='nb',
                                      dropout_rate=self.hparams['vae_dropout'],
                                      encode_covariates=encode_covariates,
                                      n_cats_per_cov=cat_cov,
                                      **model_kwargs, )
        else:
            self.module = cellPMVAE(n_input_rna=self.summary_stats["n_vars"],
                                    n_input_pro=self.summary_stats["n_proteins"],
                                    n_batch=self.summary_stats["n_batch"],
                                    n_hidden=self.hparams['vae_n_hidden'],
                                    n_latent=self.args['n_latent'],
                                    n_layers=self.hparams['vae_n_layers'],
                                    gene_likelihood='nb',
                                    dropout_rate=self.hparams['vae_dropout'],
                                    encode_covariates=encode_covariates,
                                    n_cats_per_cov=cat_cov,
                                    **model_kwargs, )

        self._model_summary_string = (
            "TotalVI Model with the following params: \n"
            "n_latent: {}, n_hidden {}, n_layers: {}, gene_likelihood: nb, dropout_rate: {}. \n"
            "Training conditions: \n"
            "Categorical covariates: {}, Subset data: {} (on PG: {} and ADJ: {})."
        ).format(
            self.args['n_latent'],
            self.hparams['vae_n_hidden'],
            self.hparams['vae_n_layers'],
            self.hparams['vae_dropout'],
            self.args["cat_cov_keys"],
            self.args["subset_data"],
            self.args["subset_PG"],
            self.args["subset_AD"],
        )
        self.init_params_ = self._get_init_params(locals())
        self.model_name = args['model_name']

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
            cls,
            adata: AnnData,
            protein_expression_obsm_key: str,
            protein_names_uns_key: Optional[str] = None,
            batch_key: Optional[str] = None,
            layer: Optional[str] = None,
            size_factor_key: Optional[str] = None,
            categorical_covariate_keys: Optional[List[str]] = None,
            continuous_covariate_keys: Optional[List[str]] = None,
            **kwargs,
    ) -> Optional[AnnData]:
        """
        %(summary)s.

        Parameters
        ----------
        %(param_adata)s
        protein_expression_obsm_key
            key in `adata.obsm` for protein expression data.
        protein_names_uns_key
            key in `adata.uns` for protein names. If None, will use the column names of `adata.obsm[protein_expression_obsm_key]`
            if it is a DataFrame, else will assign sequential names to proteins.
        %(param_batch_key)s
        %(param_layer)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        %(param_copy)s

        Returns
        -------
        %(returns)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        batch_field = CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(
                REGISTRY_KEYS.LABELS_KEY, None
            ),  # Default labels field for compatibility with TOTALVAE
            batch_field,
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
            ProteinObsmField(
                REGISTRY_KEYS.PROTEIN_EXP_KEY,
                protein_expression_obsm_key,
                use_batch_mask=True,
                batch_key=batch_field.attr_key,
                colnames_uns_key=protein_names_uns_key,
                is_count_data=True,
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def transfer_predictive_sample(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            cat_covs: Optional[list] = None,
            n_samples: int = 1,
    ) -> np.ndarray:
        r"""
        Generate observation samples from the posterior predictive distribution.
        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. Must be for unperturbed condition
        n_samples
            Number of samples for each cell.
        markers_list
            Names of genes of interest.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=self.summary_stats["n_batch"]
        )

        # switch perturbation conditions
        if cat_covs is not None:
            cat_covs = torch.tensor([[float(v)] * n_samples for v in cat_covs]).T

        data = [[],[]]
        for tensors in scdl:
            inference_kwargs = dict(n_samples=n_samples)
            inference_outputs, _, = self.module.forward(
                tensors,
                inference_kwargs=inference_kwargs,
                compute_loss=False,
            )

            dec_input_dict = self.module._get_generative_input(tensors, inference_outputs)
            dec_input_dict['cat_covs'] = cat_covs

            px_zs = self.module.generative(**dec_input_dict)

            # sample from each expert specific the reconstruction value
            for e, px_z in enumerate(px_zs):
                for i, generative_outputs in enumerate(px_z):

                    # skip cross-modality predictions
                    if e != i:
                        continue

                    px_r = generative_outputs["px_r"]
                    px_rate = generative_outputs["px_rate"]
                    px_dropout = generative_outputs["px_dropout"]

                    if self.module.gene_likelihood == "poisson":
                        l_train = px_rate
                        l_train = torch.clamp(l_train, max=1e8)
                        dist = torch.distributions.Poisson(
                            l_train
                        )  # Shape : (n_samples, n_cells_batch, n_genes)
                    elif self.module.gene_likelihood == "nb":
                        dist = NegativeBinomial(mu=px_rate, theta=px_r)
                    elif self.module.gene_likelihood == "zinb":
                        dist = ZeroInflatedNegativeBinomial(
                            mu=px_rate, theta=px_r, zi_logits=px_dropout
                        )
                    else:
                        raise ValueError(
                            "{} reconstruction error not handled right now".format(
                                self.module.gene_likelihood
                            )
                        )
                    exprs = dist.sample()

                    data[e].append(exprs.cpu().detach())

        x_new = torch.cat(data[0])  # Shape (n_cells, n_genes, n_samples)
        y_new = torch.cat(data[1])

        return [x_new.numpy(), y_new.numpy()]

    @torch.no_grad()
    def posterior_predictive_sample(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            n_samples: int = 1,
    ) -> np.ndarray:
        r"""
        Generate observation samples from the posterior predictive distribution.
        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples
            Number of samples for each cell.
        markers_list
            Names of genes of interest.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        Returns
        -------
        x_new : :list of p:clayss:`torch.Tensor`
            list with rna_sample, protein_sample
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=self.summary_stats["n_batch"]
        )

        if indices is None:
            indices = np.arange(adata.n_obs)

        x_new = [] # RNA
        y_new = [] # protein
        for tensors in scdl:
            samples = self.module.sample(tensors, n_samples=n_samples)
            x_new.append(samples[0])
            y_new.append(samples[1])

        x_new = torch.cat(x_new)  # Shape (n_cells, n_genes, n_samples)
        y_new = torch.cat(y_new)  # Shape (n_cells, n_genes, n_samples)

        return [x_new.numpy(), y_new.numpy()]

    @torch.no_grad()
    def prior_predictive_sample(
            self,
            ls_constant_genes,
            ls_constant_protein,
            n_samples: int = 1,
            cat_covs: Optional[list] = None,
    ) -> np.ndarray:
        """
        Sample a cell x genes matrix from the prior distribution
        """

        # Sample from Normal(0,1) distribution
        qz_m = torch.zeros(n_samples, self.args['n_latent'])
        qz_v = torch.ones(n_samples, self.args['n_latent'])
        z = Normal(qz_m, qz_v).sample()

        data = []

        library_genes = torch.ones(n_samples, 1) * ls_constant_genes
        library_proteins = torch.ones(n_samples, 1) * ls_constant_protein
        dec_batch_index = torch.zeros(n_samples, 1)
        y = torch.zeros(n_samples, 1)
        if cat_covs is not None:
            cat_covs = torch.tensor([[float(v)] * n_samples for v in cat_covs]).T

        # decoder pass
        dec_input_dict = dict(
            rna_z=z, pro_z=z,
            rna_library=library_genes, pro_library=library_proteins,
            batch_index=dec_batch_index,
            rna_y=y, pro_y=y,
            cont_covs=None,
            cat_covs=cat_covs,
        )

        px_zs = self.module.generative(**dec_input_dict)

        # sample from each expert specific the reconstruction value
        for e, px_z in enumerate(px_zs):
            for i, generative_outputs in enumerate(px_z):

                # skip cross-modality predictions
                if e != i:
                    continue

                px_r = generative_outputs["px_r"]
                px_rate = generative_outputs["px_rate"]
                px_dropout = generative_outputs["px_dropout"]

                if self.module.gene_likelihood == "poisson":
                    l_train = px_rate
                    l_train = torch.clamp(l_train, max=1e8)
                    dist = torch.distributions.Poisson(
                        l_train
                    )  # Shape : (n_samples, n_cells_batch, n_genes)
                elif self.module.gene_likelihood == "nb":
                    dist = NegativeBinomial(mu=px_rate, theta=px_r)
                elif self.module.gene_likelihood == "zinb":
                    dist = ZeroInflatedNegativeBinomial(
                        mu=px_rate, theta=px_r, zi_logits=px_dropout
                    )
                else:
                    raise ValueError(
                        "{} reconstruction error not handled right now".format(
                            self.module.gene_likelihood
                        )
                    )
                exprs = dist.sample()

                data.append(exprs.cpu().detach().numpy())

        return data