import logging
import warnings
from collections.abc import Iterable as IterableClass
from functools import partial
from typing import Optional, Sequence, List, Union
from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ProteinObsmField,
)
from scvi.data import AnnDataManager

from scvi.module import TOTALVAE
import numpy as np
import torch
from anndata import AnnData
from scvi.utils._docstrings import setup_anndata_dsp
from scvi.dataloaders import DataSplitter
from scvi.train import AdversarialTrainingPlan, TrainRunner

from scvi.model import TOTALVI

from torch.distributions import Normal
from scvi.distributions import NegativeBinomial, NegativeBinomialMixture, ZeroInflatedNegativeBinomial
from scvi.model._utils import (
    _init_library_size,
)
from scvi.model.base import ArchesMixin, BaseModelClass, RNASeqMixin, VAEMixin

class TOTALVI(TOTALVI):

    def __init__(
            self,
            adata: AnnData,
            args: dict,
            n_latent: int = 20,
            gene_dispersion: Literal[
                "gene", "gene-batch", "gene-label", "gene-cell"
            ] = "gene",
            protein_dispersion: Literal[
                "protein", "protein-batch", "protein-label"
            ] = "protein",
            gene_likelihood: Literal["zinb", "nb"] = "nb",
            latent_distribution: Literal["normal", "ln"] = "normal",
            empirical_protein_background_prior: Optional[bool] = None,
            override_missing_proteins: bool = False,
            **model_kwargs,
    ):
        super(TOTALVI, self).__init__(adata)
        self.hparams = args['hparams']
        self.args = args
        print("Initalised own init")

        self.protein_state_registry = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.PROTEIN_EXP_KEY
        )
        if (
                ProteinObsmField.PROTEIN_BATCH_MASK in self.protein_state_registry
                and not override_missing_proteins
        ):
            batch_mask = self.protein_state_registry.protein_batch_mask
            msg = (
                    "Some proteins have all 0 counts in some batches. "
                    + "These proteins will be treated as missing measurements; however, "
                    + "this can occur due to experimental design/biology. "
                    + "Reinitialize the model with `override_missing_proteins=True`,"
                    + "to override this behavior."
            )
            warnings.warn(msg, UserWarning)
            self._use_adversarial_classifier = True
        else:
            batch_mask = None
            self._use_adversarial_classifier = False

        emp_prior = (
            empirical_protein_background_prior
            if empirical_protein_background_prior is not None
            else (self.summary_stats.n_proteins > 10)
        )
        if emp_prior:
            prior_mean, prior_scale = self._get_totalvi_protein_priors(adata)
        else:
            prior_mean, prior_scale = None, None

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)[
                CategoricalJointObsField.N_CATS_PER_KEY
            ]
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )

        n_batch = self.summary_stats.n_batch
        use_size_factor_key = (
                REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )

        self.module = TOTALVAE(
            n_input_genes=self.summary_stats.n_vars,
            n_input_proteins=self.summary_stats.n_proteins,
            n_batch=n_batch,
            n_hidden=self.hparams['vae_n_hidden'],
            n_latent=self.args['n_latent'],
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            gene_dispersion=gene_dispersion,
            protein_dispersion=protein_dispersion,
            gene_likelihood='nb',
            latent_distribution=latent_distribution,
            protein_batch_mask=batch_mask,
            protein_background_prior_mean=prior_mean,
            protein_background_prior_scale=prior_scale,
            use_size_factor_key=use_size_factor_key,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            **model_kwargs,
        )
        self._model_summary_string = (
            "TotalVI Model with the following params: \nn_latent: {}, "
            "gene_dispersion: {}, protein_dispersion: {}, gene_likelihood: {}, latent_distribution: {}. \n"
            "Training conditions: \n"
            "Categorical covariates: {}, Subset data: {} (on PG: {} and ADJ: {})."
        ).format(
            self.args['n_latent'],
            gene_dispersion,
            protein_dispersion,
            'nb',
            latent_distribution,
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

    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 4e-3,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 0.9,
            validation_size: Optional[float] = None,
            batch_size: int = 256,
            early_stopping: bool = True,
            check_val_every_n_epoch: Optional[int] = None,
            reduce_lr_on_plateau: bool = True,
            n_steps_kl_warmup: Union[int, None] = None,
            n_epochs_kl_warmup: Union[int, None] = None,
            adversarial_classifier: Optional[bool] = None,
            plan_kwargs: Optional[dict] = None,
            **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`
            or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are met, val is checked
            every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing proteins
            are detected.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        if adversarial_classifier is None:
            adversarial_classifier = self._use_adversarial_classifier
        n_steps_kl_warmup = (
            n_steps_kl_warmup
            if n_steps_kl_warmup is not None
            else int(0.75 * self.adata.n_obs)
        )
        if reduce_lr_on_plateau:
            check_val_every_n_epoch = 1

        update_dict = {
            "lr": lr,
            "adversarial_classifier": adversarial_classifier,
            "reduce_lr_on_plateau": reduce_lr_on_plateau,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = AdversarialTrainingPlan(self.module, **plan_kwargs)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            **kwargs,
        )
        return runner()

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
        if self.module.gene_likelihood not in ["zinb", "nb", "poisson"]:
            raise ValueError("Invalid gene_likelihood.")

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=self.summary_stats["n_batch"]
        )

        # switch perturbation conditions
        if cat_covs is not None:
            cat_covs = torch.tensor([[float(v)] * n_samples for v in cat_covs]).T

        x_new = []  # RNA
        y_new = []  # protein
        for tensors in scdl:
            inference_kwargs = dict(n_samples=n_samples)
            inference_outputs, _, = self.module.forward(
                tensors,
                inference_kwargs=inference_kwargs,
                compute_loss=False,
            )

            dec_input_dict = self.module._get_generative_input(tensors, inference_outputs)
            dec_input_dict['cat_covs'] = cat_covs

            generative_outputs = self.module.generative(**dec_input_dict)

            # sample
            px_ = generative_outputs["px_"]
            py_ = generative_outputs["py_"]

            rna_dist = NegativeBinomial(mu=px_["rate"], theta=px_["r"])
            protein_dist = NegativeBinomialMixture(
                mu1=py_["rate_back"],
                mu2=py_["rate_fore"],
                theta1=py_["r"],
                mixture_logits=py_["mixing"],
            )
            rna_sample = rna_dist.sample().cpu().data.numpy()
            protein_sample = protein_dist.sample().cpu().data.numpy()

            x_new += [rna_sample]
            y_new += [protein_sample]
            if n_samples > 1:
                x_new[-1] = np.transpose(x_new[-1], (1, 2, 0))
                y_new[-1] = np.transpose(y_new[-1], (1, 2, 0))

        x_new = np.concatenate(x_new, axis=0)  # Shape (n_cells, n_genes, n_samples)
        y_new = np.concatenate(y_new, axis=0)  # Shape (n_cells, n_genes, n_samples)

        return [x_new, y_new]


    @torch.no_grad()
    def posterior_predictive_sample(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        gene_list: Optional[Sequence[str]] = None,
        protein_list: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        r"""
        Generate observation samples from the posterior predictive distribution.

        The posterior predictive distribution is written as :math:`p(\hat{x}, \hat{y} \mid x, y)`.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples
            Number of required samples for each cell
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        gene_list
            Names of genes of interest
        protein_list
            Names of proteins of interest

        Returns
        -------
        x_new : :class:`~numpy.ndarray`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        if self.module.gene_likelihood not in ["nb"]:
            raise ValueError("Invalid gene_likelihood")

        adata = self._validate_anndata(adata)
        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]
        if protein_list is None:
            protein_mask = slice(None)
        else:
            all_proteins = self.protein_state_registry.column_names
            protein_mask = [True if p in protein_list else False for p in all_proteins]

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        x_new = []  # RNA
        y_new = []  # protein
        for tensors in scdl:
            rna_sample, protein_sample = self.module.sample(
                tensors, n_samples=n_samples
            )
            rna_sample = rna_sample[..., gene_mask]
            protein_sample = protein_sample[..., protein_mask]
            #x_new.append(rna_sample.numpy())
            #y_new.append(protein_sample.numpy())
            x_new += [rna_sample]
            y_new += [protein_sample]
            if n_samples > 1:
                x_new[-1] = np.transpose(x_new[-1], (1, 2, 0))
                y_new[-1] = np.transpose(y_new[-1], (1, 2, 0))

        x_new = np.concatenate(x_new, axis=0) # Shape (n_cells, n_genes, n_samples)
        y_new = np.concatenate(y_new, axis=0) # Shape (n_cells, n_genes, n_samples)

        return [x_new, y_new]

    @torch.no_grad()
    def prior_predictive_sample(
            self,
            ls_constant_genes,
            ls_constant_protein = None,
            n_samples: int = 1,
            cat_covs: Optional[list] = None,
            gene_list: Optional[Sequence[str]] = None,
            protein_list: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Sample a cell x genes matrix from the prior distribution
        """
        if self.module.gene_likelihood not in ["zinb", "nb", "poisson"]:
            raise ValueError("Invalid gene_likelihood.")

        # Sample from Normal(0,1) distribution
        qz_m = torch.zeros(n_samples, self.module.n_latent)
        qz_v = torch.ones(n_samples, self.module.n_latent)
        z = Normal(qz_m, qz_v).sample()

        library_gene = torch.ones(n_samples, 1) * ls_constant_genes
        batch_index = torch.zeros(n_samples, 1)
        y = torch.zeros(n_samples, 1)
        if cat_covs is not None:
            cat_covs = torch.tensor([[float(v)] * n_samples for v in cat_covs]).T

        # decoder pass
        dec_input_dict = dict(
            z=z,
            library_gene=library_gene,
            batch_index=batch_index,
            label=y,
            cat_covs=cat_covs,
            cont_covs=None,
            size_factor=None,
        )
        generative_outputs = self.module.generative(**dec_input_dict)

        px_ = generative_outputs["px_"]
        py_ = generative_outputs["py_"]

        rna_dist = NegativeBinomial(mu=px_["rate"], theta=px_["r"])
        protein_dist = NegativeBinomialMixture(
            mu1=py_["rate_back"],
            mu2=py_["rate_fore"],
            theta1=py_["r"],
            mixture_logits=py_["mixing"],
        )
        rna_sample = rna_dist.sample().cpu().data.numpy()
        protein_sample = protein_dist.sample().cpu().data.numpy()

        return [rna_sample, protein_sample]

def evaluate_totalVI(filename):
    my_model = TOTALVI.load("../input/model_{}".format(filename))
    with open('../input/model_{}/{}_args.pickle'.format(filename, filename), 'rb') as handle:
        model_args = pickle.load(handle)

    adata = my_model.adata
    adata.obsm["X_totalVI"] = my_model.get_latent_representation()
    sc.pp.neighbors(adata, use_rep="X_totalVI")
    sc.tl.umap(adata, min_dist=0.4)

    sc.pl.umap(
        adata,
        color=["condition", "group", "annotation_L1"],
        frameon=False,
        ncols=3,
        wspace=.1
    )