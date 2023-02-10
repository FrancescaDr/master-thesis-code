from typing import Optional, Sequence

from anndata import AnnData
from scvi.dataloaders import DataSplitter
from scvi.module import VAE
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    LayerField,
    CategoricalObsField,
    NumericalObsField,
    CategoricalJointObsField,
    NumericalJointObsField,
)
from scvi.train import TrainingPlan, TrainRunner

import numpy as np
import torch
from torch.distributions import Normal
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from src_trainer.arguments import set_hparams_
import random

class SCVI(UnsupervisedTrainingMixin, BaseModelClass):
    """
    single-cell Variational Inference [Lopez18]_.
    """

    def __init__(
        self,
        adata: AnnData,
        args: dict,
        **model_kwargs,):

        super(SCVI, self).__init__(adata)

        if args['seed'] == 0:
            self.hparams = args['hparams']
        else:
            self.hparams = set_hparams_(random.randint(1, 10))
        self.args = args
        print(args)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )

        self.module = VAE(
            n_input=self.summary_stats["n_vars"],
            n_batch=self.summary_stats["n_batch"],
            n_hidden=self.hparams['vae_n_hidden'],
            n_latent=self.args['n_latent'],
            n_layers=self.hparams['vae_n_layers'],
            gene_likelihood='nb',
            dropout_rate=self.hparams['vae_dropout'],
            n_cats_per_cov=n_cats_per_cov,
            **model_kwargs,
        )
        self._model_summary_string = (
            "SCVI RNA Model with the following params: \n"
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
    def setup_anndata(
        cls,
        adata: AnnData,
        #protein_expression_obsm_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        categorical_covariate_keys: Optional[str] = None,
        **kwargs,
    ) -> Optional[AnnData]:
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            # Dummy fields required for VAE class.
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, None, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, None
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    """
    def train(
            self,
            check_val_every_n_epoch,
            max_epochs: Optional[int] = 100,
            use_gpu: Optional[bool] = None,
            lr: float=0.001,
            weight_decay: float =1e-06,
            **kwargs,
    ):
        
        #Train the model.
        
        # object to make train/test/val dataloaders
        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=self.args["train_size"],
            batch_size=self.summary_stats["n_batch"],
            use_gpu=use_gpu,
        )
        # defines optimizers, training step, val step, logged metrics
        training_plan = TrainingPlan(
            self.module, lr=lr, weight_decay=weight_decay,
        )
        # creates Trainer, pre and post training procedures (Trainer.fit())
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=self.args['max_epochs'],
            use_gpu=use_gpu,
            check_val_every_n_epoch=check_val_every_n_epoch,
            **kwargs,
        )
        return runner()
    """

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

        x_new = []
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

            samples = dist.sample()
            x_new.append(samples)

        x_new = torch.cat(x_new)  # Shape (n_cells, n_genes, n_samples)
        return x_new.cpu().data.numpy()

    @torch.no_grad()
    def prior_predictive_sample(
            self,
            ls_constant_genes,
            ls_constant_protein=None,
            n_samples: int = 1,
            cat_covs: Optional[list] = None,
    ) -> np.ndarray:
        """
        Sample a cell x genes matrix from the prior distribution
        Parameters
        ----------
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples
            Number of samples for each cell.
        gene_list
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

        # Sample from Normal(0,1) distribution
        qz_m = torch.zeros(n_samples, self.module.n_latent)
        qz_v = torch.ones(n_samples, self.module.n_latent)
        z = Normal(qz_m, qz_v).sample()

        library = torch.ones(n_samples, 1) * ls_constant_genes
        dec_batch_index = torch.zeros(n_samples, 1)
        y = torch.zeros(n_samples, 1)
        if cat_covs is not None:
            cat_covs = torch.tensor([[float(v)] * n_samples for v in cat_covs]).T

        # decoder pass
        dec_input_dict = dict(
            z=z,
            library=library,
            batch_index=dec_batch_index,
            y=y,
            cont_covs=None,
            cat_covs=cat_covs,
            size_factor=None,
        )
        generative_outputs = self.module.generative(**dec_input_dict)

        # sample
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

        #return exprs.numpy()
        return exprs.cpu().data.numpy()

    @torch.no_grad()
    def posterior_predictive_sample(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            n_samples: int = 1,
            markers_list: Optional[Sequence[str]] = None,
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
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        if self.module.gene_likelihood not in ["zinb", "nb", "poisson"]:
            raise ValueError("Invalid gene_likelihood.")

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=self.summary_stats["n_batch"]
        )

        if indices is None:
            indices = np.arange(adata.n_obs)
        if markers_list is None:
            markers_mask = slice(None)
        elif self.args['modality'][0]=='RNA':
            all_genes = adata.var_names
            markers_mask = [True if gene in markers_list else False for gene in all_genes]
        elif self.args['modality'][0] == 'protein':
            all_proteins = adata.obsm['adt_raw'].columns
            markers_mask = [True if p in markers_list else False for p in all_proteins]

        x_new = []
        for tensors in scdl:
            samples = self.module.sample(tensors, n_samples=n_samples)
            if markers_list is not None:
                samples = samples[:, markers_mask, ...]
            x_new.append(samples)

        x_new = torch.cat(x_new)  # Shape (n_cells, n_genes, n_samples)

        return x_new.numpy()

    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.
        This is denoted as :math:`z_n` in our manuscripts.
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            qz_m = outputs["qz_m"]
            qz_v = outputs["qz_v"]
            z = outputs["z"]

            if give_mean:
                # does each model need to have this latent distribution param?
                if self.module.latent_distribution == "ln":
                    samples = Normal(qz_m, qz_v.sqrt()).sample([mc_samples])
                    z = torch.nn.functional.softmax(samples, dim=-1)
                    z = z.mean(dim=0)
                else:
                    z = qz_m

            latent += [z.cpu()]
        return torch.cat(latent).numpy()



    def sampled_latent_space_comparison(self,
                                        adata: Optional[AnnData] = None,
                                        indices: Optional[Sequence[int]] = None,
                                        n_samples: int = 1,
                                        markers_list: Optional[Sequence[str]] = None,
                                        ) -> np.ndarray:
        """
        Compares the posterior and prior sampled latent space to check if too many of the prior values are around mean 0
        :return:
        """
        # Get sampled posterior latent space
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=self.summary_stats["n_batch"]
        )

        for tensors in scdl:
            inference_kwargs = dict(n_samples=n_samples)
            inference_outputs, generative_outputs, = self.module.forward(
                tensors,
                inference_kwargs=inference_kwargs,
                compute_loss=False,
            )

        post_z = inference_outputs["z"]

        # Get sampled prior latent space
        # Sample from Normal(0,1) distribution
        qz_m = torch.zeros(n_samples, self.module.n_latent)
        qz_v = torch.ones(n_samples, self.module.n_latent)
        prior_z = Normal(qz_m, qz_v).sample()

        diff = post_z - prior_z
        print(diff)


