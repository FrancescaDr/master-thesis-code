from typing import Dict, Iterable, Optional, Tuple, Union
from scvi.module.base import LossRecorder, auto_move_data

from scvi.module import TOTALVAE

from src_trainer.my_base_component import Encoder_cellPMVI

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.distributions import (
    NegativeBinomial,
    NegativeBinomialMixture,
)

from scvi.nn import DecoderTOTALVI, one_hot

class cellPMVAE_CITESEQ(TOTALVAE):

    def __init__(
            self,
            n_input_genes: int,
            n_input_proteins: int,
            n_batch: int = 0,
            n_labels: int = 0,
            n_hidden: int = 256,
            n_latent: int = 20,
            n_layers_encoder: int = 2,
            n_layers_decoder: int = 1,
            n_continuous_cov: int = 0,
            n_cats_per_cov: Optional[Iterable[int]] = None,
            dropout_rate_decoder: float = 0.2,
            dropout_rate_encoder: float = 0.2,
            gene_dispersion: str = "gene",
            protein_dispersion: str = "protein",
            log_variational: bool = True,
            gene_likelihood: str = "nb",
            latent_distribution: str = "normal",
            protein_batch_mask: Dict[Union[str, int], np.ndarray] = None,
            encode_covariates: bool = True,
            protein_background_prior_mean: Optional[np.ndarray] = None,
            protein_background_prior_scale: Optional[np.ndarray] = None,
            use_size_factor_key: bool = False,
            use_observed_lib_size: bool = True,
            library_log_means: Optional[np.ndarray] = None,
            library_log_vars: Optional[np.ndarray] = None,
            use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
            use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
    ):
        super(TOTALVAE, self).__init__()
        self.gene_dispersion = gene_dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_input_genes = n_input_genes
        self.n_input_proteins = n_input_proteins
        self.protein_dispersion = protein_dispersion
        self.latent_distribution = latent_distribution
        self.protein_batch_mask = protein_batch_mask
        self.encode_covariates = encode_covariates
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_means is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        # parameters for prior on rate_back (background protein mean)
        if protein_background_prior_mean is None:
            if n_batch > 0:
                self.background_pro_alpha = torch.nn.Parameter(
                    torch.randn(n_input_proteins, n_batch)
                )
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins, n_batch), -10, 1)
                )
            else:
                self.background_pro_alpha = torch.nn.Parameter(
                    torch.randn(n_input_proteins)
                )
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins), -10, 1)
                )
        else:
            if protein_background_prior_mean.shape[1] == 1 and n_batch != 1:
                init_mean = protein_background_prior_mean.ravel()
                init_scale = protein_background_prior_scale.ravel()
            else:
                init_mean = protein_background_prior_mean
                init_scale = protein_background_prior_scale
            self.background_pro_alpha = torch.nn.Parameter(
                torch.from_numpy(init_mean.astype(np.float32))
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.log(torch.from_numpy(init_scale.astype(np.float32)))
            )

        if self.gene_dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes))
        elif self.gene_dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_batch))
        elif self.gene_dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_labels))
        else:  # gene-cell
            pass

        if self.protein_dispersion == "protein":
            self.py_r = torch.nn.Parameter(2 * torch.rand(self.n_input_proteins))
        elif self.protein_dispersion == "protein-batch":
            self.py_r = torch.nn.Parameter(
                2 * torch.rand(self.n_input_proteins, n_batch)
            )
        elif self.protein_dispersion == "protein-label":
            self.py_r = torch.nn.Parameter(
                2 * torch.rand(self.n_input_proteins, n_labels)
            )
        else:  # protein-cell
            pass

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_genes = self.n_input_genes + n_continuous_cov * encode_covariates
        n_input_proteins = self.n_input_proteins + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        self.encoder = Encoder_cellPMVI(
            n_input_genes,
            n_input_proteins,
            n_latent,
            n_layers=n_layers_encoder,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )
        self.decoder = DecoderTOTALVI(
            n_latent + n_continuous_cov,
            n_input_genes,
            self.n_input_proteins,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
        )

    @auto_move_data
    def inference(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            batch_index: Optional[torch.Tensor] = None,
            label: Optional[torch.Tensor] = None,
            n_samples=1,
            cont_covs=None,
            cat_covs=None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Internal helper function to compute necessary inference quantities.

        We use the dictionary ``px_`` to contain the parameters of the ZINB/NB for genes.
        The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
        `scale` refers to the quanity upon which differential expression is performed. For genes,
        this can be viewed as the mean of the underlying gamma distribution.

        We use the dictionary ``py_`` to contain the parameters of the Mixture NB distribution for proteins.
        `rate_fore` refers to foreground mean, while `rate_back` refers to background mean. ``scale`` refers to
        foreground mean adjusted for background probability and scaled to reside in simplex.
        ``back_alpha`` and ``back_beta`` are the posterior parameters for ``rate_back``.  ``fore_scale`` is the scaling
        factor that enforces `rate_fore` > `rate_back`.

        ``px_["r"]`` and ``py_["r"]`` are the inverse dispersion parameters for genes and protein, respectively.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        y
            tensor of values with shape ``(batch_size, n_input_proteins)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        label
            tensor of cell-types labels with shape (batch_size, n_labels)
        n_samples
            Number of samples to sample from approximate posterior
        cont_covs
            Continuous covariates to condition on
        cat_covs
            Categorical covariates to condition on
        """
        x_ = x
        y_ = y
        if self.use_observed_lib_size:
            library_gene = x.sum(1).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
            y_ = torch.log(1 + y_)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input_genes = torch.cat((x_, cont_covs), dim=-1)
            encoder_input_proteins = torch.cat((y_, cont_covs), dim=-1)
        else:
            #encoder_input = torch.cat((x_, y_), dim=-1)
            encoder_input_genes = x_
            encoder_input_proteins = y_
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        qz_m_dict, qz_v_dict, ql_m, ql_v, latent, untran_latent = self.encoder(
            encoder_input_genes, encoder_input_proteins, batch_index, *categorical_input
        )
        z = {}
        untran_z = {}
        z["gene"] = latent["z_gene"]
        z["protein"] = latent["z_protein"]
        untran_z["gene"] = untran_latent["z_gene"]
        untran_z["protein"] = untran_latent["z_protein"]
        untran_l = untran_latent["l"]
        if not self.use_observed_lib_size:
            library_gene = latent["l"]

        if n_samples > 1:
            for mod in ["gene", "protein"]:
                qz_m = qz_m_dict[mod]
                qz_v = qz_v_dict[mod]
                qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
                qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
                untran_z[mod] = Normal(qz_m, qz_v.sqrt()).sample()
                z[mod] = self.encoder.z_transformation(untran_z[mod])
                ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
                ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
                untran_l = Normal(ql_m, ql_v.sqrt()).sample()
                if self.use_observed_lib_size:
                    library_gene = library_gene.unsqueeze(0).expand(
                        (n_samples, library_gene.size(0), library_gene.size(1))
                    )
                else:
                    library_gene = self.encoder.l_transformation(untran_l)

        # Background regularization
        if self.gene_dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_r = torch.exp(py_r)
        if self.n_batch > 0:
            py_back_alpha_prior = F.linear(
                one_hot(batch_index, self.n_batch), self.background_pro_alpha
            )
            py_back_beta_prior = F.linear(
                one_hot(batch_index, self.n_batch),
                torch.exp(self.background_pro_log_beta),
            )
        else:
            py_back_alpha_prior = self.background_pro_alpha
            py_back_beta_prior = torch.exp(self.background_pro_log_beta)
        self.back_mean_prior = Normal(py_back_alpha_prior, py_back_beta_prior)

        return dict(
            qz_m=qz_m_dict,
            qz_v=qz_v_dict,
            z=z,
            untran_z=untran_z,
            ql_m=ql_m,
            ql_v=ql_v,
            library_gene=library_gene,
            untran_l=untran_l,
        )

    @auto_move_data
    def generative(
            self,
            z: torch.Tensor, # dictionary of torch tensors
            library_gene: torch.Tensor,
            batch_index: torch.Tensor,
            label: torch.Tensor,
            cont_covs=None,
            cat_covs=None,
            size_factor=None,
            transform_batch: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:

        px_dict, py_dict, log_pro_back_mean_dict = {}, {}, {}
        z_dict = z

        for mod in ["gene", "protein"]:
            z = z_dict[mod]
            decoder_input = z if cont_covs is None else torch.cat([z, cont_covs], dim=-1)
            if cat_covs is not None:
                categorical_input = torch.split(cat_covs, 1, dim=1)
            else:
                categorical_input = tuple()

            if transform_batch is not None:
                batch_index = torch.ones_like(batch_index) * transform_batch

            if not self.use_size_factor_key:
                size_factor = library_gene

            px_, py_, log_pro_back_mean = self.decoder(
                decoder_input, size_factor, batch_index, *categorical_input
            )

            if self.gene_dispersion == "gene-label":
                # px_r gets transposed - last dimension is nb genes
                px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
            elif self.gene_dispersion == "gene-batch":
                px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
            elif self.gene_dispersion == "gene":
                px_r = self.px_r
            px_r = torch.exp(px_r)

            if self.protein_dispersion == "protein-label":
                # py_r gets transposed - last dimension is n_proteins
                py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
            elif self.protein_dispersion == "protein-batch":
                py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
            elif self.protein_dispersion == "protein":
                py_r = self.py_r
            py_r = torch.exp(py_r)

            px_["r"] = px_r
            py_["r"] = py_r

            px_dict[mod] = px_
            py_dict[mod] = py_
            log_pro_back_mean_dict[mod] = log_pro_back_mean

        return dict(
            px_dict=px_dict,
            py_dict=py_dict,
            log_pro_back_mean=log_pro_back_mean,
        )

    def loss_totalVI(self, inference_outputs, generative_outputs, mod, x, batch_index, y):
        qz_m = inference_outputs["qz_m"][mod]
        qz_v = inference_outputs["qz_v"][mod]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        px_ = generative_outputs["px_dict"][mod]
        py_ = generative_outputs["py_dict"][mod]

        if self.protein_batch_mask is not None:
            pro_batch_mask_minibatch = torch.zeros_like(y)
            for b in torch.unique(batch_index):
                b_indices = (batch_index == b).reshape(-1)
                pro_batch_mask_minibatch[b_indices] = torch.tensor(
                    self.protein_batch_mask[b.item()].astype(np.float32),
                    device=y.device,
                )
        else:
            pro_batch_mask_minibatch = None

        reconst_loss_gene, reconst_loss_protein = self.get_reconstruction_loss(
            x, y, px_, py_, pro_batch_mask_minibatch
        )

        # KL Divergence
        kl_div_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)
        if not self.use_observed_lib_size:
            n_batch = self.library_log_means.shape[1]
            local_library_log_means = F.linear(
                one_hot(batch_index, n_batch), self.library_log_means
            )
            local_library_log_vars = F.linear(
                one_hot(batch_index, n_batch), self.library_log_vars
            )
            kl_div_l_gene = kl(
                Normal(ql_m, torch.sqrt(ql_v)),
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_div_l_gene = 0.0

        kl_div_back_pro_full = kl(
            Normal(py_["back_alpha"], py_["back_beta"]), self.back_mean_prior
        )
        if pro_batch_mask_minibatch is not None:
            kl_div_back_pro = torch.zeros_like(kl_div_back_pro_full)
            kl_div_back_pro.masked_scatter_(
                pro_batch_mask_minibatch.bool(), kl_div_back_pro_full
            )
            kl_div_back_pro = kl_div_back_pro.sum(dim=1)
        else:
            kl_div_back_pro = kl_div_back_pro_full.sum(dim=1)

        return reconst_loss_gene, reconst_loss_protein, kl_div_z, kl_div_l_gene, kl_div_back_pro


    def loss(
            self,
            tensors,
            inference_outputs,
            generative_outputs,
            pro_recons_weight=1.0,  # double check these defaults
            kl_weight=1.0,
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """
        Returns the reconstruction loss and the Kullback divergences.
        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        y
            tensor of values with shape ``(batch_size, n_input_proteins)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        label
            tensor of cell-types labels with shape (batch_size, n_labels)
        Returns
        -------
        type
            the reconstruction loss and the Kullback divergences
        """
        rl_genes, rl_proteins, klds = [], [], []

        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]

        for mod in ["gene", "protein"]:

            reconst_loss_gene, reconst_loss_protein, kl_div_z, kl_div_l_gene, kl_div_back_pro = self.loss_totalVI(inference_outputs, generative_outputs, mod, x, batch_index, y)

            rl_genes.append(reconst_loss_gene)
            rl_proteins.append(reconst_loss_protein)
            klds.append(kl_div_z)

        kl_div_z_total = torch.stack(klds).mean(dim=0)
        reconst_loss_gene_total = torch.stack(rl_genes).mean(dim=0)
        reconst_loss_protein_total = torch.stack(rl_proteins).mean(dim=0)

        loss_total = torch.mean(
            reconst_loss_gene_total
            + pro_recons_weight * reconst_loss_protein_total
            + kl_weight * kl_div_z_total
            + kl_div_l_gene
            + kl_weight * kl_div_back_pro
        )

        loss = (1 / 2) * loss_total

        reconst_losses = dict(
            reconst_loss_gene=reconst_loss_gene,
            reconst_loss_protein=reconst_loss_protein,
        )
        kl_local = dict(
            kl_div_z=kl_div_z,
            kl_div_l_gene=kl_div_l_gene,
            kl_div_back_pro=kl_div_back_pro,
        )

        return LossRecorder(loss, reconst_losses, kl_local, kl_global=torch.tensor(0.0))

    @torch.no_grad()
    def sample(self, tensors, n_samples=1):
        inference_kwargs = dict(n_samples=n_samples)
        with torch.no_grad():
            inference_outputs, generative_outputs, = self.forward(
                tensors,
                inference_kwargs=inference_kwargs,
                compute_loss=False,
            )

        px_ = generative_outputs["px_dict"]["gene"]
        py_ = generative_outputs["py_dict"]["protein"]

        rna_dist = NegativeBinomial(mu=px_["rate"], theta=px_["r"])
        protein_dist = NegativeBinomialMixture(
            mu1=py_["rate_back"],
            mu2=py_["rate_fore"],
            theta1=py_["r"],
            mixture_logits=py_["mixing"],
        )
        rna_sample = rna_dist.sample().cpu()
        protein_sample = protein_dist.sample().cpu()

        return rna_sample, protein_sample