from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data

import numpy as np
import torch

from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial

from src_trainer.my_vae import VAE

from src_trainer.protein_vae import ProteinVAE

class cellPMVI_lp(BaseModuleClass):
    """
    Module class for MMVAE integrating RNA and protein VAE.
    Initiating multiple VAEs and defining inference, generation, loss and forward
    """

    def __init__(
            self,
            n_input_rna: int,
            n_input_pro: int,
            n_batch: int = 0,
            n_hidden: int = 128,
            n_latent: int = 10,
            dropout_rate: float = 0.1,
            gene_likelihood: str = "nb",
            n_layers: int = 1,
            loss_objective: str = 'elbo_naive',
            latent_distribution: str = 'lp',
            **model_kwargs,
    ):
        """
        Parameters:
            :param n_input_rna:
            :param n_input_pro:
            :param n_batch:
            :param n_hidden:
            :param n_latent:
            :param dropout_rate:
            :param gene_likelihood:
            :param n_layers:
            :param loss_objective:
            :param model_kwargs:

            latent_distribution
                One of

                * ``'normal'`` - Isotropic normal
                * ``'ln'`` - Logistic normal with normal params N(0, 1)
                * ``lp`` - Laplace distribution

        """

        super().__init__()

        self.loss_objective = loss_objective
        self.gene_likelihood = gene_likelihood
        self.n_latent = n_latent
        self.latent_distribution = latent_distribution

        self.rna_vae = VAE(n_input=n_input_rna,
                            n_batch=n_batch,
                            n_hidden=n_hidden,
                            n_latent=n_latent,
                            gene_likelihood=gene_likelihood,
                            dropout_rate=dropout_rate,
                            n_layers=n_layers,
                            latent_distribution=latent_distribution,
                            **model_kwargs,
                            )

        self.protein_vae = ProteinVAE(n_input=n_input_pro,
                                      n_batch=n_batch,
                                      n_hidden=n_hidden,
                                      n_latent=n_latent,
                                      gene_likelihood=gene_likelihood,
                                      dropout_rate=dropout_rate,
                                      n_layers=n_layers,
                                      latent_distribution=latent_distribution,
                                      **model_kwargs,
                                      )

        self.vaes = [self.rna_vae, self.protein_vae]

    def _get_inference_input(self, tensors):
        """Parse the dictionary to get appropriate args"""
        rna_input=self.rna_vae._get_inference_input(tensors)
        protein_input=self.protein_vae._get_inference_input(tensors)

        """
        input_dict = dict(
            rna_x=rna_input['x'], rna_batch_index=rna_input['batch_index'], rna_cont_covs=rna_input['cont_covs'], rna_cat_covs=rna_input['cat_covs'],
            pro_x=protein_input['x'], pro_batch_index=protein_input['batch_index'], pro_cont_covs=protein_input['cont_covs'], pro_cat_covs=protein_input['cat_covs'],
        )
        """

        input_dict = dict(
            rna_x=rna_input['x'], pro_x=protein_input['x'],
            batch_index=rna_input['batch_index'], cont_covs=rna_input['cont_covs'], cat_covs=rna_input['cat_covs']
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        """

        :param tensors:
        :param inference_outputs: nested dictionary in with inference outputs for both RNA and protein
        :return:
        """
        rna_input = self.rna_vae._get_generative_input(tensors, inference_outputs[0])
        protein_input = self.protein_vae._get_generative_input(tensors, inference_outputs[1])

        input_dict = dict(
            rna_z=rna_input['z'], rna_library=rna_input['library'], rna_y=rna_input['y'], rna_size_factor=rna_input['size_factor'],
            pro_z=protein_input['z'], pro_library=protein_input['library'], pro_y=protein_input['y'], pro_size_factor=protein_input['size_factor'],
            batch_index=rna_input['batch_index'], cont_covs=rna_input['cont_covs'], cat_covs=rna_input['cat_covs']
        )

        return input_dict

    @auto_move_data
    def inference(self, rna_x, pro_x, batch_index, cont_covs, cat_covs, n_samples=1):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        # Get outputs for each vae in form of: outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v, library=library)

        rna_outputs = self.rna_vae.inference(rna_x, batch_index, cont_covs, cat_covs, n_samples=n_samples) # when does this n_sample ever change???
        protein_outputs = self.protein_vae.inference(pro_x, batch_index, cont_covs, cat_covs, n_samples=n_samples)

        outputs = [rna_outputs, protein_outputs]

        return outputs

    @auto_move_data
    def generative(self,
                   rna_z, pro_z,
                   rna_library, pro_library,
                   batch_index,
                   cont_covs=None,
                   cat_covs=None,
                   rna_size_factor=None, pro_size_factor=None,
                   rna_y=None, pro_y=None,
                   transform_batch=None,):
        """
        Runs the generative model.
        Return cross modal matrix for the MMVAE
        """
        # this is hard-coded for now!!
        # RNA: with own z
        rna_out_own = self.rna_vae.generative(rna_z, rna_library, batch_index, cont_covs, cat_covs, rna_size_factor, rna_y)
        # RNA: with protein z
        rna_out_other = self.rna_vae.generative(pro_z, rna_library, batch_index, cont_covs, cat_covs, rna_size_factor, rna_y)
        # protein: with own z
        protein_out_own = self.protein_vae.generative(pro_z, pro_library, batch_index, cont_covs, cat_covs, pro_size_factor, pro_y)
        # protein: with rna z
        protein_out_other = self.protein_vae.generative(rna_z,  pro_library, batch_index, cont_covs, cat_covs, pro_size_factor, pro_y)

        # fill cross-modal matrix with likelihood values such that (diagonal: ll given own z, ll given other z's)
        px_zs = [[rna_out_own, rna_out_other],
                 [protein_out_other, protein_out_own]]

        return px_zs

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        """
        Implement loss dependend on objective
        :param tensors:
        :param inference_outputs:
        :param generative_outputs: cross_modal matrix
        :param kl_weight:
        :return:
        """
        if self.loss_objective == 'elbo_naive':
            rls, klds = self.m_elbo_naive(tensors, inference_outputs, generative_outputs, kl_weight)
            kl_local = torch.stack(klds).mean(dim=0)
            reconst_loss = torch.stack(rls).mean(dim=0)
            loss = (1 / len(self.vaes)) * (torch.mean(reconst_loss + kl_local))
        elif self.loss_objective == 'dreg_looser':
            rls, klds = self.m_dreg_looser(tensors, inference_outputs, generative_outputs, kl_weight)
            loss = 0
            reconst_loss = 0
            kl_local = 0
        else:
            raise ValueError("No valid loss objective provided.")

        return LossRecorder(loss,reconst_loss, kl_local, torch.tensor(0.0))

    def m_elbo_naive(self,
                    tensors,
                    inference_outputs,
                    generative_outputs,
                    kl_weight: float = 1.0,):
        """
        Loss function based on objective function from https://github.com/iffsid/mmvae/blob/public/src/objectives.py.
        """
        rls, klds = [], []
        for r, vae in enumerate(self.vaes):
            for d, gen_out in enumerate(generative_outputs[r]):
                loss_recorder = vae.loss(tensors, inference_outputs[r], gen_out, kl_weight)
                klds.append(loss_recorder.kl_local)
                rls.append(loss_recorder.reconstruction_loss)
        return rls, klds

    def m_dreg_looser(self,
                      tensors,
                      inference_outputs,
                      generative_outputs,
                      kl_weight: float = 1.0,):
        """
        Loss function based on objective function from https://github.com/iffsid/mmvae/blob/public/src/objectives.py:

        'DERG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
        This version is the looser bound---with the average over modalities outside the log.'

        Paramaters:
            inference_outputs: 'list' of dictionary
                contains dictionary with output variables for each VAE
            generative_outputs: '2d-list' of form n_vae x n_vae
                diagonal entries: ouputs from own modality
                off-diagonal entries: outputs from other modality
        """
        for r, vae in enumerate(self.vaes):
            zs = [zss[e][''] for e, zzs in enumerate(generative_outputs)]
            lpz = log_prob(zss[r]).sum(-1)
            #lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
            lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_]))
            lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                         .mul(model.vaes[d].llik_scaling).sum(-1)
                     for d, px_z in enumerate(px_zs[r])]
            lpx_z = torch.stack(lpx_z).sum(0)
            lw = lpz + lpx_z - lqz_x
            lws.append(lw)
        return torch.stack(lws), torch.stack(zss)



    @torch.no_grad()
    def sample(
            self,
            tensors,
            n_samples=1,
            library_size=1,
    ) -> np.ndarray:
        r"""
        Generate observation samples from the posterior predictive distribution.

        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.

        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell
        library_size
            Library size to scale scamples to

        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        inference_kwargs = dict(n_samples=n_samples)
        inference_outputs, px_zs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        exprs_list = []
        for e, row in enumerate(px_zs):
            for d, generative_outputs in enumerate(row):
                if e == d:
                    px_r = generative_outputs["px_r"]
                    px_rate = generative_outputs["px_rate"]
                    px_dropout = generative_outputs["px_dropout"]

                    if self.gene_likelihood == "poisson":
                        l_train = px_rate
                        l_train = torch.clamp(l_train, max=1e8)
                        dist = torch.distributions.Poisson(
                            l_train
                        )  # Shape : (n_samples, n_cells_batch, n_genes)
                    elif self.gene_likelihood == "nb":
                        dist = NegativeBinomial(mu=px_rate, theta=px_r)
                    elif self.gene_likelihood == "zinb":
                        dist = ZeroInflatedNegativeBinomial(
                            mu=px_rate, theta=px_r, zi_logits=px_dropout
                        )
                    else:
                        raise ValueError(
                            "{} reconstruction error not handled right now".format(
                                self.module.gene_likelihood
                            )
                        )
                    if n_samples > 1:
                        exprs = dist.sample().permute(
                            [1, 2, 0]
                        )  # Shape : (n_cells_batch, n_genes, n_samples)
                    else:
                        exprs = dist.sample()
                    exprs_list.append(exprs.cpu())

        return exprs_list

