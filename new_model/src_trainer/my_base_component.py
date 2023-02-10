from typing import Callable, Iterable, List, Optional

from scvi.nn import Encoder, FCLayers

import torch
from torch import nn as nn
from torch.distributions import Normal, Laplace
import torch.nn.functional as F

import collections
from typing import Callable, Iterable, List, Optional

import torch
from torch import nn as nn
from torch.distributions import Normal



def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()

def identity(x):
    return x

class Encoder(Encoder):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
        **kwargs,
    ):
        super(Encoder, self).__init__(n_input, n_output)

        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity
        self.var_activation = torch.exp if var_activation is None else var_activation

        print("My Encoder")

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""
        The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        if self.distribution == 'lp':
            q_v = self.var_encoder(q)
            q_v = F.softmax(self.var_encoder(q), dim=-1) * q_v.size(-1) + self.var_eps
            latent = self.z_transformation(Laplace(q_m, q_v).rsample())
        else: # default
            q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
            latent = self.z_transformation(reparameterize_gaussian(q_m, q_v))
        return q_m, q_v, latent

class Encoder_cellPMVI(nn.Module):

    """
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.
    Uses two fully connected neural networks (one for RNA and one for protein).
    Based on EncoderTOTALVI implementation.

    Parameters
    ----------
    n_input_gene
        The dimensionality of the gene input (n_cells x n_genes)
    n_input_protein
        The dimensionality of the protein input (n_cells x n_proteins)
    n_output
        The dimensionality of the output (latent space)
    """

    def __init__(
            self,
            n_input_gene: int,
            n_input_protein: int,
            n_output: int,
            n_cat_list: Iterable[int] = None,
            n_layers: int = 2,
            n_hidden: int = 256,
            dropout_rate: float = 0.1,
            distribution: str = "ln",
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
    ):
        super().__init__()

        self.encoder_gene = FCLayers(
            n_in=n_input_gene,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.z_mean_encoder_gene = nn.Linear(n_hidden, n_output)
        self.z_var_encoder_gene = nn.Linear(n_hidden, n_output)

        self.encoder_protein = FCLayers(
            n_in=n_input_protein,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.z_mean_encoder_protein = nn.Linear(n_hidden, n_output)
        self.z_var_encoder_protein = nn.Linear(n_hidden, n_output)

        self.l_gene_encoder = FCLayers(
            n_in=n_input_gene,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.l_gene_mean_encoder = nn.Linear(n_hidden, 1)
        self.l_gene_var_encoder = nn.Linear(n_hidden, 1)

        self.distribution = distribution

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity

        self.l_transformation = torch.exp

    def reparameterize_transformation(self, mu, var):
        untran_z = Normal(mu, var.sqrt()).rsample()
        z = self.z_transformation(untran_z)
        return z, untran_z

    def forward(self, data_genes: torch.Tensor, data_proteins: torch.Tensor, *cat_list: int):
        r"""
        The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. latent distribution

        The dictionary ``latent`` contains the samples of the latent variables, while ``untran_latent``
        contains the untransformed versions of these latent variables. For example, the library size is log normally distributed,
        so ``untran_latent["l"]`` gives the normal sample that was later exponentiated to become ``latent["l"]``.
        The logistic normal distribution is equivalent to applying softmax to a normal sample.

        Parameters
        ----------
        data_genes
            tensor with shape ``(n_input_genes,)``
        data_proteins
            tensor with shape ``(n_input_proteins,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        6-tuple. First 4 of :py:class:`torch.Tensor`, next 2 are `dict` of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for gene latent distribution
        q_gene = self.encoder_gene(data_genes, *cat_list)
        qz_m_gene = self.z_mean_encoder_gene(q_gene)
        qz_v_gene = torch.exp(self.z_var_encoder_gene(q_gene)) + 1e-4
        z_gene, untran_z_gene = self.reparameterize_transformation(qz_m_gene, qz_v_gene)

        # Parameters for protein latent distribution
        q_protein = self.encoder_protein(data_proteins, *cat_list)
        qz_m_protein = self.z_mean_encoder_protein(q_protein)
        qz_v_protein = torch.exp(self.z_var_encoder_protein(q_protein)) + 1e-4
        z_protein, untran_z_protein = self.reparameterize_transformation(qz_m_protein, qz_v_protein)

        ql_gene = self.l_gene_encoder(data_genes, *cat_list)
        ql_m = self.l_gene_mean_encoder(ql_gene)
        ql_v = torch.exp(self.l_gene_var_encoder(ql_gene)) + 1e-4
        log_library_gene = torch.clamp(reparameterize_gaussian(ql_m, ql_v), max=15)
        library_gene = self.l_transformation(log_library_gene)

        latent = {}
        untran_latent = {}
        qz_m = {}
        qz_v = {}
        qz_m["gene"] = qz_m_gene
        qz_m["protein"] = qz_m_protein
        qz_v["gene"] = qz_v_gene
        qz_v["protein"] = qz_v_protein
        latent["z_gene"] = z_gene
        latent["z_protein"] = z_protein
        latent["l"] = library_gene
        untran_latent["z_gene"] = untran_z_gene
        untran_latent["z_protein"] = untran_z_protein
        untran_latent["l"] = log_library_gene

        return qz_m, qz_v, ql_m, ql_v, latent, untran_latent