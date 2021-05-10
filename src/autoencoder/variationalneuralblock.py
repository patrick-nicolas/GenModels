__author__ = "Patrick Nicolas"

import torch
from torch.autograd import Variable
from util.ioutil import IOUtil


"""
    Variational neural block used to implement the variational connections with 4 components
    Full connected flattening layer
    Mean layer 
    log-variance layer
    sampling layer
    :param input_size; Size of the flattening layer
    :param hidden_dim: Size of the input layer for the mean and variance layers
    :param latent_size: Size/dimension of the latent space
"""


class VariationalNeuralBlock(torch.nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, latent_size: int):
        super(VariationalNeuralBlock, self).__init__()
        self.fc = torch.nn.Linear(input_size, hidden_dim)
        self.mu = torch.nn.Linear(hidden_dim, latent_size)
        self.log_var = torch.nn.Linear(hidden_dim, latent_size)
        self.sampler_fc = torch.nn.Linear(latent_size, input_size)

    def __repr__(self):
        return f'   {repr(self.fc)}\n   {repr(self.mu)}\n   {repr(self.log_var)}\n   {repr(self.sampler_fc)}'

    def in_channels(self):
        return self.fc.in_features

    @classmethod
    def re_parametrize(cls, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = log_var.mul(0.5).exp_()
        std_dev = std.data.new(std.size()).normal_()
        eps = Variable(std_dev)
        return eps.mul_(std).add_(mu)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        IOUtil.size(x, 'input variational')
        x = self.fc(x)
        IOUtil.size(x, 'fc variational')
        mu = self.mu(x)
        IOUtil.size(mu, 'mu variational')
        log_var = self.log_var(x)
        z = VariationalNeuralBlock.re_parametrize(mu, log_var)
        IOUtil.size(z, 'z variational')
        return self.sampler_fc(z), mu, log_var





