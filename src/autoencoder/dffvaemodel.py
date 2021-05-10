__author__ = "Patrick Nicolas"

import torch
from dffn.dffmodel import DFFModel
from autoencoder.variationalneuralblock import VariationalNeuralBlock
from util.ioutil import IOUtil

"""
    Feed-forward model for Variational auto-encoder
    :param model_id: Identifier for the feed-forward network model for the variational auto encoder
    :param encoder_model: Feed-forward model as encoder for the variational auto encoder
    :param decoder_model: Feed-forward model as decoder for the variational auto encoder
    :param variational_block: Variational neural block
"""


class DFFVAEModel(torch.nn.Module):
    def __init__(self,
                 model_id: str,
                 encoder_model: DFFModel,
                 decoder_model: DFFModel,
                 variational_block: VariationalNeuralBlock):
        super(DFFVAEModel, self).__init__()
        self.model_id = model_id
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.variational_block = variational_block

    def __repr__(self):
        return f'Encoder:{repr(self.encoder_model)}\nVariational model:\n{repr(self.variational_block)}\nDecoder: {repr(self.decoder_model)}'

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        IOUtil.size(x, 'Input dff_vae')
        x = self.encoder_model(x)
        IOUtil.size(x, 'after encoder_model')
        batch, a = x.shape
        x = x.view(batch, -1)
        IOUtil.size(x, 'flattened')
        z, mu, log_var = self.variational_block(x)
        IOUtil.size(z, 'after variational')
        IOUtil.size(mu, 'mu')
        z = z.view(batch, a)
        IOUtil.size(z, 'z.view')
        z = self.decoder_model(z)
        return z, mu, log_var

