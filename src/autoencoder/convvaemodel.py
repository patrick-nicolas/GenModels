__author__ = "Patrick Nicolas"

import torch
from cnn.convmodel import ConvModel
from cnn.deconvmodel import DeConvModel
from autoencoder.variationalneuralblock import VariationalNeuralBlock
from autoencoder.vae import VAE
from util.ioutil import IOUtil
from nnet.neuralmodel import NeuralModel

"""
    Convolutional model for the variational auto-encoder
    :param model_id: Identifier for the convolutional variational auto-encoder_model
    :param encoder_model: Convolutional model as encoder_model for this auto-encoder_model
    :param decoder_model: De-convolutional model as Decoder for this auto-encoder_model
    :param variational_block: Variational components
"""


class ConvVAEModel(torch.nn.Module):
    def __init__(self,
                 model_id: str,
                 encoder_model: ConvModel,
                 decoder_model: DeConvModel,
                 variational_block: VariationalNeuralBlock):
        super(ConvVAEModel, self).__init__()
        self.model_id = model_id
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.variational_block = variational_block


    def input_channels(self):
        return self.encoder_model.input_size

    def __repr__(self):
        return f'Encoder:{repr(self.encoder_model)}\n\nVariational:\n{repr(self.variational_block)}\n\nDecoder:{repr(self.decoder_model)}'


    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        IOUtil.size(x, 'input')
        x = self.encoder_model.conv_model(x)
        IOUtil.size(x, 'after encoder_model')
        shapes = list(x.shape)
        print(f' ... shape {len(shapes)}')
        batch, a, b, c = x.shape
        x = x.view(batch, -1)
        IOUtil.size(x, 'flattened')
        z, mu, log_var = self.variational_block(x)
        IOUtil.size(z, 'after variational')
        IOUtil.size(mu, 'mu')
        z = VAE.reshape_output_variation(shapes, z)
        IOUtil.size(z, 'z.view')
        z = self.decoder_model(z)
        return (z, mu, log_var)
