__author__ = "Patrick Nicolas"

import torch
import constants
from torch import nn
from nnet.neuralmodel import NeuralModel
from dffn.dffnet import DFFNet
from cnn.deconvnet import DeConvNet

"""
    Generator using Feed-forward network model. There are three constructors
    - Default constructor using any give neural model 
    - De-convolutional model build_de_conv_model
    - Deep feed forward network build_feed_forward
    
    :param neural_model: Neural network model (i.e. Convolutional net, Feed-forward net)
    :type neural_model: A sub-class of nnet.neuralmodel.NeuralModel
"""


class GANGenerator(nn.Module):
    def __init__(self, neural_model: NeuralModel, dim_z: int):
        super(GANGenerator, self).__init__()
        self.model = neural_model.get_model()
        self.dim_z = dim_z

    '''
        Alternative constructor for generator using a de convolutional neural model
        The convolutional parameters are: kernel_size, stride, padding, batch_norm, activation.
        :param model_id: Identifier for the model used a generator
        :param dim: Dimension of the convolution (1 time series, 2 images, 3 video..)
        :param input_size: Size of the latent space
        :param hidden_dim: Size of the intermediate blocks
        :param output_size: Number of output channels
        :param params: List of convolutional parameters {kernel_size, stride, padding, batch_norm,  activation}
        :returns: Instance of GAN generator
    '''
    @classmethod
    def build_from_de_conv(cls,
                           model_id: str,
                           dim: int,
                           z_dim: int,
                           hidden_dim: int,
                           out_dim: int,
                           params: list) -> NeuralModel:
        de_conv_model = DeConvNet.build(model_id, dim, z_dim, hidden_dim, out_dim, params)
        return cls(de_conv_model, z_dim)


    '''
        Build a feed-forward network model as GAN generator
        :param model_id: Identifier for the decoder
        :param input_size: Size of the connected input layer
        :param hidden_dim: Size of the last hidden layer. The size of previous layers are halved from the 
                   previous layer
        :param output_size: Size of the output layer
        :param dff_params: List of parameters tuple{ (activation_func, drop-out rate)
        :returns: Instance of GAN generator
       '''
    @classmethod
    def build_from_dff(cls,
                       model_id: str,
                       input_size: int,
                       hidden_dim: int,
                       output_size: int,
                       params: list) -> NeuralModel:
        dff_encoder_model = DFFNet.build_encoder(model_id, input_size, hidden_dim, output_size, params)
        return cls(dff_encoder_model, output_size)

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        return repr(self.model)

    '''
        Create noise tensor for the latent space
        :param num_samples: Number of random samples from the latent space
        :param input_size: Dimension of the latent sapce
        :returns: Torch tensor
    '''
    def noise(self, num_samples: int) -> torch.Tensor:
        return torch.randn(num_samples, self.z_dim, device=constants.torch_device)