__author__ = "Patrick Nicolas"

from torch import nn
from nnet.neuralmodel import NeuralModel
from cnn.convnet import ConvNet
from dffn.dffnet import DFFNet

"""
    Generic discriminator for Generative Adversarial Networks. 
    :param neural_model: Neural network model (i.e. Convolutional or Feed-forward model)
    :type neural_model: Class derived from nnet.neuralmodel.NeuralModel
"""


class GANDiscriminator(nn.Module):
    def __init__(self, neural_model: NeuralModel):
        super(GANDiscriminator, self).__init__()
        self.model = neural_model.get_model()


    '''
           Alternative constructor for discriminator using a convolutional neural model
           The convolutional parameters are: kernel_size, stride, padding, batch_norm, max_pooling_kernel, activation.
           :param model_id: Identifier for the model used a generator
           :param dim: Dimension of the convolution (1 time series, 2 images, 3 video..)
           :param input_size: Size of the latent space
           :param hidden_dim: Size of the intermediate blocks
           :param output_size: Number of output channels
           :param params: List of convolutional parameters 
                           {kernel_size, stride, padding, batch_norm, max_pooling_kernel, activation}
           :returns: Instance of GAN generator
       '''
    @classmethod
    def build_from_conv(cls,
                        model_id: str,
                        dim: int,
                        z_dim: int,
                        hidden_dim: int,
                        out_dim: int,
                        params: list) -> NeuralModel:
        conv_model = ConvNet.build(model_id, dim, z_dim, hidden_dim, out_dim, params)
        return cls(conv_model)


    '''
            Build a feed-forward network model as GAN discriminator
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
        dff_decoder_model = DFFNet.build_decoder(model_id, input_size, hidden_dim, output_size, params)
        return cls(dff_decoder_model)

    def forward(self, x):
        output = self.model(x)
        return output

