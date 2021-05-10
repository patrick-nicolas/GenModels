__author__ = "Patrick Nicolas"

import torch
from autoencoder.convvaemodel import ConvVAEModel
from nnet.hyperparams import HyperParams
from util.ioutil import IOUtil
from autoencoder.vae import VAE

"""
    Convolutional variational auto-encoder have to specializes the loss function unique to convolutional networkds
    Design patterns: Bridge

    :param conv_vae_model: A variational auto-encoder model implemented as PyTorch Module
    :type conv_vae_model: cnn.convvaemodel.ConvVAEModel
    :param hyper_params: Instance of hyper-parameters for this variational auto-encoder
    :type hyper_params: nnet.hyperparams.HyperParams
    :param debug: Optional debugging method
"""


class ConvVAE(VAE):
    def __init__(self,
                 conv_vae_model: ConvVAEModel,
                 hyper_params: HyperParams,
                 debug):
        super(ConvVAE, self).__init__(conv_vae_model, hyper_params, debug)

    def loss_func(
            self,
            predicted: torch.Tensor,
            x: torch.Tensor,
            mu: torch.Tensor,
            log_var: torch.Tensor) -> float:
        criterion = self.hyper_params.loss_function
        sz = self.vae_model.input_channels()
        try:
            # Cross-entropy for reconstruction loss for binary values
            # and MSE for continuous (TF-IDF) variable
            IOUtil.size(x, 'x before loss')
            IOUtil.size(predicted, 'predict_x')
            x_value = x.view(-1, sz).squeeze(1)
            x_predict = predicted.view(-1, sz).squeeze(1)

            IOUtil.size(x_value, 'x_value')
            IOUtil.size(x_predict, 'x_predict')
            reconstruction_loss = criterion(x_predict, x_value)

            # Kullback-Leibler divergence for Normal distribution
            return VAE.compute_loss(reconstruction_loss, mu, log_var, sz)
        except RuntimeError as e:
            IOUtil.log_error(f'Runtime error {str(e)}')
            return None
        except ValueError as e:
            IOUtil.log_error(f'Value error {str(e)}')
            return None
        except KeyError as e:
            IOUtil.log_error(f'Key error {str(e)}')
            return None



