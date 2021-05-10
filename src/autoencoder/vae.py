__author__ = "Patrick Nicolas"

import torch
from torch.utils.data import Dataset, DataLoader
from util.ioutil import IOUtil
from nnet.hyperparams import HyperParams
from torch.optim import Adam
from util.plottermixin import PlotterParameters
from abc import abstractmethod

"""
    Base generic class for the hierarchy of Variational Auto-encoders. Specialized auto-encoder sucha
    as feed-forward and convolutional variational auto-encoder have to specify 
    Design patterns: Bridge
    
    :param vae_model: A variational auto-encoder model implemented as PyTorch Module
    :type vae_model: torch.nn.Module
    :param hyper_params: Instance of hyper-parameters for this variational auto-encoder
    :type hyper_params: nnet.hyperparams.HyperParams
    :param debug: Optional debugging method
"""


class VAE(object):
    def __init__(self, vae_model: torch.nn.Module, hyper_params: HyperParams, debug):
        self.vae_model = vae_model
        self.hyper_params = hyper_params
        self.debug = debug
        self.is_training = True

    '''
        Train and evaluation method using a data set as input
        :param input_dataset: Input dataset that contains features and labels
        :type input_dataset: torch.utils.data.Dataset
    '''
    def train_and_eval(self, input_dataset: Dataset):
        self._train_and_eval(DataLoader(dataset=input_dataset, batch_size=self.hyper_params.batch_size, shuffle=True))

    '''
        Extract the model parameters for the encoder and decoder
        :returns: pair of encoder and decoder parameters (dictionaries)
    '''
    def enc_dec_params(self) -> (dict, dict):
        return self.vae_model.encoder_model.parameters(), self.vae_model.decoder_model.parameters()

    def __repr__(self):
        return f'Model:\n{repr(self.vae_model)}\nHyper parameters:\n{repr(self.hyper_params)}'

    @abstractmethod
    def loss_func(
            self,
            predicted: torch.Tensor,
            x: torch.Tensor,
            mu: torch.Tensor,
            log_var: torch.Tensor) -> float:
        pass

    def _train_and_eval(self, train_loader: DataLoader, test_loader: DataLoader):
        torch.manual_seed(42)
        encoder_params, decoder_params = self.enc_dec_params()
        encoder_optimizer = Adam(
            encoder_params,
            lr=self.hyper_params.learning_rate,
            betas=(self.hyper_params.momentum, 0.999))
        decoder_optimizer = Adam(
            decoder_params,
            lr=self.hyper_params.learning_rate,
            betas=(self.hyper_params.momentum, 0.999))

        average_training_loss_history = []
        average_eval_loss_history = []

        for epoch in range(self.hyper_params.epochs):
            training_loss = self.__train(epoch, encoder_optimizer, decoder_optimizer, train_loader)
            average_training_loss_history.append(training_loss)
            eval_loss, mu, log_var = self.__eval(epoch, test_loader)
            average_eval_loss_history.append(eval_loss)

        plotter_parameters = PlotterParameters(self.hyper_params.epochs, 'epoch', 'training loss', 'LinearVAE')
        self.two_plot(average_training_loss_history, average_eval_loss_history, plotter_parameters)
        del average_training_loss_history, average_eval_loss_history


    @staticmethod
    def reshape_output_variation(shapes: list, z: torch.Tensor) -> torch.Tensor:
        if len(shapes) == 4:
            return z.view(shapes[0], shapes[1], shapes[2], shapes[3])
        elif len(shapes) == 3:
            return z.view(shapes[0], shapes[1], shapes[2])
        else:
            raise Exception(f'Shape {str(shapes)} for variational auto encoder should have at least 3 dimension')

    @staticmethod
    def compute_loss(reconstruction_loss: float, mu: torch.Tensor, log_var: torch.Tensor, sz: int) -> float:
        kullback_leibler = (-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())) / sz
        IOUtil.log_info(f"Reconstruction loss {reconstruction_loss} KL divergence {kullback_leibler}")
        return reconstruction_loss + kullback_leibler

            # ---------------------
            #   Private methods
            # --------------------

    def __train(
            self,
            epoch: int,
            encoder_optimizer: torch.optim.Optimizer,
            decoder_optimizer: torch.optim.Optimizer,
            data_loader: DataLoader) -> float:
        self.is_training = True
        total_loss = 0

        for idx, (features, labels) in enumerate(data_loader):
            try:
                for params in self.vae_model.parameters():
                    params.grad = None
                recon_batch, mu, latent_var = self.vae_model(features)
                loss = self.loss_func(recon_batch, features, mu, latent_var)

                loss.backward(retain_graph=True)
                total_loss += loss.data
                encoder_optimizer.step()
                decoder_optimizer.step()
            except RuntimeError as e:
                print(e)
            except AttributeError as e:
                print(e)
            except Exception as e:
                print(e)

        average_loss = total_loss / len(data_loader)
        print(f'Training average loss for epoch {str(epoch)} is {average_loss}')
        return average_loss

    def __eval(self, epoch: int, data_loader: DataLoader) -> float:
        self.is_training = False
        total_loss = 0
        with torch.no_grad():
            for idx, (features, labels) in enumerate(data_loader):
                recon_batch, mu, log_var = self.vae_model(features)
                loss = self.loss_func(recon_batch, features, mu, log_var)
                total_loss += loss.data

            average_loss = total_loss / len(data_loader)
            print(f"Evaluation: average loss for epoch {str(epoch)} is {average_loss}")
            return average_loss
