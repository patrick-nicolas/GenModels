__author__ = "Patrick Nicolas"

import torch
import constants
from gan.gangenerator import GANGenerator
from gan.gandiscriminator import GANDiscriminator
from nnet.hyperparams import HyperParams
from torch.utils.data import DataLoader

"""
    Generic Generic Adversarial Network 
    :param model_id: Model identifier
    :param gen: Generator (~ decoder)
    :param disc: Discriminator (i.e. Binary classifier)
    :param hyper_params: Hyper parameters
"""


class GAN(object):
    def __init__(self, model_id: str, gen: GANGenerator, disc: GANDiscriminator, hyper_params: HyperParams):
        self.model_id = model_id
        self.gen = gen.to(device = constants.torch_device)
        self.disc = disc.to(device = constants.torch_device)
        self.hyper_params = hyper_params
        self.display_step = 100
        self.gen_opt = self.hyper_params.optimizer(self.gen)
        self.disc_opt = self.hyper_params.optimizer(self.disc)

    '''
        Main training method for the GAN
        :param data_loader: Torch data loader
    '''
    def train(self, data_loader: DataLoader):
        mean_generator_loss = 0.0
        mean_discriminator_loss = 0.0

        for epoch in range(self.hyper_params.epochs):
            for real, _ in data_loader:
                mean_discriminator_loss += self.__discriminate(real)
                mean_generator_loss += self.__generate(real)

            # ---------- Private supporting methods ----------------

    def __discriminate(self, real: torch.Tensor) -> float:
        # Reset gradient to zero
        for params in self.disc.parameters():
            params.grad = None

        fake = self.__set_noise(real)

        disc_fake_pred = self.disc(fake.detach())
        disc_fake_loss = self.hyper_params.loss_function(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = self.disc(real)
        disc_real_loss = self.hyper_params.loss_function(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = 0.5*(disc_fake_loss + disc_real_loss)

        # Compute the loss in the discriminator/classifier
        mean_disc_loss = disc_loss.item() / self.display_step
        # Update gradients
        disc_loss.backward(retain_graph = True)
        # Update optimizer
        self.disc_opt.step()
        return mean_disc_loss


    def __generate(self, real: torch.Tensor) -> float:
        # Reset gradient to zero
        for params in self.gen.parameters():
            params.grad = None

        # Generate noise
        fake  = self.__set_noise(real)
        disc_fake_pred = self.disc(fake)

        # It is assumed that this is
        gen_loss = self.hyper_params.loss_function(disc_fake_pred, torch.ones_like(disc_fake_pred))
        mean_gen_loss = gen_loss.item() / self.display_step
        gen_loss.backward()
        self.gen_opt.step()

        # Keep track of the average generator loss
        return mean_gen_loss

    def __set_noise(self, real: torch.Tensor) -> torch.Tensor:
        fake_noise = self.gen.noise(len(real))
        return self.gen(fake_noise)

