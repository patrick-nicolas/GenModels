__author__ = "Patrick Nicolas"

import torch.optim as opt

"""
Encapsulates the optimizers for the discriminator and generator network
"""


class GANOptimizer(object):
    def __init__(self, discriminator, generator, optimizer_label):
        if optimizer_label == "adam":
            self.disc_optimizer = opt.Adam(
                discriminator.parameters(),
                lr=discriminator.lr,
                momentum = discriminator.momentum)

            self.gen_optimizer = opt.Adam(
                generator.parameters(),
                lr=generator.lr,
                momentum=generator.momentum)
        else:
            self.disc_optimizer = opt.SGD(
                discriminator.parameters(),
                lr=discriminator.lr,
                momentum=discriminator.momentum)

            self.gen_optimizer = opt.SGD(
                generator.parameters(),
                lr=generator.lr,
                momentum=generator.momentum)