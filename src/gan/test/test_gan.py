from unittest import TestCase

import torch
from gan.gandiscriminator import GANDiscriminator


class TestGAN(TestCase):

    def test_train(self):
        try:
            dim = 2
            z_dim = 10
            hidden_dim = 13
            out_dim = 1
            params = [
                (3, 2, 1, True, 2, torch.nn.ReLU()),
                (2, 2, 0, 2, True, torch.nn.ReLU()),
                (2, 2, 0, False, 0, torch.nn.Tanh())
            ]
            disc = GANDiscriminator.build_from_conv('De-conv classifier', dim, z_dim, hidden_dim, out_dim, params)

        except Exception as e:
            self.fail(str(e))
