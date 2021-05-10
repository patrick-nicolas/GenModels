from unittest import TestCase


import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from cnn.convmodel import ConvModel
from cnn.convneuralblock import ConvNeuralBlock
from cnn.deconvneuralblock import DeConvNeuralBlock
from cnn.deconvmodel import DeConvModel
from nnet.hyperparams import HyperParams
from autoencoder.convvaemodel import ConvVAEModel
from autoencoder.convvae import ConvVAE
from autoencoder.variationalneuralblock import VariationalNeuralBlock


class TestConvVAE(TestCase):
    def test_train_and_eval(self):
        try:
            input_channels = 1     # Gray colors 1 feature
            in_channels = 5
            output_channels = 10   # 10 digits => 10 features/classes
            conv_2d_model = TestConvVAE.__create_2d_conv_model("Conv2d", input_channels, in_channels, output_channels)

            latent_size = 16
            fc_hidden_dim = 22
            flatten_input = 90
            variational_block = VariationalNeuralBlock(flatten_input, fc_hidden_dim, latent_size)

            de_conv_2d_model = TestConvVAE.__create_de_2d_conv_model('DeConv2d', output_channels, in_channels, input_channels)
            conv_vae_model = ConvVAEModel('Conv LinearVAE', conv_2d_model, de_conv_2d_model, variational_block)

            lr = 0.001
            momentum = 0.9
            epochs = 20
            optim_label = 'adam'
            batch_size = 14
            early_stop_patience = 3
            loss_function = torch.nn.BCELoss(reduction='sum')
            hyper_params = HyperParams(lr, momentum, epochs, optim_label, batch_size, early_stop_patience, loss_function)

            # Step 3: Load data set
            train_loader, test_loader = TestConvVAE.__load_data(batch_size)
            conv_vae = ConvVAE(conv_vae_model, hyper_params, None)
            print(repr(conv_vae))
            conv_vae._train_and_eval(train_loader, test_loader)
        except Exception as e:
            self.fail(str(e))

        # --------------  Supporting methods -------------------

    @staticmethod
    def __create_2d_conv_model(model_id: str, input_channels: int, in_channels: int, output_channels: int) -> ConvModel:
        conv_neural_block_1 = TestConvVAE.__create_2d_conv_block(
            2,
            input_channels,
            in_channels,
            torch.nn.LeakyReLU(0.2),
            False,
            0)
        conv_neural_block_2 = TestConvVAE.__create_2d_conv_block(
            2,
            in_channels,
            in_channels*2,
            torch.nn.LeakyReLU(0.2),
            False,
            1)
        conv_neural_block_3 = TestConvVAE.__create_2d_conv_block(
            2,
            in_channels*2,
            output_channels,
            torch.nn.LeakyReLU(0.2),
            False,
            1)
        return ConvModel(
            model_id,
            2,
            [conv_neural_block_1, conv_neural_block_2, conv_neural_block_3],
            None)

    @staticmethod
    def __create_2d_conv_block(
                dim: int,
                in_channels: int,
                out_channels: int,
                activation: torch.nn.Module,
                batch_norm: bool,
                padding: int) -> ConvNeuralBlock:
        kernel_size = 4
        max_pooling_kernel = -1
        bias = False
        flatten = False
        stride = 2
        return ConvNeuralBlock(
                dim,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                batch_norm,
                max_pooling_kernel,
                activation,
                bias,
                flatten)

    @staticmethod
    def __create_de_2d_conv_model(model_id: str, in_channels: int, hidden_dim: int, output_size: int) -> DeConvModel:
        de_conv_neural_block_1 = TestConvVAE.__create_2d_de_conv_block(
            2,
            in_channels,
            hidden_dim * 4,
            torch.nn.ReLU(),
            False,
            0)
        de_conv_neural_block_2 = TestConvVAE.__create_2d_de_conv_block(
            2,
            hidden_dim * 4,
            hidden_dim*2,
            torch.nn.ReLU(),
            False,
            1)
        de_conv_neural_block_3 = TestConvVAE.__create_2d_de_conv_block(
            2,
            hidden_dim*2,
            output_size,
            torch.nn.Sigmoid(),
            False,
            1)
        return DeConvModel(model_id, 2, [de_conv_neural_block_1, de_conv_neural_block_2, de_conv_neural_block_3])

    @staticmethod
    def __create_2d_de_conv_block(
            dim: int,
            in_channels: int,
            out_channels: int,
            activation: torch.nn.Module,
            batch_norm: bool,
            padding: int) -> DeConvNeuralBlock:

        kernel_size = 4
        bias = False
        stride = 2
        return DeConvNeuralBlock(
                dim,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                batch_norm,
                activation,
                bias)

    @staticmethod
    def __load_data(batch_size: int) -> (DataLoader, DataLoader):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        train_loader = DataLoader(
            MNIST('../../data/', train=True, download=True, transform=transform),
            batch_size=batch_size,
            shuffle=True)
        test_loader = DataLoader(
            MNIST('../../data/', train=False, download=True, transform=transform),
            batch_size=batch_size,
            shuffle=False)
        return train_loader, test_loader

