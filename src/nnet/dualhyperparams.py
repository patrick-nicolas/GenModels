__author__ = "Patrick Nicolas"

import torch
import torch.nn as nn
from nnet.hyperparams import HyperParams

"""
    Extended Hyper parameters for dual optimizer. The first optimizer is generated from the first argument 
    hyper_params, the second optimizer is generated by reusing the first set of parameters and overriding
    the learning rate, momentum and optimization type
    
    :param hyper_params: Hyper parameters for the first optimizer
    :param learning_rate2: Learning rate for the second optimizer
    :param momentum2: Momentum for the second optimizer
    :param optim_label2: Type of the second optimizer
"""


class DualHyperParams(object):
    def __init__(self, hyper_params: HyperParams, learning_rate2: float, momentum2: float, optim_label2: str):
        self.hyper_params = hyper_params
        self.learning_rate2 = learning_rate2
        self.momentum2 = momentum2
        self.optim_label2 = optim_label2

    '''
        Select the optimizer for generated from encoder_model parameters given the optimization label
        - SGD with nesterov momentum
        - Adam
        - Plain vanilla SGD
        :param model: ML model for which to extract parameters
        :returns: Two optimizers
    '''
    def optimizer(self, model: nn.Module) -> (torch.optim.Optimizer, torch.optim.Optimizer):
        optimizer1 = self.hyper_params.optimizer(model)
        second_hyper_params = self.__create_second_hyper_params()
        optimizer2 = second_hyper_params.optimizer(model)
        optimizer1, optimizer2

    def __create_second_hyper_params(self) -> HyperParams:
        return HyperParams(
            self.learning_rate2,
            self.momentum2,
            self.hyper_params.epochs,
            self.optim_label2,
            self.hyper_params.batch_size,
            self.hyper_params.early_stop_patience,
            self.hyper_params.loss_function,
            self.hyper_params.normal_weight_initialization)