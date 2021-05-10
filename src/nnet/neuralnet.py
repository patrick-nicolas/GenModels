__author__ = "Patrick Nicolas"

import torch
import constants
from torch.utils.data import Dataset, DataLoader
from nnet.hyperparams import HyperParams
from nnet.earlystoplogger import EarlyStopLogger
from util.plottermixin import PlotterMixin, PlotterParameters
from util.s3util import IOUtil
from abc import abstractmethod

"""
    Generic Neural Network abstract class. There are 2 version of train and evaluation
    - _train_and_evaluate Training and evaluation from a pre-configure train loader
    -  train_and_evaluate Training and evaluation from a raw data set
    The method transform_label has to be overwritten in the inheriting classes to support
    transformation/conversion of labels if needed.
    The following methods have to be overwritten in derived classes
    - transform_label Transform the label tensor if necessary
    - model_label Model identification
    
    :param hyper_params: Training parameters
    :type hyper_params: nnet.hyperparams.HyperParams
"""


class NeuralNet(PlotterMixin):
    def __init__(self, hyper_params: HyperParams, debug):
        self.hyper_params = hyper_params
        self.debug = debug


    '''
        Override the abstract method defined in NeuralNet class and return the original labels
        :param features: List of features tensors
        :type features: lst
        :param labels: List of label tensors
        :type labels: lst
    '''
    @abstractmethod
    def apply_debug(self, features: list, labels: list, title: str):
        pass

    @abstractmethod
    def model_label(self) -> str:
        pass

    '''
        Train and evaluation of a neural network given a data loader for a training set, a 
        data loader for the evaluation/test set and a encoder_model. The weights of the various linear modules
        (neural_blocks) will be initialize if self.hyper_params.
        :param train_loader: Data loader for the training set
        :type train_loader: torch.data.utils.DataLoader
        :param test_loader: Data loader for the valuation set
        :type test_loader: torch.data.utils.DataLoader
        :param model: Torch encoder_model such as ConvNet...
        :type model: torch.nn.Module
    '''
    def _train_and_eval(self, train_loader: DataLoader, test_loader: DataLoader, model: torch.nn.Module):
        self.hyper_params.initialize_weight(model.modules())

        # Create a train loader from this data set
        optimizer = self.hyper_params.optimizer(model)
        early_stop_logger = EarlyStopLogger(self.hyper_params.early_stop_patience)

        # Train and evaluation process
        for epoch in range(self.hyper_params.epochs):
            # Set training mode and execute training
            train_loss = self.__train(optimizer, epoch, train_loader, model)
            # Set evaluation mode and execute evaluation
            eval_loss = self.__eval(epoch, test_loader, model)
            early_stop_logger(epoch, train_loss, eval_loss, -1.0)
        # Generate summary
        early_stop_logger.summary(self.__plotting_params())
        del early_stop_logger


    '''
          Train and evaluation of a neural network given a data set
          :param dataset: Data set containing both training and evaluation set
          :type dataset: torch.data.utils.Dataset
          :param encoder_model: Torch encoder_model such as ConvNet...
          :type encoder_model: torch.nn.Module
      '''
    def train_and_eval(self, dataset: Dataset, model: torch.nn.Module):
        # Create a train loader from this data set
        train_loader, test_loader = NeuralNet.init_data_loader(self.hyper_params.batch_size ,dataset)
        NeuralNet._train_and_eval(self, train_loader, test_loader, model)

    @staticmethod
    def forward(features: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        with torch.no_grad():
            try:
                return model(features)
            except RuntimeError as e:
                IOUtil.log_info(e)
            except AttributeError as e:
                IOUtil.log_info(e)
            except Exception as e:
                IOUtil.log_info(e)


    @staticmethod
    def init_data_loader(batch_size: int, dataset: Dataset) -> (DataLoader, DataLoader):
        torch.manual_seed(42)

        _len = len(dataset)
        train_len = int(_len * constants.train_eval_ratio)
        test_len = _len - train_len
        train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len])

        # Finally initialize the training and test loader
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True)
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=True)
        return train_loader, test_loader


                # ----------------
                #  Private methods
                # -----------------

    def __train(self,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                train_loader: DataLoader,
                model: torch.nn.Module) -> float:
        total_loss = 0
        model.train()
        # Initialize the gradient for the optimizer
        loss_function = self.hyper_params.loss_function

        for idx, (features, labels) in enumerate(train_loader):
            try:
                # Reset the gradient to zero
                for params in model.parameters():
                    params.grad = None

                self.apply_debug(features, labels, self.model_label())
                predicted = model(features)  # Call forward - prediction
                loss = loss_function(predicted, labels)
                # Set back propagation
                loss.backward(retain_graph=True)
                total_loss += loss.data
                optimizer.step()
            except RuntimeError as e:
                IOUtil.log_info(str(e))
            except AttributeError as e:
                IOUtil.log_info(str(e))
            except Exception as e:
                IOUtil.log_info(str(e))

        average_loss = total_loss / len(train_loader)
        IOUtil.log_info(f'Training average loss for epoch {str(epoch)} is {average_loss}')
        return average_loss

    def __eval(self, epoch: int, test_loader: DataLoader, model: torch.nn.Module) -> float:
        total_loss = 0
        loss_func = self.hyper_params.loss_function
        model.eval()
        with torch.no_grad():
            for idx, (features, labels) in enumerate(test_loader):
                try:
                    self.apply_debug(features, labels, self.model_label())
                    predicted = model(features)
                    loss = loss_func(predicted, labels)
                    total_loss += loss.data
                except RuntimeError as e:
                    IOUtil.log_info(e)
                except AttributeError as e:
                    IOUtil.log_info(e)
                except Exception as e:
                    IOUtil.log_info(e)

        average_loss = total_loss / len(test_loader)
        IOUtil.log_info(f"Evaluation: average loss for epoch {str(epoch)} is {average_loss}")
        return average_loss

    def __plotting_params(self) -> list:
        return [
            PlotterParameters(self.hyper_params.epochs, '', 'training loss', self.model_label()),
            PlotterParameters(self.hyper_params.epochs, 'epoch', 'eval loss', '')
        ]

    def __repr__(self) -> str:
        return repr(self.hyper_params)