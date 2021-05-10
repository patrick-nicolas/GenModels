__author__ = "Patrick Nicolas"

from util.plottermixin import PlotterMixin
from util.ioutil import IOUtil

"""
    Enforce early stopping for any training/evaluation pair of execution and records loss for profiling and 
    summary
    The early stopping algorithm is implemented as follows:
       Step 1: Record new minimum evaluation loss, min_eval_loss
       Step 2: If min_eval_loss < this_eval_loss Start decreasing patience count
       Step 3: If patience count < 0, apply early stopping
    Patience for the early stop is an hyper-parameter. A metric can be optionally recorded if value is >= 0.0
    
    :param patience: Number of time the eval_loss has been decreasing
    :type ratio: int
    :param min_diff: Minimum difference (min_eval_loss - lastest eval loss)
    :type min_diff: float
"""


class EarlyStopLogger(PlotterMixin):
    def __init__(self, patience: int, min_diff: float = -1e-5):
        self.patience = patience
        self.training_losses = []
        self.eval_losses = []
        self.metric = []
        self.min_loss = -1.0
        self.min_diff = min_diff

    '''
        Implement the early stop and logging of training, evaluation loss. An metric < 0.0 is not recorded
        :param epoch: Current epoch index (starting with 1)
        :type epoch: int
        :param train_loss: Current training loss
        :type train_loss: list
        :param eval_loss: Evaluation loss
        :type eval_loss: list
        :param metric: Current metric, used only if metric >= 0.0  (default -1.0)
        :type metric: float
        :returns: True if early stopping criteria is met, False otherwise
    '''
    def __call__(self, epoch: int, train_loss: float, eval_loss: float, metric: float = -1.0) -> bool:
        # Step 1. Apply early stopping criteria
        is_early_stopping = self.__evaluate(eval_loss)
        # Step 2: Record training, evaluation losses and metric
        self.__record(epoch, train_loss, eval_loss, metric)
        return is_early_stopping

    '''
        Summary with plotting capability
        :param plotter_parameters: List of plotter parameters
        :type plotter_parameters: list
    '''
    def summary(self, plotter_parameters: list):
        if self.metric:
            self.three_plot(self.training_losses, self.eval_losses, self.metric, plotter_parameters)
        else:
            self.two_plot(self.training_losses, self.eval_losses, plotter_parameters)

            # ----------------
            # Private methods
            # ----------------
    def __evaluate(self, eval_loss: float) -> bool:
        is_early_stopping = False
        if self.min_loss < 0.0:
            self.min_loss = eval_loss
        elif self.min_loss - eval_loss > self.min_diff:
            self.min_loss = eval_loss
        elif self.min_loss - eval_loss <= self.min_diff:
            self.patience =- 1
            if self.patience < 0:
                IOUtil.log_info('Early stopping')
                is_early_stopping = True
        return is_early_stopping


    def __record(self, epoch: int, train_loss: float, eval_loss: float, accuracy: float):
        if accuracy >= 0.0:
            metric_str = f', Accuracy: {accuracy}'
        else:
            metric_str = ''
        status_msg = f'Epoch: {epoch}, Train loss: {train_loss}, Eval loss: {eval_loss}{metric_str}' \
                     f'\nEval loss - Train loss: {eval_loss - train_loss}'
        IOUtil.log_info(status_msg)

        self.training_losses.append(train_loss)
        self.eval_losses.append(eval_loss)
        if accuracy >= 0.0:
            self.metric.append(accuracy)





