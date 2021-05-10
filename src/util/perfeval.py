__author__ = "Patrick Nicolas"

import torch
import time

"""
    Compute the performance of the execution of a function
    :param func Function to execute and timed
"""


class PerfEval(object):
    def __init__(self, func):
        self.func = func

    def eval(self):
        torch.set_default_tensor_type(torch.FloatTensor)
        self.__time()
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            print(f'Default tensor type: {torch.get_default_dtype()}')
            self.__time()
        else:
            print(f'CUDA not available')

    def __time(self):
        start = time.time()
        self.func()
        duration = time.time() - start
        print(f'Duration {duration} for {torch.get_default_dtype()}')