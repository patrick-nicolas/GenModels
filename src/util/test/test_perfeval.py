from unittest import TestCase

import torch
from util.perfeval import PerfEval


def perf_test_func():
    x = torch.rand(200000)
    for index in range(100000):
        x = torch.exp(x * 0.9)
        x = x * 1.34


class TestPerfEval(TestCase):

    def test_eval(self):
        try:
            eval_perf = PerfEval(perf_test_func)
            eval_perf.eval()
        except Exception as e:
            print(str(e))
            self.fail()
