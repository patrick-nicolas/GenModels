__author__ = "Patrick Nicolas"

import torch
from torch import nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input: int) -> torch.Tensor:
        return input.view(input.size(0), -1)


import constants
if __name__ == "__main__":
    X = torch.tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], dtype=torch.float32, device = constants.torch_device)
    print(X)
    Y = X.view(1, 3, 4)   # Add an extra dimension
    # [[ [1., 2., 3., 4.],
    #    [5., 6., 7., 8.],
    #    [9., 10., 11., 12.] ]]
    Z = X.view(-1, 3)   # 3 column_names and whatever number of rows
    #  [[1., 2., 3.],
    #   [4., 5., 6.],
    #   [7., 8., 9.],
    #   [10., 11., 12.]]
    W = X.view(4, -1)   # 4 rows and whatever number of cols
    # [[1., 2., 3.],
    #  [4., 5., 6.],
    #  [7., 8., 9.],
    #  [10., 11., 12.]]
    V = Y.squeeze()       # Remove the extract dimension (if == 1)
    T = Y.flatten()       # [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.]
    S = X.unsqueeze(0)    # Equivalent to view(1, 3, 4)
    #  [[[1., 2., 3., 4.],
    #    [5., 6., 7., 8.],
    #    [9., 10., 11., 12.]]]
    S = X.unsqueeze(1)    # Add extra dimension  Same as view(3, 1, 4)
    S = X.unsqueeze(2)    # Same as view(3, 4, 1)

