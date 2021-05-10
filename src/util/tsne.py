__author__ = "Patrick Nicolas"

import torch
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np

"""
    Apply the Stochastic Neigbhors Embedding to a two or 3 dimension embeddings.
    The main method, forward generates the appropriate image stored in 'images' directory
    and return the embedding
    :param n_components Number of principal components (dimension) of the embedding
    :param cmap Color map used for display 
    :param fig_save_dir Directory the plot is stored
    :param title Title of the plot
"""


class T_SNE(object):
    def __init__(self, n_components: int, cmap: str, fig_save_dir: str, title: str):
        assert 2 <= n_components <= 3, f'TSNE: num of components {n_components} should be [2, 3]'
        self.t_sne = TSNE(n_components = n_components)
        self.cmap = cmap
        self.fig_save_dir = fig_save_dir
        self.title = title

    '''
        Method to generate TSNE embedding with plot stored in a given directory
        :param features Input torch tensor
    '''
    def forward(self, x: torch.Tensor) -> np.array:
        # Apply the TSNE transform
        embedded = self.t_sne.fit_transform(x.detach().numpy())
        n_points = len(embedded)
        # Just random colors
        colors = np.random.randn(n_points)
        # Set up the display - plotting
        fig = plt.figure()
        if self.t_sne.n_components == 2:
            plt.scatter(embedded[:,0], embedded[:,1], c= colors, cmap= self.cmap)
            plt.colorbar()
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(*zip(*embedded[:,:2]), c=colors, cmap=self.cmap)
        plt.show()
        plt.title(self.title)
        # Save the plot
        if self.fig_save_dir:
            fig.savefig(self.fig_save_dir)
        return embedded


