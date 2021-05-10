__author__ = "Patrick Nicolas"

from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

default_output_dir = '../output/images'

"""
    Conversion between a tensor into an image
    to_images Generate images from a list of tensors
    to_dataset Generate a data loader from a list of images
    
    :param images_dir: Directory containing input images
    :type images_dir: str
"""


class ImageTensor(object):
    def __init__(self, images_dir: str):
        (ImageTensor, self).__init__()
        self.images_dir = images_dir

    def to_dataset(self) -> DataLoader:
        dataset = datasets.ImageFolder(default_output_dir, transform=transforms.ToTensor())
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        return data_loader

    def to_images(self, tensors: list) -> list:
        [self._toimage(tensor, idx) for idx, tensor in enumerate(tensors)]

    def __to_image(self, tensor: torch.Tensor, idx: int):
        shapes = list(tensor.size())
        t = torch.zeros((shapes[0], shapes[1], 3), dtype=torch.uint8)
        for i in range(shapes[0]):
            for j in range(shapes[1]):
                for k in range(3):
                    if tensor[i, j] > 0.0:
                        t[i, j, k] = (tensor[i, j] * 255).int()
        data = t.numpy()
        img = Image.fromarray(data, 'RGB')
        img.save(f'{self._absolutedir()}{idx}.png')
        del img

    def __absolute_dir(self) -> str:
        return f'{default_output_dir}/{self.images_dir}/'

    '''
        Display image tensors for debugging purpose. The total number of images displayed are num_cols by 6 rows 
        :param input_data: List of torch tensor for images
        :type input_data: lst
        :param target_data: List of torch tensor for labels
        :type target_data: lst
        :param num_items: Number of items to display
        :type num_items: int
        :param title: Title of the graph
        :type title: str
    '''
    @staticmethod
    def show_image(input_data: list, target_data: list, title: str, num_items: int):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        for i in range(num_items):
            plt.subplot(3, num_items / 2, i + 1)
            plt.tight_layout()
            plt.imshow(input_data[i][0], cmap='gray', interpolation='none')
            plt.title(f'{title}: {target_data[i]}')
            plt.xticks([])
            plt.yticks([])
        fig.show()