from unittest import TestCase

import torch
from util.imagetensor import ImageTensor
img_dir = '../../output/test'


class TestImageTensor(TestCase):
    def test_to_images(self):
        try:
            h = 256
            w = 480
            input_tensor1 = torch.rand((h, w), dtype=torch.float)
            input_tensor2 = torch.full((h, w), 0.5)
            image_conversion = ImageTensor(img_dir)
            image_conversion.to_images([input_tensor1, input_tensor2])
        except Exception as e:
            print(str(e))
            self.fail()

    def test_to_dataset(self):
        try:
            image_conversion = ImageTensor(img_dir)
            data_loader = image_conversion.to_dataset()
            it = iter(data_loader)
            print(it.next())
        except Exception as e:
            print(str(e))
            self.fail()