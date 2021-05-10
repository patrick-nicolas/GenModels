from unittest import TestCase

import torch
import constants
from datasets.numericdataset import NumericDataset
from util.ioutil import IOUtil


class TestNumericDataset(TestCase):
    def test_from_tensor_no_key(self):
        try:
            x = torch.tensor([[3, 0], [6, 1], [10, 1], [4, 0], [7, 1]], dtype=torch.float32, device=constants.torch_device)
            numeric_dataset = NumericDataset.from_tensor(x)
            print(str(numeric_dataset))
            self.assertEqual(len(numeric_dataset), len(x))
            train_input, eval_input = numeric_dataset[1]
            self.assertEqual(train_input, 6.0)
            self.assertEqual(eval_input, 1.0)
            print(train_input)
        except Exception as e:
            self.fail(str(e))

    def test_from_json_file(self):
        try:
            input_path = '../../data/requests/input_test2.json'
            numeric_dataset = NumericDataset.from_json_file(input_path)
            print(str(numeric_dataset))
        except Exception as e:
            self.fail(str(e))


    def test_from_json_file_with_key(self):
        try:
            input_path = '../../data/requests/input.json'
            numeric_dataset = NumericDataset.from_json_column(input_path, 'score', 'label')
            print(str(numeric_dataset))
            train_score, eval_score = numeric_dataset[0]
            print(str(train_score))
        except Exception as e:
            self.fail(str(e))

    def test_from_s3_files_with_key(self):
        try:
            s3_bucket = 'aideo-tech-autocoding-v1'
            s3_folder = 'test/frequency'
            col_names = ['autocoded', 'frequency', 'cf']
            filter_file_extension = '.json'
            numeric_dataset = NumericDataset.from_s3_columns(s3_bucket, s3_folder, col_names, filter_file_extension)
            IOUtil.log_info(repr(numeric_dataset))
        except Exception as e:
            self.fail(str(e))

