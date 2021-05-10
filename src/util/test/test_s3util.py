
from collections import OrderedDict
from unittest import TestCase
import pandas as pd
import constants
from util.s3util import S3Util
import torch


class TestS3Util(TestCase):
    def test_write_ordered_dict(self):
        s3_folder = 'temp/models/test-1'
        ordered_dict = OrderedDict([('l0.weight', torch.tensor([[0.1400, 0.4563, -0.0271, -0.4406],
                                           [-0.3289, 0.2827, 0.4588, 0.2031]])),
                     ('l0.bias', torch.tensor([0.0300, -0.1316])),
                     ('l1.weight', torch.tensor([[0.6533, 0.3413]])),
                     ('l1.bias', torch.tensor([-0.1112]))])

        s3_util = S3Util(constants.default_s3_bucket_name, s3_folder, False)
        s3_util.write_ordered_dict(ordered_dict)

    def test_read_ordered_dict(self):
        s3_folder = 'temp/models/test-1'
        s3_util = S3Util(constants.default_s3_bucket_name, s3_folder, False)
        loaded_ordered_dict = s3_util.read_ordered_dict()
        print(str(loaded_ordered_dict))

    def test_write_value(self):
        value = 98
        s3_folder = 'temp/models/test-2'
        s3_util = S3Util(constants.default_s3_bucket_name, s3_folder, False)
        s3_util.write_value(str(value), 'val')

    def test_read_value(self):
        s3_folder = 'temp/models/test-2'
        s3_util = S3Util(constants.default_s3_bucket_name, s3_folder, False)
        new_value = s3_util.read_value('val')
        print(f'New value {new_value}')

    def test_dataframe(self):
        data = [{'noteId': "hhh", 'terms': ["bbb"]}, {'noteId': "lll", 'terms': ["ccc"]}]
        df = pd.DataFrame(data, columns=['noteId', 'terms'])
        terms_df = df['terms']
        print(terms_df)
        s3_folder_name = "reports/embeddedLayer/test/7/3/terms"
        s3_util = S3Util(constants.default_s3_bucket_name, s3_folder_name, False)
        df = s3_util.read_dataframe(['noteId', 'terms'], 'terms')
        print(df)

    def test_to_json(self):
        s3_folder_name = "reports/embeddedLayer/test/7/3/terms"
        s3_util = S3Util(constants.default_s3_bucket_name, s3_folder_name, False)
        records = s3_util.read_json()
        print(str(records))
