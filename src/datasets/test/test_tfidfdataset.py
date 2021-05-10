from unittest import TestCase

from datasets.tfidfdataset import TfIdfDataset
from util.ioutil import IOUtil


class TestTfIdfDataset(TestCase):
    def test___init__(self):
        try:
            s3_bucket_name = 'aideo-tech-autocoding-v1'
            s3_folder = 'test/unittest/count'
            file_extension = '.json'
            to_scale = True
            is_nested = True
            mention_count_dataset = TfIdfDataset.from_s3_column(
                s3_bucket_name,
                s3_folder,
                ['mentionCounts'],
                file_extension,
                to_scale,
                is_nested)
            IOUtil.log_info(str(mention_count_dataset.tf_idf_tensor))
        except Exception as e:
            self.fail(str(e))


