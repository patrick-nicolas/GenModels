from unittest import TestCase
from datasets.s3datasetloader import S3DatasetLoader
from util.ioutil import IOUtil


class TestDatasetS3Loader(TestCase):
    def test_int(self):
        s3_bucket = 'aideo-tech-autocoding-v1'
        s3_folder =  'test/frequency'
        col_names = ['autocoded', 'frequency', 'cf']
        filter_file_extension = '.json'

        dataset_s3_loader = S3DatasetLoader(s3_bucket, s3_folder, col_names, False, filter_file_extension)
        self.assertGreater(len(dataset_s3_loader.df), 1)
        IOUtil.log_info(dataset_s3_loader['autocoded'])
