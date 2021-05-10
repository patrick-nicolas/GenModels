from unittest import TestCase
from datasets.s3datasetloader import S3DatasetLoader
from util.ioutil import IOUtil
from features.tfidffeaturizer import TfIdfFeaturizer


class TestTfIdfFeaturizer(TestCase):
    def test___init__(self):
        s3_bucket = 'aideo-tech-autocoding-v1'
        s3_folder = 'test/unittest/count'
        file_extension = '.json'
        is_nested = True
        dataset_s3_loader = S3DatasetLoader(s3_bucket, s3_folder, ['mentionCounts'], is_nested, file_extension)
        IOUtil.log_info(str(dataset_s3_loader.df))

        key = 'name'
        tf_idf_featurizer = TfIdfFeaturizer(dataset_s3_loader.df, key)
        for idx, entry in enumerate(tf_idf_featurizer.tf_idf_df.values):
            print(f'{idx}: {entry}')


