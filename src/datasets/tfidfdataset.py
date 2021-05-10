__author__ = "Patrick Nicolas"

import torch
import pandas as pd
from torch.utils.data import Dataset
from features.tfidffeaturizer import TfIdfFeaturizer
from features.loader import Loader
from datasets.s3datasetloader import S3DatasetLoader
import numpy as np
import constants

"""
    Data set for the TF-IDF mention distribution 
    :param input_df Input data frame loaded from S3 bucket
    :param torch_device Device (CPU, CUDA) used in processing the dataset
"""


class TfIdfDataset(Dataset):
    def __init__(self, input_df: pd.DataFrame, to_scale: bool):
        super(TfIdfDataset, self).__init__()
        tf_idf_featurizer = TfIdfFeaturizer(input_df)
        self.column_names = tf_idf_featurizer.tokens
        tf_idf_tensor = tf_idf_featurizer.to_tensor()
        if to_scale:
            t_values = tf_idf_tensor.numpy()
            min = np.min(t_values)
            delta = np.max(t_values) - min
            normalized_values = (t_values - min) / float(delta)
            self.tf_idf_tensor = torch.from_numpy(normalized_values)
        else:
            self.tf_idf_tensor = tf_idf_tensor


    '''
        Alternative constructor using the content of a  S3 file as input. (Decorator design pattern)
        :param s3_bucket: S3 bucket containing the input data
        :param s3_folder: Path of file containing input data
        :param col_name: Optional name of columns for features and label
        :param file_extension: File extension used to filter the files from which the features and label are extracted
        :param to_scale: Flag to specify this constructor has to normalize the frequencies
        :returns: Dataset
    '''
    @classmethod
    def from_s3_column(cls,
                       s3_bucket: str,
                       s3_folder: str,
                       col_name: str,
                       file_extension: str,
                       to_scale: bool,
                       is_nested: bool) -> Dataset:
        dataset_s3_loader = S3DatasetLoader(s3_bucket, s3_folder, [col_name], is_nested, file_extension)
        return cls(dataset_s3_loader.df, to_scale)

    def __len__(self) -> int:
        return self.tf_idf_tensor.shape[0]

    def __getitem__(self, idx) -> torch.Tensor:
        try:
            return self.tf_idf_tensor[idx]
        except IndexError as e:
            print(e)
            return None


if __name__ == "__main__":
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    s3_bucket_name = 'aideo-tech-autocoding-v1'
    s3_folder = 'reports/test/7/3/mentions/count'

    loader = Loader(s3_folder, s3_bucket_name)
    print(loader.df)
    mention_count_dataset = TfIdfDataset(loader.df)
    print(f'Size of dataset: {len(mention_count_dataset)}')
    item = mention_count_dataset[1]
    print(f'Non null item\n{item[item > 0.0]}')