__author__ = "Patrick Nicolas"

import torch
from torch.utils.data import Dataset
import pandas as pd
from datasets.s3datasetloader import S3DatasetLoader
from util.ioutil import IOUtil

"""
    Simple dataset of floating point values (numeric) with the following format
        [feature1, feature2, ... featureN, label]
    The selection of features is applied during instantiation using one or more column/features names.
     The data set can be instantiated from
    - Pandas data frame  (__init__)
    - Local JSON file  (from_json_columns)
    - S3/HDFS files  (from_s3_columns)
    
    :param df: Pandas data frame
    :type df: Pandas data frame
"""


class NumericDataset(Dataset):
    # default constructor
    def __init__(self, df: pd.DataFrame):
        super(NumericDataset, self).__init__()
        self.data_df = df
        self.num_cols = len(list(df))

    '''
        Alternative constructor using a Torch Tensor as input: (Decorator design pattern). The layout of the tensor
        should be [feature, label]
        :param x: Pytorch tensor with at least two entries [feature, label]
        :type x: Tensor
        :param column_names: Optional list of column_names name
        :type column_names: List of str
        :returns: Dataset
    '''
    @classmethod
    def from_tensor(cls, x: torch.Tensor, columns: list = None) -> Dataset:
        return cls(pd.DataFrame(x, columns))


    '''
        Alternative constructor using the content of a file as input. (Decorator design pattern)
        :param input_path: Path of file containing input data
        :param feature_name: Optional feature_name to extract numerical data
        :param label_name: Optional label key
        :returns: Dataset with numerical values
    '''
    @classmethod
    def from_json_column(cls, input_path: str, feature_name: str, label_name: str) -> Dataset:
        return cls.from_json_file(input_path, [feature_name], label_name)


    '''
        Alternative constructor using the content of a file as input. (Decorator design pattern)
        :param input_path: Path of file containing input data
        :param feature_names: Optional list of feature_name to extract numerical data
        :param label_name: Optional label key
        :returns: Dataset
    '''
    @classmethod
    def from_json_columns(cls, input_path: str, feature_names: list, label_name: str) -> Dataset:
        df = pd.read_json(input_path)
        if feature_names is None:
            return cls(df)
        else:
            return cls(df[[*feature_names, label_name]])


    '''
        Alternative constructor using the content of a  S3 file as input. (Decorator design pattern)
        :param s3_bucket: S3 bucket containing the input data
        :param s3_folder: Path of file containing input data
        :param col_name: Optional name of columns for features and label
        :param file_extension: File extension used to filter the files from which the features and label are extracted
        :returns: Dataset
    '''
    @classmethod
    def from_s3_columns(cls, s3_bucket: str, s3_folder: str, col_names: list, file_extension: str) -> Dataset:
        dataset_s3_loader = S3DatasetLoader(s3_bucket, s3_folder, col_names, False, file_extension)
        return cls(dataset_s3_loader.df)


    def __len__(self):
        return self.data_df.shape[0]

    '''
        Implement the subscript operator [] for this data set. It is assume that this data set is 
        flat and each row has the following layout
            [feature1, feature2, .., label]
        :param row: index of the row
        :type row: int
        :returns: tuple features tensor, label tensor
    '''
    def __getitem__(self, row: int) -> (torch.Tensor, torch.Tensor):
        try:
            data_row = list(self.data_df.iloc[row])
            num_columns = self.data_df.shape[1]
            # Retrieve the all the column_names except the last one as feature
            features = data_row[0:num_columns-1]
            # Retrieve the last column as label
            label = data_row[num_columns-1]
            # Finally generate pair of features, label tensors
            return torch.tensor(features), torch.tensor(label)
        except IndexError as e:
            IOUtil.log_error(e)
            return None

    def shape(self) -> list:
        return self.data_df.shape

    def col_names(self) -> list:
        return list(self.data_df)

    def __repr__(self) -> str:
        return repr(self.data_df)
