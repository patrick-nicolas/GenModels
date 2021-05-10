__author__ = "Patrick Nicolas"

import torch
from pandas import json_normalize
import pandas as pd
from features.loader import Loader
from sklearn.preprocessing import MultiLabelBinarizer
from features.basefeaturizer import BaseFeaturizer
import constants


def key(lst):
    return lst.head


class CodeFeaturizer(BaseFeaturizer):
    def __init__(self, input_df: pd.DataFrame):
        super(CodeFeaturizer, self).__init(torch_device)
        bag_of_codes_df = input_df['bagOfCodes']

        codes_list = []
        for bag_of_codes in bag_of_codes_df:
            note_and_codes_df = json_normalize(bag_of_codes)
            codes_df = note_and_codes_df['codes']
            codes = [(code['codeId'], code['source'].split(',')) for codes in codes_df for code in codes]
            codes_list.append(codes)

        self.count = len(codes_list)
        encoder = MultiLabelBinarizer()
        features_val = encoder.fit_transform(codes_list)
        self.features_df = pd.DataFrame(features_val)
        del codes_list

    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.features_df.values, device = constants.torch_device)





if __name__ == "__main__":
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lst = ['B', 'C', 'A']
    sorted = sorted(lst)
    s3_bucket_name = 'aideo-tech-autocoding-v1'
    s3_folder = 'nlp/test/7/6'
    loader = Loader(s3_folder, s3_bucket_name)

    featurizer = CodeFeaturizer(loader.df, torch_device)
    print(featurizer.features_df)
    print(featurizer.to_tensor())




