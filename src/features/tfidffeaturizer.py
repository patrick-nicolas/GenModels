__author__ = "Patrick Nicolas"

import torch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from features.loader import Loader
from pandas import json_normalize
import constants

"""
    Input data format (JSON) format 1
    [
       'entry': [
            {'name':'MRNUM', .... },
            {'name':'BILLING', .... },
            ....
        ],
        'entry': [
            {'name':'NAME', ... },
            {'name':'OSYK', .... },
        ],
        ....
    ]
    
    Input data format 2 
    [
       ['MRNUM', 'BILLING', ...],
       ['NAME', 'OSYK', ..]
    ]
"""


class TfIdfFeaturizer(object):
    def __init__(self, input_df: pd.DataFrame, key: str):
        mention_counts_df = input_df[0]
        self.mention_count_tensors = []
        text = []
        for mention_count in mention_counts_df:
            mention_name_count_df = json_normalize(mention_count)
            mention_name_counts = mention_name_count_df[key].values
            entry = ' '.join(mention_name_counts)
            text.append(entry)

        tf_idf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
        tf_idf = tf_idf_vectorizer.fit_transform(text)
        self.tokens = tf_idf_vectorizer.get_feature_names()
        self.count = len(text)
        del text
        self.tf_idf_df = pd.DataFrame(
            data=tf_idf.toarray(),
            index=np.arange(self.count),
            columns=self.tokens)

    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.tf_idf_df.values).float().to(constants.torch_device)

    def vocabulary(self) -> list:
        return list(set(self.tokens))