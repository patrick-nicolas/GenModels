__author__ = "Patrick Nicolas"

import torch
from pandas import json_normalize
import pandas as pd
import numpy as np
from features.loader import Loader
from sklearn.feature_extraction.text import TfidfVectorizer
from features.basefeaturizer import BaseFeaturizer
import constants

"""
    Input data format (JSON) S3: nlp/$dataSource
    {
      "context": {
        "claimType": "C1",
        "sectionMode": "",
        "sectionGroups": [],
        "age": 22,
        "gender": "F",
        "taxonomy": "radiology",
        "placeOfService": "OC",
        "dateOfService": "2021-02-03",
        "EMCode": "",
        "EMRCpts": [
          {"order": 0, "cpt": "77067", "modifiers": ["26"], "icds": [], "quantity": 1, "unit": "UN"}
        ],
        "EMRIcds": [],
        "providerId": "71",
        "patientId": "1027636",
        "sectionHeaders": [],
        "planId": "40/7/utrad-mammo"
      },
      "bagOfCodes": [
        {
          "noteId": "14150dff-155e-449e-82c7-e6614c31d7ac",
          "codes": [
            {"codeId": "77067", "codeSet": "CPT", "mentions": [], "source": "DM", "score": 1.0, "sections": []},
            {"codeId": "77063", "codeSet": "CPT", "mentions": [], "source": "DM", "score": 1.0, "sections": []},
            {"codeId": "26", "codeSet": "MOD", "mentions": [], "source": "DM", "score": 1.0, "sections": []},
            {"codeId": "GC", "codeSet": "MOD", "mentions": [], "source": "DM", "score": 1.0, "sections": []},
            {"codeId": "7025F", "codeSet": "QC", "mentions": [], "source": "DM", "score": 1.0, "sections": []},
            {"codeId": "Z12.31", "codeSet": "ICD10", "mentions": [], "source": "DM", "score": 1.0, "sections": []}
          ]
        }
      ]
    }
"""


class MentionRawFeaturizer(object):
    def __init__(self, df: pd.DataFrame):
        bag_of_codes_df = df['bagOfCodes']
        corpus = []
        for bag_of_codes in bag_of_codes_df:
            note_and_codes_df = json_normalize(bag_of_codes)
            codes_df = note_and_codes_df['codes']
            mention_names = [mention['name'].lower() for code in codes_df for c in code for mention in c['mentions']]
            corpus.append(' '.join(mention_names))
            del mention_names

        tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(corpus)
        self.tokens = tfidf_vectorizer.get_feature_names()
        self.count = len(corpus)
        del corpus
        self.tfidf_df = pd.DataFrame(
                data = tfidf.toarray(),
                index = np.arange(self.count),
                columns=self.tokens)

    def totensor(self) -> torch.Tensor:
        return torch.from_numpy(self.tfidf_df.values, device = constants.torch_device)

    def vocabulary(self) -> list:
        return list(set(self.tokens))


if __name__ == "__main__":
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    s3_folder = 'nlp/test/7/6'
    loader = Loader(s3_folder, constants.default_s3_bucket_name)
    featurizer = MentionRawFeaturizer(loader.df, torch_device)
    print(featurizer.tfidf_df)
    print(featurizer.totensor())
    print(featurizer.vocabulary())




