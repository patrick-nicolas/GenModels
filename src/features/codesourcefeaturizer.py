__author__ = "Patrick Nicolas"

import torch
import pandas
from pandas import json_normalize
from features.loader import Loader
from features.basefeaturizer import BaseFeaturizer

"""
    Featurize the code vs source 
    Input data format:
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


class CodeSourceFeaturizer(BaseFeaturizer):
    def __init__(self, input_df: pandas.DataFrame, torch_device: torch.device):
        super(CodeSourceFeaturizer, self).__init(torch_device)
        bag_of_codes_df = input_df['bagOfCodes']

        codes_list = []
        sources_list = []
        for bag_of_codes in bag_of_codes_df:
            note_and_codes_df = json_normalize(bag_of_codes)
            codes_df = note_and_codes_df['codes']
            codes = [(code['codeId'], code['source'].split(',')) for codes in codes_df for code in codes]

            for code in codes:
                _, sources = code
                for src in sources:
                    sources_list.append(src)
            codes_list.append(codes)

        all_codes = [code for codes_sources in codes_list for code, _ in codes_sources]
        all_distinct_codes = list(set(all_codes))
        code_index_map = {}
        for idx, code in enumerate(all_distinct_codes):
            code_index_map[code] = idx
        num_distinct_codes = len(all_distinct_codes)
        del all_distinct_codes

        all_distinct_sources = list(set(sources_list))
        source_index_map = {}
        for idx, source in enumerate(all_distinct_sources):
            source_index_map[source] = idx
        num_distinct_sources = len(all_distinct_sources)
        del all_distinct_sources

        self.tensors_list = []
        for code_sources in codes_list:
            tensor = torch.zeros((num_distinct_codes, num_distinct_sources), device = self.torch_device)
            for code, sources, in code_sources:
                code_idx = code_index_map[code]
                for src in sources:
                    src_idx = source_index_map[src]
                    tensor[code_idx, src_idx] = 1.0
            self.tensors_list.append(tensor)
        del codes_list, source_index_map, code_index_map


if __name__ == "__main__":
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    s3_bucket_name = 'aideo-tech-autocoding-v1'
    s3_folder = 'nlp/test/7/6'

    loader = Loader(s3_folder, s3_bucket_name)
    featurizer = CodeSourceFeaturizer(loader.df, torch_device)
    print(featurizer.tensors_list[0])




