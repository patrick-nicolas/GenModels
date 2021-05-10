__author__ = "Patrick Nicolas"

import pandas as pd
import logging
import constants
import torch


class IOUtil(object):
    def __init__(self, path):
        self.path = path

    def to_lines(self) -> list:
        with open(self.path, 'r') as f:
            lines = f.readlines()
        return lines

    def to_text(self) -> str:
        return ''.join(self.to_lines())

    def to_json(self):
        with open(self.path, 'r') as f:
            json_content = f.readlines()
        return json_content

    def to_dataframe(self) -> pd.DataFrame:
        pd.read_json(self.path)

    @staticmethod
    def model_id_s3path(model_id: str) -> str:
        return model_id.replace('-', '/')

    @staticmethod
    def s3path_model_id(s3_path: str) -> str:
        return s3_path.replace('/', '-')

    @staticmethod
    def size(x: torch.Tensor, comment: str = ""):
        assert isinstance(x, torch.Tensor), f'Not a Tensor type'
        IOUtil.log_info(f'{list(x.size())} {comment} ')

    @staticmethod
    def log_info(msg: str):
        if constants.is_log_info:
            IOUtil.__log_info(f'INFO: {msg}')

    @staticmethod
    def log_error(msg: str):
        IOUtil.__log_info(f'ERROR: {msg}')

    @staticmethod
    def __log_info(msg: str):
        if constants.is_print_to_log:
            logging.info(msg)
        else:
            print(msg)


