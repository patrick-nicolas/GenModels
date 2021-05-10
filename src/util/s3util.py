__author__ = "Patrick Nicolas"

import io
import json
from typing import Optional
import boto3
import pandas as pd
from pandas import json_normalize
from util.ioutil import IOUtil
import pickle
from collections import OrderedDict


AWS_ACCESS_KEY_ID='AKIAQVWRM24MJK2VXHKY'
AWS_SECRET_ACCESS_KEY='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'
AWS_SHARED_CREDENTIALS_FILE="~/.aws/credentials2"

session = boto3.Session(profile_name='default', region_name='us-east-2', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
s3 = session.resource('s3')


"""
    Utility class for manipulating S3 data
    :param s3_bucket: name of bucket
    :param s3_folder_path: path to the S3 folder containing the file to load
    :param is_nested: Boolean flag to specify whether the json structure is nested
"""


class S3Util(object):
    def __init__(self, s3_bucket: str, s3_folder_name: str, is_nested: bool):
        self.s3_bucket_name = s3_bucket
        self.s3_folder_name = s3_folder_name
        self.is_nested = is_nested

    '''
        Load the content of S3 folder into a data frame. Optional feature_names can be used to extract
        only the relevant fields.
        :param file_extension: File extention to match if defined.
        :param feature_names: Optional names of column_names for the features (and label) to extract
        :returns: Pandas dataframe
    '''
    def read_dataframe(self, file_extension: str = None, col_names: list = None) -> pd.DataFrame:
        my_bucket = s3.Bucket(self.s3_bucket_name)

        accu = []
        for object_summary in my_bucket.objects.filter(Prefix = self.s3_folder_name):
            # We extract a subset of file in this folder if either an extension is undefined
            # or if the file name ends with the file extension
            if file_extension is not None and object_summary.key.endswith(file_extension):
                data_in_bytes = object_summary.get()['Body'].read()  # data in the form of bytes array.
                decoded_data = data_in_bytes.decode('utf-8')  # Decode it in 'utf-8' format
                stringio_data = io.StringIO(decoded_data)  # IO module for creating a StringIO object.
                data_list = stringio_data.readlines()
                # Extract the data for each column
                for data in data_list:
                    dict = json.loads(data)
                    accu.append([dict[col_name] for col_name in col_names])
                del object_summary, data_in_bytes, decoded_data, stringio_data
        # Finally generate the data frame
        df = pd.DataFrame(accu)
        del accu
        return df

    def write_ordered_dict(self, ordered_dict: OrderedDict, ext: str = ''):
        obj = pickle.dumps(ordered_dict)
        s3.Object(self.s3_bucket_name, self.__s3_folder(ext)).put(Body=obj)

    def read_ordered_dict(self, ext: str = '') -> OrderedDict:
        obj = s3.Object(self.s3_bucket_name, self.__s3_folder(ext)).get()['Body'].read()
        return pickle.loads(obj)

    def write_value(self, value: str, ext: str = ''):
        s3.Object(self.s3_bucket_name, self.__s3_folder(ext)).put(Body=value)

    def read_value(self, ext: str = ''):
        return s3.Object(self.s3_bucket_name, self.__s3_folder(ext)).get()['Body'].read()


    def read_json(self, file_extension: str) -> list:
        my_bucket = s3.Bucket(self.s3_bucket_name)

        # edit counter to limit record count
        counter = 200000
        prediction_data = []
        for object_summary in my_bucket.objects.filter(Prefix=self.s3_folder_name):
            # We extract a subset of file in this folder if either an extension is undefined
            # or if the file name ends with the file extension
            if file_extension is not None and object_summary.key.endswith(file_extension):
                data_in_bytes = object_summary.get()['Body'].read()  # data in the form of bytes array.
                decoded_data = data_in_bytes.decode('utf-8')  # Decode it in 'utf-8' format
                stringio_data = io.StringIO(decoded_data)  # IO module for creating a StringIO object.
                data_list = stringio_data.readlines()
                json_data = list(map(json.loads, data_list))
                IOUtil.log_info(object_summary.key, "  count:", len(json_data))

                counter = counter - len(json_data)
                if counter <= 0:
                    json_data = json_data[0: counter]
                    prediction_data += json_data
                    break
                else:
                    prediction_data += json_data
        del object_summary, data_in_bytes, decoded_data, stringio_data, json_data
        return prediction_data

    def to_dataframe(self, file_extension: str ='') -> pd.DataFrame:
        prediction_data = self.read_json(file_extension)
        if self.is_nested:
            return json_normalize(prediction_data)
        else:
            return pd.DataFrame.from_records(prediction_data)

    @staticmethod
    def to_csv(output_csv: str, data_frame: pd.DataFrame) -> Optional[str]:
        data_frame.to_csv(output_csv)

    def __s3_folder(self, ext: str):
        return f'{self.s3_folder_name}.{ext}'