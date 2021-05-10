__author__ = "Patrick Nicolas"

from util.ioutil import IOUtil
from util.s3util import S3Util


class Loader(object):
    def __init__(self, folder, s3_bucket_name = ""):
        if s3_bucket_name != "":
            s3_util = S3Util(s3_bucket_name, folder, True)
            self.df = s3_util.to_dataframe()
        else:
            ioutil = IOUtil(folder)
            self.df = ioutil.to_dataframe()
