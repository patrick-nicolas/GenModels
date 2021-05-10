__author__ = "Patrick Nicolas"

import json
import requests
import time
from util.ioutil import IOUtil


class HttpPost(object):
	def __init__(self, url: str, headers: list):
		self.headers = headers
		self.url = url

	def post_batch(self, input_file: str, output_file: str):
		if output_file != "":
			response_file = open(output_file, "w")
		else:
			response_file = ""

		with open(input_file) as input:
			all_requests = input.readlines()
			for line in all_requests:
				response = requests.post(self.url, data=line, headers=self.headers)
				IOUtil.log_info("Received ML Response: {}".format(response.status_code))
				if response.status_code == 200:
					json_response = response.json()
					IOUtil.log_info(json_response)
					if response_file != "":
						json.dump(json_response , response_file)
						response_file.write('\n')
				else:
					IOUtil.log_info(f'Error {line}')
				time.sleep(1)

