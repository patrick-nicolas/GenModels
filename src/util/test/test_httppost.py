from unittest import TestCase

import urllib3
from util.httppost import HttpPost


class TestHttpPost(TestCase):
    def test_process(self):
        try:
            # in_file = "../../data/requests/test2.json"
            in_file = "../../data/requests/Mixed-modalities.json"
            # in_file = "../../data/requests/ct-issue-request.json"
            # in_file = "../../data/requests/76-encounters-requests.json"
            # in_file = "../../data/requests/40_7_utrad_mri_requests.json"
           # url = "http://ip-10-5-47-158.us-east-2.compute.internal:8087/geminiml/predict"
            # url = "http://ip-10-5-35-209.us-east-2.compute.internal:8087/geminiml/predict"
            url = "http://ip-10-5-55-122.us-east-2.compute.internal:8087/geminiml/predict"
            # url = "http://127.0.0.1:8080/geminiml/predict"
            new_headers = {'Content-type': 'application/json'}
            post = HttpPost(url, new_headers)
            post.post_batch(in_file, "")
        except urllib3.exceptions.ProtocolError as e:
            print(str(e))
            self.fail(str(e))
        except Exception as e:
            print(str(e))
            self.fail(str(e))
