from unittest import TestCase

from util.asynchttppost import AsyncHttpPost


class TestAsyncHttpPost(TestCase):
    def test_post(self):
        try:
            in_file = "../../data/requests/Mixed-modalities.json"

            url = "http://127.0.0.1:8080/geminiml/predict"
            headers = {'Content-type': 'application/json'}
            async_http_post = AsyncHttpPost(url, headers)
            responses = async_http_post.post_batch(in_file)
            print(str(responses))
        except Exception as e:
            print(str(e))
            self.fail()
