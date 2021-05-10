__author__ = "Patrick Nicolas"

import asyncio
import concurrent.futures
import requests
from timeit import default_timer


class AsyncHttpPost(object):
    def __init__(self, url, headers):
        self.start_time = default_timer()
        self.url = url
        self.headers = headers


        # send request handle response
    def post_batch(self, input_file: str) -> list:
        with open(input_file) as input:
            all_requests = input.readlines()
            responses = []
            for request in all_requests:
                response = self.post(request)
                responses.append(response)
            return responses

    # send request handle response
    def post(self, request: str) -> str:
        with requests.post(self.url, json=request, headers=self.headers) as response:
            data = response.text
            if response.status_code != 200:
                print("Failed::{0}".format(self.url))

            elapsed = default_timer() - self.start_time
            time_completed_at = "{:5.2f}s".format(elapsed)
            print("{0:<30} {1:>20} {2:>40}".format('request', time_completed_at, data))
            return data

        # async and threadpooling happens here
    async def execute(self, request: str) -> str:
        loaded = request
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            loop = asyncio.get_event_loop()
            self.start_time = default_timer()
            futures = [
                loop.run_in_executor(executor, self.post, loaded)
                for i in range(20)
            ]
        for response in await asyncio.gather(*futures):
            pass
        return response