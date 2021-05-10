__author__ = "Patrick Nicolas"

import time


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        print(args[0])
        func(*args, **kwargs)
        duration = time.time() - start
        print(f'Duration: {duration}')
        return 0
    return wrapper
