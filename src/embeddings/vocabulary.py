__author__ = "Patrick Nicolas"

from util.ioutil import  IOUtil
import constants


class Vocabulary(object):
    def __init__(self, terms_path: str = constants.vocab_path):
        io_util = IOUtil(terms_path)
        vocab = set([w.rstrip().lower() for w in io_util.to_lines()])
        self.word_index_dict = {word: i for i, word in enumerate(vocab)}
        self.index_word_dict = {i: word for i, word in enumerate(vocab)}

    def __len__(self):
        return len(self.word_index_dict)

    def __repr__(self) -> str:
        return str(self.word_index_dict) + '\n' + str(self.index_word_dict)