from unittest import TestCase

from embeddings.vocabulary import Vocabulary

class TestVocabulary(TestCase):
    def test_vocabulary_fail(self):
        try:
            vocab = Vocabulary()
            print(repr(vocab))
        except Exception as e:
            self.fail(f'Failed vocabulary test with {str(e)}')


