from unittest import TestCase
from datasets.ngramdataset import NGramDataset
from embeddings.context import Context
from util.ioutil import IOUtil
import constants


class TestNGramDataset(TestCase):
    def test_from_s3(self):
        try:
            ngram_dataset = TestNGramDataset.__create_ngram_dataset(3)
            num_elements = len(ngram_dataset)
            ctx, tgt = ngram_dataset.target_tensors[0]
            IOUtil.log_info(f'Ctx {ctx} Tgt: {tgt}')
            assert num_elements > 0, 'Data set should not be empty'
            IOUtil.log_info(f'Number of elements {num_elements}')
        except Exception as e:
            self.fail(str(e))

    def test_getitem(self):
        try:
            ngram_dataset = TestNGramDataset.__create_ngram_dataset(4)
            ctx, tgt = ngram_dataset[0]
            IOUtil.log_info(f'Ctx {ctx} Tgt: {tgt}')

            assert ngram_dataset[0], "First element should be non empty"
            assert ngram_dataset[1], "Second element should be non empty"
            IOUtil.log_info(f'First element: {ngram_dataset[0]}\nSecond element {ngram_dataset[1]}')
        except Exception as e:
            self.fail(str(e))


    @staticmethod
    def __create_ngram_dataset(context_stride: int) -> NGramDataset:
        data_source = 'test/7/3'
        context_stride = 4
        context = Context(context_stride)
        s3_folder = f'{constants.embedding_layer_s3_folder}/{data_source}/terms'
        return NGramDataset.from_s3(constants.default_s3_bucket_name, s3_folder, context, ['terms'], '.json', 20)


# [tensor([[28240, 92815,  9120, 25769,  9803, 28240],
#         [19339, 76460, 94114, 94114, 85603, 67848],
#         [48408, 67569,  9165, 94114,  9165, 23257],
#         [84356, 16866, 44715, 97845, 19339, 78948]]), tensor([28240, 28240, 52464, 11992])]