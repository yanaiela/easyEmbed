import unittest
from easyEmbed import embeddings as Embedding
from easyEmbed import easyEmbed as emb


class TestBaseEmbeddings(unittest.TestCase):

    def testNotImplemented(self):
        emb_ = Embedding.BaseEmbedding()
        self.assertRaises(NotImplementedError, emb_.download, None)
        self.assertRaises(NotImplementedError, emb_.decompress, None)
        self.assertRaises(NotImplementedError, emb_.load_binaries, None)

    def testLoadGloVe300(self):
        type = emb.Embeddings.glove
        voc_path = 'tests/glove_reduced_vocab.npy'
        emb_path = 'tests/glove_reduced_embeddings.npy'
        vocab, embeds = emb.read_vocab_subset(type, voc_path, emb_path)
        self.assertTrue(len(vocab), 2)
        self.assertTrue('apple' in vocab)
        self.assertTrue('horse' in vocab)
        self.assertTrue(all([len(x) == 300 for x in embeds]))
