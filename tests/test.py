import unittest
from easyEmbed import embeddings as Embedding
from easyEmbed import easyEmbed as emb
import numpy as np

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

    def testLoadW2V(self):
        emb_type = emb.Embeddings.w2v
        emb_file = 'data/w2v_reduced'
        word_list = ['for', 'said', 'in', 'that']
        vocab, embeds, voc_path, emb_path = emb.persist_vocab_subset(emb_type, emb_file, word_list)
        self.assertEqual(vocab[word_list[0]], 0)
        self.assertEqual(vocab[word_list[1]], 1)
        self.assertEqual(vocab[word_list[2]], 2)
        self.assertEqual(vocab[word_list[3]], 3)

        revocab, reembeds = emb.read_vocab_subset(emb_type, voc_path, emb_path)

        self.assertEqual(vocab, revocab)
        self.assertTrue(np.array_equal(embeds, reembeds))
