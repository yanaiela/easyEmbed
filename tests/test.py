import unittest
from easyEmbed.embeddings import *
from easyEmbed import easyEmbed as emb
import numpy as np


class TestBaseEmbeddings(unittest.TestCase):
    def testNotImplemented(self):
        emb_ = BaseEmbedding()
        self.assertRaises(NotImplementedError, emb_.download, None)
        self.assertRaises(NotImplementedError, emb_.decompress, None)
        self.assertRaises(NotImplementedError, emb_.load_binaries, None)


class TestW2v(unittest.TestCase):
    emb_type = emb.Embeddings.w2v
    emb_file = 'data/w2v_reduced'
    word_list = ['for', 'said', 'in', 'that']

    def tearDown(self):
        import os
        os.remove('data/word2vec_reduced_embeddings.npy')
        os.remove('data/word2vec_reduced_vocab.npy')

    def testLoadW2V(self):
        vocab, embeds, voc_path, emb_path = emb.persist_vocab_subset(self.emb_type,
                                                                     self.emb_file, self.word_list)
        self.assertEqual(vocab[self.word_list[0]], 0)
        self.assertEqual(vocab[self.word_list[1]], 1)
        self.assertEqual(vocab[self.word_list[2]], 2)
        self.assertEqual(vocab[self.word_list[3]], 3)

        revocab, reembeds = emb.read_vocab_subset(self.emb_type, voc_path, emb_path)

        self.assertEqual(vocab, revocab)
        self.assertTrue(np.array_equal(embeds, reembeds))

    def testNormalizationW2V(self):
        _, embeds, _, _ = emb.persist_vocab_subset(self.emb_type, self.emb_file,
                                                   self.word_list, normalize=True)
        self.assertEqual(np.around(np.sum([x ** 2 for x in embeds[0]]), decimals=5), 1)


class TestGloVe(unittest.TestCase):
    def testLoadGloVe300(self):
        emb_type = emb.Embeddings.glove
        voc_path = 'tests/glove_reduced_vocab.npy'
        emb_path = 'tests/glove_reduced_embeddings.npy'
        vocab, embeds = emb.read_vocab_subset(emb_type, voc_path, emb_path)
        self.assertTrue(len(vocab), 2)
        self.assertTrue('apple' in vocab)
        self.assertTrue('horse' in vocab)
        self.assertTrue(all([len(x) == 300 for x in embeds]))


class TestIO(unittest.TestCase):

    def testLoad(self):
        emb_type = emb.Embeddings.w2v
        self.assertRaises(IOError, lambda: emb.read_vocab_subset(emb_type,
                                                                 'tests/glove_reduced_vocab.npy', 'non-existing'))
        self.assertRaises(IOError, lambda: emb.read_vocab_subset(emb_type,
                                                                 'non-existing', 'tests/glove_reduced_embeddings.npy'))
        self.assertRaises(IOError, lambda: emb.read_vocab_subset(emb_type,
                                                                 'non-existing', 'non-existing'))


