# Inner file.
import subprocess
import gzip
import zipfile
from collections import defaultdict
import itertools
import numpy as np
import os

W2V_DOWNLOAD_SCRIPT = './word2vec-download300model.sh'
GLOVE_DOWNLOAD_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'

SEP = os.sep


class BaseEmbedding(object):

    REDUCED_VOC = 'reduced_vocab.npy'
    REDUCED_EMB = 'reduced_embeddings.npy'

    def download(self, directory):
        raise NotImplementedError

    def decompress(self, emb_file):
        raise NotImplementedError

    def get_vectors(self, emb_file, vocab, missing_embed):
        w2v = self.load_binaries(emb_file)

        words = defaultdict(itertools.count(0).next)
        embeds = []
        for w in vocab:
            # if w in w2v:
            words[w]
            if self.word_exists(w, w2v):
                # embeds.append(w2v[w])
                embeds.append(self.get_vector(w, w2v))
            else:
                embeds.append(missing_embed().flatten())
        return dict(words), np.array(embeds)

    def load_binaries(self, emb_file):
        raise NotImplementedError

    def persist_reduced(self, vocab, embeds, directory):
        if directory[-1] != '/':
            directory += '/'
        vocab_n = directory + self.name + '_' + self.REDUCED_VOC
        embed_n = directory + self.name + '_' + self.REDUCED_EMB
        np.save(vocab_n, dict(vocab))
        np.save(embed_n, embeds)
        print 'saved reduced files succesfully'
        return vocab_n, embed_n

    def load_reduced(self, vocab_n, embed_n):
        vocab = np.load(vocab_n).item(0)  # as np saves everything as ndarray
        embed = np.load(embed_n)
        return vocab, embed

    def word_exists(self, w, w2v):
        raise NotImplementedError

    def get_vector(self, w, w2v):
        raise NotImplementedError


class Word2Vec(BaseEmbedding):
    def __init__(self):
        self.name = 'word2vec'
        self.file = 'GoogleNews-vectors-negative300.bin'
        self._compress = '.gz'

    def download(self, directory):
        subprocess.call([W2V_DOWNLOAD_SCRIPT, directory])
        self.decompress(directory)
        return self.file

    def decompress(self, directory):
        if directory[-1] != SEP:
            directory += SEP
        inF = gzip.open(directory + self.file + self._compress, 'rb')
        outF = open(directory + self.file, 'wb')
        outF.write(inF.read())
        inF.close()
        outF.close()

    def load_binaries(self, emb_file):
        import gensim
        w2v = gensim.models.KeyedVectors.load_word2vec_format(emb_file, binary=True)
        return w2v

    def word_exists(self, w, w2v):
        return w in w2v

    def get_vector(self, w, w2v):
        return w2v[w]


class GloVe(BaseEmbedding):
    def __init__(self):
        self.name = 'glove'
        self.file = 'glove.840B.300d.txt'
        self._compress = '.zip'

    def download(self, directory):
        subprocess.call(['wget', GLOVE_DOWNLOAD_URL, '-P', directory])
        self.decompress(directory)
        return self.file

    def decompress(self, directory):
        if directory[-1] != SEP:
            directory += SEP
        zip_ref = zipfile.ZipFile(directory + '.'.join(self.file.split('.')[:-1]) + self._compress, 'r')
        zip_ref.extractall(directory)
        zip_ref.close()

    def load_binaries(self, emb_file):
        import pandas as pd
        import csv
        w2v = pd.read_table(emb_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        return w2v

    def word_exists(self, w, w2v):
        return w in w2v.index.values

    def get_vector(self, w, w2v):
        return w2v.loc[w].as_matrix()

