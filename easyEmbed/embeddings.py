# Inner file.
import subprocess
import gzip
import zipfile
import bz2
from collections import defaultdict
import itertools
import numpy as np
import os

W2V_DOWNLOAD_SCRIPT = './word2vec-download300model.sh'
GLOVE_DOWNLOAD_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
W2VF_DOWNLOAD_URL = 'http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2'

SEP = os.sep


class BaseEmbedding(object):

    REDUCED_VOC = 'reduced_vocab.npy'
    REDUCED_EMB = 'reduced_embeddings.npy'

    def download(self, directory):
        """
        Downloading the pre-trained word embeddings
        :param directory: where to download the file(s) to
        :return: nothing
        """
        raise NotImplementedError

    def decompress(self, emb_file):
        """
        Decompressing (if needed) the downloaded file
        :param emb_file: downloaded file
        :return: nothing
        """
        raise NotImplementedError

    def get_vectors(self, emb_file, vocab, missing_embed, normalize=False):
        """
        Building the internal (needed) representation for the development process
        :param emb_file: the pre-trained embedding file
        :param vocab: the relevant vocabulary of the train, dev and test
        :param missing_embed: a function which is called in case of a missing word.
        Expecting a numpy vector
        :return: dictionary of the vocabulary to their new indices and a numpy matrix of the relevant vocabulary
        """
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
        embeds = np.array(embeds)
        if normalize:
            row_norm = np.sum(np.abs(embeds) ** 2, axis=-1) ** (1. / 2)
            embeds /= row_norm[:, np.newaxis]
        return dict(words), embeds

    def load_binaries(self, emb_file):
        raise NotImplementedError

    def persist_reduced(self, vocab, embeds, directory):
        if directory[-1] != SEP:
            directory += SEP
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


class Word2VecF(BaseEmbedding):
    def __init__(self):
        self.name = 'word2vecf'
        self.file = 'deps.words'
        self._compress = '.bz2'

    def download(self, directory):
        subprocess.call(['wget', W2VF_DOWNLOAD_URL, '-P', directory])
        self.decompress(directory)
        return self.file

    def decompress(self, directory):
        if directory[-1] != SEP:
            directory += SEP
        inF = bz2.BZ2File(directory + self.file + self._compress)
        outF = open(directory + self.file, 'wb')
        outF.write(inF.read())
        inF.close()
        outF.close()

    def load_binaries(self, emb_file):
        import pandas as pd
        import csv
        w2v = pd.read_table(emb_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        return w2v

    def word_exists(self, w, w2v):
        return w in w2v.index.values

    def get_vector(self, w, w2v):
        return w2v.loc[w].as_matrix()


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


class CustomEmb(BaseEmbedding):
    """
    This is a custom embedding class. Its' relevant functions, name and file name
    should be passed through the constructor.
    """
    def __init__(self, name, custom_emb_file, load_bin_f, word_exists_f, get_vec_f):
        self.name = name
        self.file = custom_emb_file

        self.load_binaries = load_bin_f
        self.word_exists = word_exists_f
        self.get_vector = get_vec_f

    def __init__(self):
        pass

    def download(self, directory):
        raise NotImplementedError('This method is not implemented. you should own your own embedding file')

    def decompress(self, directory):
        raise NotImplementedError('This method is not implemented. your file should already be decompressed')

    def load_binaries(self, emb_file):
        pass

    def word_exists(self, w, w2v):
        pass

    def get_vector(self, w, w2v):
        pass
