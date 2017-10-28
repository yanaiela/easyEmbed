# The interface file
from embeddings import *


class Embeddings(object):
    w2v = Word2Vec()
    glove = GloVe()
    # add more here...


def download(emb_type, directory='~/.embeddings'):
    """
    Downloads the required pre-trained Embedding
    :param emb_type: One of the supported Embedding classes (Word2Vec, GloVe etc..)
    :param directory: The directory where the file will be downloaded
            default dir: ~/.embeddings
    :return: the full path of the downloaded file
    """
    if directory[-1] != SEP:
        directory += SEP
    if not (os.path.exists(directory) and os.path.isdir(directory)):
        print 'directory does not exist.. create one'
        os.makedirs(directory)

    f_emb = directory + emb_type.file
    if not os.path.exists(f_emb):
        print 'downloading the required file: ' + emb_type.name
        emb_type.download(directory)
    else:
        print 'file already exist..'
    return f_emb


def persist_vocab_subset(emb_type, emb_file, word_set, missing_embed=lambda: np.random.rand(1, 300),
                         normalize=False):
    """
    Extracting the required vocabulary from the word embeddings.
    :param emb_type: One of the supported Embedding classes (Word2Vec, GloVe etc..)
    :param emb_file: The embedding file which was downloaded (should be the decompressed file)
    :param word_set: a list of string - all relevant words for the development stage
    :param missing_embed: a function which fills up an embedding vector when such is missing.
            default one is a random vector (1,300) taken from a uniform distribution over [0,1).
            It should return a numpy 1 dimension array with same dimension as the rest embeddings
    :return: vocab - dictionary mapping between word to the embeds index
            embeds - the new (diminished) word embeddings
            voc_path - the file path where the vocabulary dictionary was saved
            emb_path - the file path where the embedding matrix was saved
    """
    emb_dir = SEP.join(emb_file.split(SEP)[:-1])
    if (os.path.exists(emb_dir + '/' + emb_type.name + '_' + emb_type.REDUCED_VOC) and
                os.path.exists(emb_dir + '/' + emb_type.name + '_' + emb_type.REDUCED_EMB)):
        raise IOError('reduced files already exist, please delete them first')
    if not os.path.exists(emb_file):
        print emb_file
        raise ValueError('embedding file does not exist, please download the file first')
    vocab, embeds = emb_type.get_vectors(emb_file, word_set, missing_embed, normalize)
    voc_path, emb_path = emb_type.persist_reduced(vocab, embeds, emb_dir)
    return vocab, embeds, voc_path, emb_path


def read_vocab_subset(emb_type, voc_path, emb_path):
    """
    Reading and retrieving the reduced embeddings from persisted files
    :param emb_type:
    :param voc_path: path to the vocabulary dictionary
    :param emb_path: path to the embedding file
    :return: the vocabulary dictionary (maps between word and id) and the embedding mat
    """
    if not (os.path.exists(voc_path) and os.path.exists(emb_path)):
        raise IOError('vocabulary / embedding binaries files doesn\'t exists')
    vocab, embeds = emb_type.load_reduced(voc_path, emb_path)
    return vocab, embeds
