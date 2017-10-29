# EasyEmbed
[![Build Status](https://travis-ci.org/yanaiela/easyEmbed.svg?branch=master)](https://travis-ci.org/yanaiela/easyEmbed)
[![PyPI version](https://badge.fury.io/py/easyEmbed.svg)](https://badge.fury.io/py/easyEmbed)


EasyEmbed is a tool for people who use pre-trained word embeddings which 
will save you time while developing and playing around with embeddings


## Necessity
1. No need to handle the downloading process of well known pre-trained embeddings
2. A lot of applications doesn't need the million/billions word vectors which
the common pre-trained models provide. Something which, while development 
make you lose much time of waiting for the matrices to load.
    
    With this tool one can load the matrix once, supply it the models' relevant
     words (for the development phase it's ok to supply also words which doesn't
     appear in train), and save them apart of the original file. Just dont forget
     when the model is ready for deployment to use the whole matrix embeddings. 


## Features
* Let you forget about the location of the pretrained matrices. Currently supporing the following:
    * [Google Word2Vec](https://code.google.com/archive/p/word2vec/)
        * Google News dataset (~100b words, 3 million words and phrases, 300-dim)
    * [Stanford GloVe](https://github.com/stanfordnlp/GloVe)
        * Common Crawl (840B tokens, 2.2M vocab, cased, 300-dim)
    * [Word2VecF](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/)
        * Wikipedia dataset - dependency-based embeddings (175K vocab, 300-dim)
    * Custom pre-trained!
        * Trained your own model? no problem, just implement a few
         methods for telling the tool how to load and use them,
         and you're good to go.
* Keep in separate files only the words you care about, for avoiding
 large amount of time for the full matrix to load.
* Keeping a unify format for all embeddings. You don't have to deal with
how the pre-trained are saved or their format
 
 
 ## Requirements
 * numpy
 * if using word2vec: gensim
 * if using GloVe: pandas, csv
 
 
 ## Instalation
 ```bash
pip install easyEmbed
```
 
 ## Usage
 #### Init:
 ```python
from easyEmbed import easyEmbed as emb
emb_type = emb.Embeddings.w2v
# emb_type = emb.Embeddings.glove # the alternative
```
 
 #### For dowloading the pre-trained embeddings:
 ```python
download_dir = 'path-to-dir'
emb_file = emb.download(emb_type, download_dir) 
 ```
 This will download the relevant file and decompress it.
 
 #### Creating new reduced files
 Once you have downloaded (or already have) the data
```python
emb_file = 'path-to/word2vec_data.bin' # or the emb_file variable from previous section
vocab, embeds, voc_path, emb_path = emb.persist_vocab_subset(emb_type, emb_file, word_set)
```
Notice: 
* Big vocabulary pre-trained (lile the GloVe) require a lot of memory
* The `persist_vocab_subset` function has another parameter: `missing_embed`.
This parameter is a function that returns an embedding vector when the required
word doesn't exist in the pre-trained embeddings.

#### Retrieving the dictionary and embeddings
```python
# or keep them from last section
voc_path = 'path-to-vocab-dict-file'  
emb_path = 'path-to-embedding-file'
vocab, embeds = emb.read_vocab_subset(emb_type, voc_path, emb_path)
```

#### Full Usage Example
```python
# First run
from easyEmbed import easyEmbed as emb
emb_type = emb.Embeddings.w2v
download_dir = 'path-to-dir'
emb_file = emb.download(emb_type, download_dir) 
vocab, embeds, voc_path, emb_path = emb.persist_vocab_subset(emb_type, emb_file, word_set)

# for the rest of your experiments
vocab, embeds = emb.read_vocab_subset(emb_type, voc_path, emb_path)
```


#### Custom model usage
* This imaginary pre-trained is similar to the GloVe embeddings,
where each row consists of the word itself, space and the vecor
```python
from easyEmbed import easyEmbed as emb
import pandas as pd
import csv
load_emb_f = lambda f: pd.read_table(f, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
word_exists_f = lambda w, w2v: w in w2v.index.values
get_vector_f = lambda w, w2v: w2v.loc[w].as_matrix()

emb_f = 'path-to/vectors'
type = emb.CustomEmb('foo-vectors', emb_f,
                          load_emb_f, word_exists_f, get_vector_f)
emb_file = emb_f
vocab, embeds, voc_path, emb_path = emb.persist_vocab_subset(type, emb_file, words_set)
```

#### MISC
* Tested on python 2.7

## TODO
- [ ] More vectors size for GloVe
- [x] Add Word2vecf 
- [x] Custom embeddings
- [ ] Support for windows
- [ ] Test for other python version
- [ ] Add more tests


* Any suggestions, feedback and PR are welcome