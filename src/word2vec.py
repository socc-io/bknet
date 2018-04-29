
import numpy as np
import pickle

_word2vec = None
_word2idx = {}
_word_arr = []
_dim = None

unknown_wordvec = None
unknown_wordidx = None

UNKNOWN_WORD = 'UNK'
CONSTANT_PATH = '/tmp/nsml/word2vec/leeck/'

def _init(dim):
    global _dim
    global _word2vec
    global _word2idx
    global _word_arr
    global unknown_wordvec
    global unknown_wordidx

    if _word2vec == None:
        if CONSTANT_PATH is None:
            BASE = '/tmp/nsml/word2vec/d{}/'.format(dim)
        else:
            BASE = '/tmp/nsml/word2vec/leeck/'
        _word2vec = pickle.load(open(BASE + 'vectors.pk', 'rb'))
        _dim = dim
        _word_arr.append('')
        for idx, key in enumerate(_word2vec):
            _word2idx[key] = idx + 1
            _word_arr.append(key)
        unknown_wordvec = _word2vec[UNKNOWN_WORD]
        unknown_wordidx = _word2idx[UNKNOWN_WORD]

def get_wordvec(word, dim):
    global _word2vec
    global unknown_wordvec

    _init(dim)

    return _word2vec.get(word, unknown_wordvec)

def get_wordidx(word, dim):
    global _word2idx
    global unknown_wordidx

    _init(dim)
    return _word2idx.get(word, unknown_wordidx)

def get_word(idx, dim):
    global _word_arr

    _init(dim)
    return _word_arr[idx]

def wordvec_lookup(arr, dim):
    _init(dim)
    o_shape = arr.shape
    arr = arr.reshape([-1])
    res = np.zeros([len(arr), dim])
    for idx in range(len(arr)):
        if arr[idx] == 0: continue
        res[idx] = get_wordvec(get_word(arr[idx],dim),dim)
    return res.reshape(o_shape + (dim,))

if __name__ == '__main__':
    pass

