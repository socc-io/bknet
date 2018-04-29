# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import numpy as np
from kor_parser import decompose_str_as_one_hot, syllable_one_hot
from tqdm import tqdm
from collections import Counter

from konlpy.tag import Mecab
from word2vec import get_wordvec, get_word, get_wordidx, wordvec_lookup, unknown_wordvec

class DevDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.offset = dataset.train_len
    def __len__(self):
        return self.dataset.dev_len
    def __getitem__(self, idx):
        return self.dataset[idx.start + self.offset:idx.stop + self.offset]

class KinQueryDataset:
    """
        지식인 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다.
    """
    def __init__(self, dataset_path,  max_sentence_length, max_word_length, max_syll_num,
        is_partial_dev=False, train_ratio=0.9, word_dim=200):
        queries_path = os.path.join(dataset_path, 'train', 'train_data')
        labels_path = os.path.join(dataset_path, 'train', 'train_label')
        with open(queries_path, 'rt', encoding='utf8') as f:
            (self.l_x, self.l_x_len, self.l_cx, self.l_c_len, self.l_sx, self.l_s_len), \
            (self.r_x, self.r_x_len, self.r_cx, self.r_c_len, self.r_sx, self.r_s_len) = \
            preprocess_kin(f.readlines(), max_sentence_length, max_word_length, max_syll_num, word_dim=word_dim)
        with open(labels_path) as f:
            self.labels = np.array([[np.float32(x)] for x in f.readlines()])
        self._len = len(self.l_x)
        self.train_len = int(self._len * train_ratio)
        self.dev_len = self._len - self.train_len
        self.word_dim = word_dim
        self.ret_len = self._len
        self.shuffle_dataset()
        if is_partial_dev:
            self.ret_len = self.train_len
        else:
            self.ret_len = self._len

    def __len__(self):
        return self.ret_len

    def __getitem__(self, idx):
        l_x = wordvec_lookup(self.l_x[idx], self.word_dim)
        r_x = wordvec_lookup(self.r_x[idx], self.word_dim)
        return (l_x, self.l_x_len[idx], self.l_cx[idx], self.l_c_len[idx], self.l_sx[idx], self.l_s_len[idx]), \
               (r_x, self.r_x_len[idx], self.r_cx[idx], self.r_c_len[idx], self.r_sx[idx], self.r_s_len[idx]), \
               self.labels[idx]

    def shuffle_dataset(self):
        s = np.arange(self.ret_len)
        np.random.shuffle(s)
        self.l_x = self.l_x[s]
        self.l_x_len = self.l_x_len[s]
        self.l_cx = self.l_cx[s]
        self.l_c_len = self.l_c_len[s]
        self.l_sx = self.l_sx[s]
        self.l_s_len = self.l_s_len[s]
        self.r_x = self.r_x[s]
        self.r_x_len = self.r_x_len[s]
        self.r_cx = self.r_cx[s]
        self.r_c_len = self.r_c_len[s]
        self.r_sx = self.r_sx[s]
        self.r_s_len = self.r_s_len[s]
        self.labels = self.labels[s]


class MovieReviewDataset:
    """
    영화리뷰 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다.
    """
    def __init__(
        self,
        dataset_path,
        max_sentence_length=0,
        max_word_length=0,
        max_syll_num=0,
        is_partial_dev=False,
        train_ratio=0.9,
        word_dim=200,
        use_word_embedding=False,
        max_embedding_num=10000):

        data_review = os.path.join(dataset_path, 'train', 'train_data')
        data_label = os.path.join(dataset_path, 'train', 'train_label')
        with open(data_review, 'rt', encoding='utf-8') as f:
            self.sents, self.sent_lengths, self.words, self.word_lengths, self.sylls, self.syll_num = \
                preprocess(f.readlines(), max_sentence_length, max_word_length, max_syll_num, word_dim=word_dim)
            if not use_word_embedding:
                self.sents = word2idx(self.sents, max_sentence_length, max_embedding_num)
        with open(data_label) as f:
            self.labels = np.array([[np.float32(x)] for x in f.readlines()], dtype=np.float32)
        self._len = len(self.sents)
        self.train_len = int(self._len * train_ratio)
        self.dev_len = self._len - self.train_len
        self.word_dim = word_dim
        self.max_sentence_length = max_sentence_length
        self.ret_len = self._len
        self.use_word_embedding = use_word_embedding
        self.shuffle_dataset()
        if is_partial_dev:
            self.ret_len = self.train_len
        else:
            self.ret_len = self._len

    def __len__(self):
        return self.ret_len

    def __getitem__(self, idx):
        if self.use_word_embedding:
            sents = wordvec_lookup(self.sents[idx], self.word_dim)
        else:
            sents = self.sents[idx]
        return sents, \
               self.sent_lengths[idx], \
               self.words[idx], \
               self.word_lengths[idx], \
               self.sylls[idx], \
               self.syll_num[idx], \
               self.labels[idx], \

    def shuffle_dataset(self):
        real_len = len(self.sents)
        s = np.arange(self.ret_len)
        np.random.shuffle(s)
        s = np.concatenate([s, np.arange(self.ret_len,real_len)])
        self.sents = self.sents[s]
        self.sent_lengths = self.sent_lengths[s]
        self.words = self.words[s]
        self.word_lengths = self.word_lengths[s]
        self.labels = self.labels[s]
        self.sylls = self.sylls[s]
        self.syll_num = self.syll_num[s]

def word2idx(x_str, max_word_num, max_embedding_num):
    global word_cnt

    sent_num = len(x_str)

    x = np.zeros([sent_num, max_word_num], dtype=np.int32)

    word_idx = {}
    for idx, (word, cnt) in enumerate(word_cnt.most_common(max_embedding_num)):
        word_idx[word] = idx + 1
    unknown_word_idx = max_embedding_num + 1
    for idx, words in enumerate(x_str):
        i_words = [word_idx.get(word, unknown_word_idx) for word in words]
        i_word_len = len(i_words)
        x[idx][:len(i_words)] = np.array(i_words)
    
    return x

def preprocess(sentences, max_word_num, max_char_num, max_syll_num, word_dim, use_word_embedding=True):
    '''
    @param sentences: string[]
    @param max_word_num -> int
    @param max_char_num -> int
    @param max_syll_num -> int
    @param use_word_embedding -> bool:
        If this variable affects to the shape of x, returned variable
    @return x: Array<float32>[len(sentences), max_word_num]
        or     Array<string>[len(sentences), max_word_num] (if not use_word_embedding)
    @return x_len: Array<int32>[len(sentences)]
    @return cx: Array<int32>[len(sentences), max_word_num, max_char_num]
    @return c_len: Array<int32>[len(sentences), max_word_num]
    @return sx: Array<int32>[len(sentences), max_word_num, max_syll_num]
    @return s_len: Array<int32>[len(sentences), max_word_num]
    '''
    global word_cnt

    sent_num = len(sentences)

    if use_word_embedding:
        x = np.zeros([sent_num, max_word_num], dtype=np.int32)
    else:
        x_str = []
    x_len = np.zeros([sent_num], dtype=np.int32)
    cx    = np.zeros([sent_num, max_word_num, max_char_num], dtype=np.int32)
    c_len = np.zeros([sent_num, max_word_num], dtype=np.int32)
    sx    = np.zeros([sent_num, max_word_num, max_syll_num], dtype=np.int32)
    s_len = np.zeros([sent_num, max_word_num], dtype=np.int32)

    tagger = Mecab()

    word_cnt = Counter()

    word_num_cnt = {}
    char_num_cnt = {}
    syll_num_cnt = {}

    for s_idx, sent in enumerate(tqdm(sentences)):
        words = ['{}/{}'.format(obj[0], obj[1]) for obj in tagger.pos(sent)]

        # Replace ac### -> __ACTOR__
        # Replace mv### -> __MOVIE__
        for idx, word in enumerate(words):
            if   word == 'ac' and idx != len(words)-1:
                words[idx] = '__ACTOR__'
                words[idx+1] = '__REMOVE__'
            elif word == 'mv' and idx != len(words)-1:
                words[idx] = '__MOVIE__'
                words[idx+1] = '__REMOVE__'

        words = [w for w in words if w != '__REMOVE__']  # Remove '__REMOVE__'
        word_num_cnt[len(words)] = word_num_cnt.get(len(words),0) + 1
        words = words[:max_word_num]
        word_num = len(words)
        if word_num == 0:
            continue
        if use_word_embedding:
            v_words = [get_wordidx(w, dim=word_dim) for w in words]
            x[s_idx][:word_num] = np.array(v_words)
            x_len[s_idx] = word_num
        else:
            x_str.append(words)
            for word in words:
                word_cnt[word] += 1

        for w_idx, word in enumerate(words):
            decomposed = list(decompose_str_as_one_hot(word, warning=False))
            char_num_cnt[len(decomposed)] = char_num_cnt.get(len(decomposed),0) + 1
            decomposed = decomposed[:max_char_num]
            char_num = len(decomposed)
            cx[s_idx][w_idx][:char_num] = np.array(decomposed)
            c_len[s_idx][w_idx] = len(decomposed)

        for w_idx, word in enumerate(words):
            sylls = [syllable_one_hot(w) for w in list(word)]
            syll_num_cnt[len(sylls)] = syll_num_cnt.get(len(sylls),0) + 1
            sylls = sylls[:max_syll_num]
            syll_num = len(sylls)
            sx[s_idx][w_idx][:syll_num] = np.array(sylls)
            s_len[s_idx][w_idx] = syll_num

    print("Input shapes ==============================================================")
    print("x      : ", x.shape)
    print("x_len  : ", x_len.shape)
    print("cx     : ", cx.shape)
    print("c_len : ", c_len.shape)
    print("sx     : ", sx.shape)
    print("s_len  : ", s_len.shape)
    print("===========================================================================")

    import operator
    print('word_num_cnt : ', sorted(word_num_cnt.items(), key=operator.itemgetter(0)))
    print('char_num_cnt : ', sorted(char_num_cnt.items(), key=operator.itemgetter(0)))
    print('syll_num_cnt : ', sorted(syll_num_cnt.items(), key=operator.itemgetter(0)))
    print("===========================================================================")

    if use_word_embedding:
        return (x,     x_len, cx, c_len, sx, s_len)
    else:
        return (x_str, x_len, cx, c_len, sx, s_len)

def preprocess_kin(sentences, *args, **kwargs):
    l_sentences = []
    r_sentences = []
    for sent in sentences:
        l_sent, r_sent = sent.split('\t')
        l_sentences.append(l_sent)
        r_sentences.append(r_sent)

    l_res = preprocess(l_sentences, *args, **kwargs)
    r_res = preprocess(r_sentences, *args, **kwargs)

    return l_res, r_res
