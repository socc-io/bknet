
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


import argparse
import os

import numpy as np
import tensorflow as tf
from random import shuffle
import time

import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from dataset import MovieReviewDataset, preprocess

from core_layer import han1_char_rnn

DEBUG1_PATH = '../sample_data/movie/debug_1'
DEBUG2_PATH = '../sample_data/movie/debug_2'
DEBUG3_PATH = '../sample_data/movie/debug_3'

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        we, sl, cs, wl, _, _ = preprocess(raw_data, config.max_sentence_length, config.max_word_length, 0)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(y_logits, feed_dict={word_embed: we, sent_len: sl, chars: cs, word_len: wl, is_training: False})
        return list(zip(np.zeros(len(pred)), pred))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)

def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]

def _batch_debug_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='train | test_local')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--epochs', type=int, default=40)
    args.add_argument('--batch', type=int, default=800)
    args.add_argument('--strmaxlen', type=int, default=400)

    args.add_argument('--embedding', type=int, default=100)
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--keep_prob', type=float, default=0.8)

    args.add_argument('--log_freq', type=int, default=30)
    args.add_argument('--debug', action="store_true")
    args.add_argument('--debug_freq', type=int, default=100)
    args.add_argument('--test', action="store_true")

    args.add_argument('--n_units', type=int, default=10)
    args.add_argument('--max_sentence_length', type=int, default=20)
    args.add_argument('--max_word_length', type=int, default=20)
    args.add_argument('--word_dim', type=int, default=50)
    args.add_argument('--char_dim', type=int, default=50)
    args.add_argument('--rnn_dim', type=int, default=100)

    config = args.parse_args()

    if config.debug :
        DATASET_PATH = DEBUG1_PATH
    if config.test:
        DATASET_PATH = DEBUG2_PATH

    ##############################################################################################################
    # 모델의 specification
    character_size = 252
    n_units = config.n_units
    max_sent_len = config.max_sentence_length
    max_word_len = config.max_word_length
    word_dim = config.word_dim
    char_dim = config.char_dim
    learning_rate = config.lr

    is_training=tf.placeholder(dtype=tf.bool, name='is_training')
    word_embed = tf.placeholder(tf.float32, (None, max_sent_len, word_dim), name='word_embed')
    sent_len = tf.placeholder(tf.int32, (None,), name='sent_len')
    chars = tf.placeholder(tf.int32, (None, max_sent_len, max_word_len), name='chars')
    word_len = tf.placeholder(tf.int32, (None, max_sent_len), name='word_len')
    y_ = tf.placeholder(tf.float32, (None, 1), name='y')

    char_embedder = tf.get_variable('char_embedder', (character_size, char_dim))
    char_embed = tf.nn.embedding_lookup(char_embedder, chars)

    attn = han1_char_rnn(config, word_embed, sent_len, char_embed, word_len, n_units, is_training)

    proj_w = weight_variable([n_units, 1])
    proj_b = bias_variable([1])
    output = tf.matmul(attn, proj_w) + proj_b

    output = tf.reshape(output, [-1])
    ry = tf.reshape(y_, [-1])

    y_logits = tf.sigmoid(output) * 9 + 1
    
    predictions = y_logits
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.round(predictions), tf.round(ry))))

    # loss & optimizer
    loss = tf.losses.mean_squared_error(ry, y_logits)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    ##############################################################################################################

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)

    def get_feed_dict(we, sl, cs, wl, label, train=False):
        return {
            word_embed: we,
            sent_len: sl,
            chars: cs,
            word_len: wl,
            y_: label,
            is_training: train
        }

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = MovieReviewDataset(DATASET_PATH, max_sent_len, max_word_len)
        dataset_len = len(dataset)
        one_batch_size = dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1

        if config.debug :
            debugset = MovieReviewDataset(DEBUG3_PATH, max_sent_len, max_word_len)
            debugset_len = len(debugset)
            one_debug_size = debugset_len // config.batch
            if debugset_len % config.batch != 0:
                one_debug_size += 1

        train_step = 0
        best_acc = 0.0
        # epoch마다 학습을 수행합니다.
        start_time = time.time()
        for epoch in range(config.epochs):
            train_loss = 0.0
            train_acc = 0.0
            dataset.shuffle_dataset()
            for i, (we, sl, cs, wl, _, _, labels) in enumerate(_batch_loader(dataset, config.batch)):
                train_step += 1
                _, acc_, loss_ = sess.run([train_op, acc, loss],
                                          feed_dict=get_feed_dict(we,sl,cs,wl,labels,True))
                train_loss += float(loss_)
                train_acc += float(acc_)

                do_log = train_step % config.log_freq == 0
                do_debug = train_step % config.debug_freq == 0

                save_epoch = train_step / config.log_freq
                log_str = ""

                if do_log:
                    took_time = time.time() - start_time
                    print(('%d epoch , %d step | batch_acc: %.6f , batch_loss: %.6f (took %d sec for %d step)'
                                % (epoch, train_step, float(acc_), float(loss_), took_time, config.log_freq)))
                if config.debug and do_debug:
                    debug_loss = 0.0
                    debug_acc = 0.0
                    print("debug start ....")
                    for (we, sl, cs, wl, _, _, labels) in _batch_debug_loader(debugset, config.batch):
                        debug_acc_, debug_loss_ = sess.run([acc, loss],
                                                           feed_dict=get_feed_dict(we,sl,cs,wl,labels,False))
                        debug_loss += float(debug_loss_)
                        debug_acc += float(debug_acc_)
                    debug_acc = float(debug_acc / one_debug_size)
                    log_str += (' ---[DEBUG] acc: %.6f , loss: %.6f'% (debug_acc, float(debug_loss / one_debug_size)))
                    if debug_acc >= best_acc:
                        log_str += ' (got best! : %.4f)'%debug_acc
                        best_acc = debug_acc
                    print(log_str)

                if do_log:
                    nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(train_loss/one_batch_size), step=train_step)
                    # DONOTCHANGE (You can decide how often you want to save the model)
                    nsml.save(train_step)
                    start_time = time.time()

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)
