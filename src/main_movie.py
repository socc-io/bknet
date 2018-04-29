
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
from dataset import MovieReviewDataset, preprocess, DevDataset
from tensorflow.python.layers.core import fully_connected
import tensorflow.contrib.layers as layers
from tensorflow.contrib.keras import initializers
from tensorflow.contrib.keras import activations


from word2vec import wordvec_lookup
DEBUG1_PATH = '../sample_data/movie/debug_1'
DEBUG2_PATH = '../sample_data/movie/debug_2'

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
        infer_args = list(preprocess(
            raw_data,
            config.max_word_num,
            config.max_char_num,
            config.max_syll_num,
            word_dim=config.word_dim))
        infer_args[0] = wordvec_lookup(infer_args[0], config.word_dim)
        infer_args += (None, False) # Set label is None, is_training = False
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(output, feed_dict=get_feed_dict(*infer_args))
        pred = np.reshape(pred, [-1])
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

def _dev_loader(iterable, n=1):
    length = iterable._len
    offset = iterable.train_len
    for n_idx in range(offset, length, n):
        yield iterable[n_idx:min(n_idx+n, length)]

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

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='train | test_local')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # Training options
    args.add_argument('--id', type=str, default="base")
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--epochs', type=int, default=40)
    args.add_argument('--batch', type=int, default=800)
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--keep_prob', type=float, default=0.7)
    args.add_argument('--log_freq', type=int, default=30)

    # Debug options
    args.add_argument('--debug', action="store_true")
    args.add_argument('--dev', action="store_true")
    args.add_argument('--test', action="store_true")
    args.add_argument('--debug_freq', type=int, default=100)
    args.add_argument('--train_ratio', type=float, default=0.9)

    # Input options
    args.add_argument('--max_word_num', type=int, default=25)
    args.add_argument('--max_char_num', type=int, default=15)
    args.add_argument('--max_syll_num', type=int, default=10)

    args.add_argument('--char_voca_size', type=int, default=251)
    args.add_argument('--syll_voca_size', type=int, default=11173)

    args.add_argument('--word_dim', type=int, default=50)
    args.add_argument('--char_dim', type=int, default=50)
    args.add_argument('--syll_dim', type=int, default=50)

    args.add_argument('--use_word_embedding', type=int, default=1)
    args.add_argument('--max_embedding_num', type=int, default=10000)

    # Core options
    args.add_argument('--core_output_dim', type=int, default=50)
    args.add_argument('--syll_filter_size', type=int, default=3)
    args.add_argument('--rnn_dim', type=int, default=50)
    args.add_argument('--cell_stack_count', type=int, default=1)
    args.add_argument('--fc_dim', type=int, default=10)

    config = args.parse_args()

    if config.debug :
        DATASET_PATH = DEBUG1_PATH
    if config.test:
        DATASET_PATH = DEBUG2_PATH

    ##############################################################################################################
    # 모델의 specification
    core_output_dim = config.core_output_dim
    max_word_num = config.max_word_num
    max_char_num = config.max_char_num
    max_syll_num = config.max_syll_num
    syll_voca_size = config.syll_voca_size
    char_voca_size = config.char_voca_size
    word_dim = config.word_dim
    char_dim = config.char_dim
    syll_dim = config.syll_dim
    learning_rate = config.lr
    fc_dim = config.fc_dim

    use_word_embedding = config.use_word_embedding
    max_embedding_num = config.max_embedding_num

    is_training = tf.placeholder(tf.bool, name='is_training')
    if use_word_embedding:
        wx = tf.placeholder(tf.float32, (None, max_word_num, word_dim), name='wx')
    else:
        wx = tf.placeholder(tf.int32, (None, max_word_num), name='wx')
    wx_num      = tf.placeholder(tf.int32,   (None,), name='wx_num')
    cx_idx      = tf.placeholder(tf.int32,   (None, max_word_num, max_char_num), name='cx_idx')
    cx_num      = tf.placeholder(tf.int32,   (None, max_word_num), name='cx_num')
    sx_idx      = tf.placeholder(tf.int32,   (None, max_word_num, max_syll_num), name='sx_idx')
    sx_num      = tf.placeholder(tf.int32,   (None, max_word_num), name='sx_num')
    y_          = tf.placeholder(tf.float32, (None, 1), name='y_')

    if use_word_embedding:
        wx_inp = wx
    else:
        word_embedder = tf.get_variable('word_embedder', (max_embedding_num, word_dim))
        wx_inp = tf.nn.embedding_lookup(word_embedder, wx)

    char_embedder = tf.get_variable('char_embedder', (char_voca_size, char_dim))
    cx = tf.nn.embedding_lookup(char_embedder, cx_idx)

    syll_embedder = tf.get_variable('syll_embedder', (syll_voca_size, syll_dim))
    sx = tf.nn.embedding_lookup(syll_embedder, sx_idx)

    # 여기 위로 이제 변경 금지 ㅋㅋㅋ
    ################################ㅋ#############################################################################

    import core_layer as core
    core_layer_output, core_loss = core.rr_rnn(config, wx_inp, wx_num, cx, cx_num, sx, sx_num, core_output_dim, is_training)

    #############################################################################################################
    # 여기 아래로 이제 변경 금지 ㅋㅋㅋ
    with tf.variable_scope("output"):
        output = fully_connected(core_layer_output, fc_dim, use_bias=True, activation=activations.get("relu"), kernel_initializer=initializers.get("glorot_uniform"))
        output = layers.dropout(output, keep_prob=config.keep_prob, is_training=is_training)
        output = fully_connected(output, 1, use_bias=True, activation=None, kernel_initializer=initializers.get("glorot_uniform"))

    output_mult = tf.get_variable('prob_W', [1], initializer=tf.constant_initializer(9.))
    output_bias = tf.get_variable('prob_b', [1], initializer=tf.constant_initializer(1.))
    output = tf.nn.sigmoid(output) * output_mult + output_bias
    output_y = tf.reshape(output, [-1])

    predictions = tf.round(output_y)
    acc = tf.reduce_mean(tf.cast(tf.equal(predictions, y_), tf.float32))
    y_label = tf.reshape(y_, [-1])

    # loss & optimizer
    loss = tf.losses.mean_squared_error(y_label, output_y)
    if core_loss != None :
        loss += tf.reduce_mean(core_loss) * 0.004
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    ##############################################################################################################
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    def get_feed_dict(*args):
        feeder = {
            wx    : args[0],
            wx_num: args[1],
            cx_idx: args[2],
            cx_num: args[3],
            sx_idx: args[4],
            sx_num: args[5],
            is_training: args[7],
        }
        if args[6] is not None:
            feeder[y_] = args[6]
        return feeder

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = MovieReviewDataset(
            DATASET_PATH,
            max_word_num,
            max_char_num,
            max_syll_num,
            word_dim=word_dim,
            is_partial_dev=config.dev,
            use_word_embedding=use_word_embedding,
            max_embedding_num=max_embedding_num)
        dataset_len = len(dataset)
        one_batch_size = dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1

        if config.debug or config.dev:
            if config.dev:
                debugset = DevDataset(dataset)
            elif config.debug:
                debugset = MovieReviewDataset(
                    DEBUG2_PATH,
                    max_word_num,
                    max_char_num,
                    max_syll_num,
                    word_dim=word_dim,
                    use_word_embedding=use_word_embedding,
                    max_embedding_num=max_embedding_num)
            debugset_len = len(debugset)
            one_debug_size = debugset_len // config.batch
            if debugset_len % config.batch != 0:
                one_debug_size += 1

        train_step = 0
        best_mse = 99999.0
        # epoch마다 학습을 수행합니다.
        start_time = time.time()
        for epoch in range(config.epochs):
            train_loss = 0.0
            train_acc = 0.0
            dataset.shuffle_dataset()
            for i, batch_args in enumerate(_batch_loader(dataset, config.batch)):
                train_step += 1
                batch_args += (True,) # Set is_training = True
                _, acc_, loss_ = sess.run([train_op, acc, loss],
                    feed_dict=get_feed_dict(*batch_args))
                train_loss += float(loss_)
                train_acc += float(acc_)

                do_log = train_step % config.log_freq == 0
                do_debug = train_step % config.debug_freq == 0

                save_epoch = train_step / config.log_freq
                log_str = ""

                if do_log:
                    took_time = time.time() - start_time
                    print(('%d epoch , %d step | batch_acc: %.6f , batch_mse: %.6f (took %d sec for %d step)'
                                % (epoch, train_step, float(acc_), float(loss_), took_time, config.log_freq)))
                if (config.debug or config.dev) and do_debug:
                    debug_loss = 0.0
                    debug_acc = 0.0
                    print("debug start ....")
                    for debug_args in _dev_loader(dataset, config.batch):
                        debug_args += (False,) # Set is_training = False
                        debug_acc_, debug_loss_, debug_pred_ = sess.run([acc, loss, predictions], feed_dict=get_feed_dict(*debug_args))
                        debug_loss += float(debug_loss_)
                        debug_acc += float(debug_acc_)
                    log_str += (' ---[DEBUG] acc: %.6f , mse: %.6f'% (float(debug_acc / one_debug_size), float(debug_loss / one_debug_size)))
                    if debug_loss < best_mse:
                        log_str += ' (got best mse! : %.4f)'% (float(debug_loss / one_debug_size))
                        best_mse = debug_loss
                    print(log_str)

                if do_log:
                    nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(train_loss/i+1), step=train_step)
                    # DONOTCHANGE (You can decide how often you want to save the model)
                    nsml.save(train_step)
                    start_time = time.time()

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DEBUG2_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)
