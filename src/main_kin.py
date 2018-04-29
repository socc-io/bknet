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
from tensorflow.python.layers.core import fully_connected
import tensorflow.contrib.layers as layers
from tensorflow.contrib.keras import activations
from tensorflow.contrib.keras import initializers
import time

import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from dataset import KinQueryDataset, preprocess_kin
from word2vec import wordvec_lookup

DEBUG1_PATH = '../sample_data/kin/debug_1'
DEBUG2_PATH = '../sample_data/kin/debug_2'

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
        (_x1, _x1_len, _cx1, _cx1_len, _sx1, _sx1_len), (_x2, _x2_len, _cx2, _cx2_len, _sx2, _sx2_len) \
          = preprocess_kin(raw_data, config.max_word_num, config.max_char_num, config.max_syll_num, word_dim=config.word_dim)

        _x1_e = wordvec_lookup(_x1, config.word_dim)
        _x2_e = wordvec_lookup(_x2, config.word_dim)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(y_logits, feed_dict={
            x1_embedded: _x1_e,
            x1_len : _x1_len,
            cx1: _cx1,
            cx1_len : _cx1_len,
            sx1 : _sx1,
            sx1_len : _sx1_len,
            x2_embedded: _x2_e,
            x2_len : _x2_len,
            cx2: _cx2,
            cx2_len : _cx2_len,
            sx2 : _sx2,
            sx2_len : _sx2_len,
            is_training : False
        })
        clipped = np.array(pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(pred.flatten(), clipped.flatten()))

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


def get_feed_dict(a, b, l=None, is_train = False):
    return {
        x1_embedded: a[0],
        x1_len: a[1],
        cx1: a[2],
        cx1_len: a[3],
        sx1: a[4],
        sx1_len: a[5],
        x2_embedded: b[0],
        x2_len: b[1],
        cx2: b[2],
        cx2_len: b[3],
        sx2: b[4],
        sx2_len: b[5],
        y_: l,
        is_training: is_train
    }

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='train | test_local')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # Training options
    args.add_argument('--id', type=str, default="base")
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--epochs', type=int, default=200)
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
    args.add_argument('--char_dim', type=int, default=16)
    args.add_argument('--syll_dim', type=int, default=24)

    # Core options
    args.add_argument('--core_output_dim', type=int, default=50)
    args.add_argument('--syll_filter_size', type=int, default=3)
    args.add_argument('--rnn_dim', type=int, default=50)
    args.add_argument('--cell_stack_count', type=int, default=1)
    args.add_argument('--fc_dim', type=int, default=50)

    config = args.parse_args()

    print("Argument :", config)

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
    char_dim = config.char_dim
    word_dim = config.word_dim
    syll_dim = config.syll_dim
    rnn_dim = config.rnn_dim
    fc_dim = config.fc_dim

    is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    x1_embedded = tf.placeholder(tf.float32, [None, max_word_num, word_dim], name='x1')
    x1_len = tf.placeholder(tf.int32, [None], name = 'x1_len')
    cx1 = tf.placeholder(tf.int32, [None, max_word_num, max_char_num], name='cx1')
    cx1_len = tf.placeholder(tf.int32, [None, max_word_num], name = 'cx1_len')
    sx1 = tf.placeholder(tf.int32, [None, max_word_num, max_syll_num], name='sx1')
    sx1_len = tf.placeholder(tf.int32, [None, max_word_num], name='sx1_len')
    
    x2_embedded = tf.placeholder(tf.float32, [None, max_word_num, word_dim], name='x2')
    x2_len = tf.placeholder(tf.int32, [None], name = 'x2_len')
    cx2 = tf.placeholder(tf.int32, [None, max_word_num, max_char_num], name='cx2')
    cx2_len = tf.placeholder(tf.int32, [None, max_word_num], name = 'cx2_len')
    sx2 = tf.placeholder(tf.int32, [None, max_word_num, max_syll_num], name='sx2')
    sx2_len = tf.placeholder(tf.int32, [None, max_word_num], name='sx2_len')

    y_ = tf.placeholder(tf.float32, [None, 1], name = 'y')

    # char embedding
    char_embedding = tf.get_variable('char_embedding', [char_voca_size, char_dim])
    char1_embedded = tf.nn.embedding_lookup(char_embedding, cx1)
    char2_embedded = tf.nn.embedding_lookup(char_embedding, cx2)

    # syll embedding
    syll_embedding = tf.get_variable('syll_embedding', [syll_voca_size, syll_dim])

    syll1_embedded = tf.nn.embedding_lookup(syll_embedding, sx1)
    syll2_embedded = tf.nn.embedding_lookup(syll_embedding, sx2)

    # 여기 위로 이제 변경 금지 ㅋㅋㅋ
    ################################ㅋ#############################################################################

    import core_layer as core
    core_func = core.rr_rnn
    with tf.variable_scope("core" ):
        core_layer_output_1, core_loss_1 = \
            core_func(config, x1_embedded, x1_len, char1_embedded, cx1_len, syll1_embedded, sx1_len, core_output_dim, is_training)
    with tf.variable_scope("core" , reuse=True):
        core_layer_output_2, core_loss_2 = \
            core_func(config, x2_embedded, x2_len, char2_embedded, cx2_len, syll2_embedded, sx2_len, core_output_dim, is_training)

    #############################################################################################################
    # 여기 아래로 이제 변경 금지 ㅋㅋㅋ

    core_layer_output = tf.concat([core_layer_output_1, core_layer_output_2], axis=1)

    with tf.variable_scope("output"):
        output = fully_connected(core_layer_output, fc_dim, use_bias=True, activation=activations.get("relu"),
                             kernel_initializer=initializers.get("glorot_uniform"))
        output = layers.dropout(output, keep_prob=config.keep_prob, is_training=is_training)
        output = fully_connected(output, 1, use_bias=True, activation=None,
                             kernel_initializer=initializers.get("glorot_uniform"))

    y_logits = tf.sigmoid(output)
    predictions = tf.to_float(tf.greater_equal(y_logits, config.threshold))
    acc = tf.reduce_mean(tf.to_float(tf.equal(predictions, tf.round(y_))))

    # loss & optimizer
    loss = tf.reduce_mean(-(y_ * tf.log(tf.clip_by_value(y_logits,1e-4,1))) - (1 - y_) * tf.log(tf.clip_by_value(1 - y_logits, 1e-4,1)))
    if core_loss_1 != None and core_loss_2 != None:
        loss += (tf.reduce_mean(core_loss_1) + tf.reduce_mean(core_loss_2)) * 0.004
    train_op = tf.train.AdamOptimizer(config.lr).minimize(loss)
    ##############################################################################################################
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = KinQueryDataset(DATASET_PATH, max_word_num, max_char_num, max_syll_num, word_dim=word_dim)
        dataset_len = len(dataset)
        one_batch_size = dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1

        if config.debug :
            debugset = KinQueryDataset(DEBUG2_PATH, max_word_num, max_char_num, max_syll_num, word_dim=word_dim)
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
            for i, (left, right, label) in enumerate(_batch_loader(dataset, config.batch)):
                train_step += 1
                _, acc_, loss_ = sess.run([train_op, acc, loss],
                                          feed_dict=get_feed_dict(left, right, label, True))
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
                    for debug_left, debug_right, debug_label in _batch_debug_loader(debugset, config.batch):
                        debug_acc_, debug_loss_ = sess.run([acc, loss],
                                                           feed_dict=get_feed_dict(debug_left, debug_right, debug_label))
                        debug_loss += float(debug_loss_)
                        debug_acc += float(debug_acc_)
                    debug_acc = float(debug_acc / one_debug_size)
                    log_str += (' ---[DEBUG] acc: %.6f , loss: %.6f'% (debug_acc, float(debug_loss / one_debug_size)))
                    if debug_acc > best_acc:
                        log_str += ' (got best! : %.4f)' % debug_acc
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
        with open(os.path.join(DEBUG2_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)
