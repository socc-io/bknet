from custom_cell import BNLSTMCell
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, DropoutWrapper
from tensorflow.contrib.cudnn_rnn import CudnnGRU
import tensorflow.contrib.layers as layers
from tensorflow.python.layers.core import fully_connected
from tensorflow.contrib.keras import activations
from tensorflow.contrib.keras import initializers

from custom_layer import *

import tensorflow as tf


def rr_han(config, word_embed, sent_len, char_embed, word_len, syll_embed, syll_len, n_unit, is_training):
    '''
    HAN 1 layer with char rnn

    @ Input spec

    word_embed [batch_size, max_sent_len, word_dim]
    sent_len [batch_size]
    char_embed [batch_size, max_sent_len, max_word_len, char_dim]
    word_len [batch_size, max_sent_len]
    syll_embed [batch_size, max_sent_len, max_syll_len, syll_dim]
    syll_len [batch_size, max_sent_len]

    @ Output spec
    return [batch, n_unit]
    '''

    char_dim = config.char_dim
    syll_dim = config.syll_dim
    max_sent_len = config.max_sentence_length
    max_word_len = config.max_word_length
    max_syll_num = config.max_syll_num
    keep_prob = config.keep_prob
    rnn_dim = config.rnn_dim

    with tf.variable_scope('syll_rnn') as scope:
        cell_stack_count = 2
        syll_cell = MultiRNNCell([GRUCell(syll_dim)] * cell_stack_count)
        syll_embed = tf.cast(tf.reshape(syll_embed, [-1, max_syll_num, syll_dim]), tf.float32)
        syll_len = tf.reshape(syll_len, [-1])

        _, syll_rnn_embed = bidirectional_rnn(
            syll_cell, syll_cell, syll_embed, syll_len, scope=scope)

        syll_rnn_embed = tf.reshape(syll_rnn_embed, [-1, max_sent_len, syll_dim * 2 * cell_stack_count])

    with tf.variable_scope('char_rnn') as scope:
        cell_stack_count = 2
        char_cell =MultiRNNCell([GRUCell(char_dim)] * cell_stack_count)
        char_embed = tf.cast(tf.reshape(char_embed, [-1, max_word_len, char_dim]), tf.float32)
        word_len = tf.reshape(word_len, [-1])

        _, char_rnn_embed = bidirectional_rnn(
            char_cell, char_cell, char_embed, word_len, scope=scope)

        char_rnn_embed = tf.reshape(char_rnn_embed,[-1, max_sent_len, char_dim * 2 * cell_stack_count])

    word_char_concat = tf.concat([word_embed, char_rnn_embed, syll_rnn_embed], axis=2)

    with tf.variable_scope('embedding') as scope:
        word_char_embed = fully_connected(word_char_concat, rnn_dim, use_bias=True,
           activation=activations.get("relu"),
           kernel_initializer=initializers.get("glorot_uniform"))

        with tf.variable_scope('dropout'):
            word_char_embed = layers.dropout(
                word_char_embed, keep_prob=keep_prob,
                is_training=is_training,
            )

    with tf.variable_scope('encoder') as scope:
        cell = MultiRNNCell([GRUCell(rnn_dim)] * 3)
        encoder_output, _ = bidirectional_rnn(
        cell, cell, word_char_embed, sent_len, scope=scope)

        with tf.variable_scope('attention') as scope:
            attn_sum_output = task_specific_attention(
            encoder_output, n_unit, scope=scope)

        with tf.variable_scope('dropout'):
            attn_sum_output = layers.dropout(
            attn_sum_output, keep_prob=keep_prob,
            is_training=is_training,
        )

    return attn_sum_output


def cc_vdcnn(config, wx, w_num, cx, cx_num, sx, sx_num, n_unit, is_training):

    word_dim = config.word_dim
    max_word_num = config.max_word_num
    max_char_num = config.max_char_num
    max_syll_num = config.max_syll_num
    keep_prob = config.keep_prob

    with tf.variable_scope('cnn_embed_sx') as scope :
        sx_dim = 25
        sx_filter_sizes = [2,3]
        cnn_sx = tf.reshape(sx, [-1, sx.get_shape()[2], sx.get_shape()[3]])
        cnn_sx_len = tf.reshape(sx_num, [-1])
        cnn_sx = shallow_cnn(cnn_sx, cnn_sx_len, max_syll_num, sx_dim, filter_sizes=sx_filter_sizes, pool="max_pool", scope=scope)
        cnn_sx = tf.reshape(cnn_sx, [-1, max_word_num, sx_dim * len(sx_filter_sizes)])
        layers.dropout(cnn_sx, keep_prob=keep_prob, is_training=is_training)

    with tf.variable_scope('cnn_embed_cx') as scope:
        cx_dim = 25
        cx_filter_sizes = [3,4,5,6]
        cnn_cx = tf.reshape(cx, [-1, cx.get_shape()[2], cx.get_shape()[3]])
        cnn_cx_len = tf.reshape(cx_num, [-1])
        cnn_cx = shallow_cnn(cnn_cx, cnn_cx_len, max_char_num, cx_dim, filter_sizes=cx_filter_sizes, pool="max_pool", scope=scope)
        cnn_cx = tf.reshape(cnn_cx, [-1, max_word_num, cx_dim * len(cx_filter_sizes)])
        layers.dropout(cnn_cx, keep_prob=keep_prob, is_training=is_training)

    x = tf.concat([wx, cnn_sx, cnn_cx], axis=2) # [batch, word_num, 200 + 100 + 50]
    x = fully_connected(x, word_dim)
    x = layers.dropout( x, keep_prob=keep_prob, is_training=is_training)

    with tf.variable_scope('vdcnn') as scope:
        flat = vdcnn(x, filter_width=5, init_channel=128, num_layers=[2,2], use_shortcut=True, k=8, is_training=is_training, scope=scope)
        flat = layers.dropout(flat, keep_prob=keep_prob, is_training=is_training )

    return flat, None

def haha_sattn(config, wx, wx_num, cx, cx_num, sx, sx_num, n_unit, is_training):
    keep_prob = config.keep_prob
    max_word_num = wx.get_shape()[1].value

    with tf.variable_scope('char_encode') as scope:
        cx_e = char_han_encoder(config, cx, cx_num)
    with tf.variable_scope('syll_encode') as scope:
        sx_e = char_han_encoder(config, sx, sx_num)

    wx_e = tf.concat([wx, cx_e, sx_e], axis=2)
    wx_e = fully_connected(wx_e, config.rnn_dim)

    with tf.variable_scope('dropout'):
        wx_e = layers.dropout( wx_e, keep_prob=keep_prob, is_training=is_training,)

    with tf.variable_scope('word_rnn') as scope:
        cell = MultiRNNCell([GRUCell(config.rnn_dim)] * config.cell_stack_count)
        out, _ = bidirectional_rnn(cell, cell, wx_e, wx_num, scope=None)

    attn, loss = struct_self_attn(out, wx_num,  3, [25])
    attn = tf.reshape(attn, [-1, attn.get_shape()[1] * attn.get_shape()[2]])

    return attn, loss

def rr_sattn(config, wx, wx_num, cx, cx_num, sx, sx_num, n_unit, is_training):
    keep_prob = config.keep_prob
    max_word_num = wx.get_shape()[1].value

    with tf.variable_scope('char_encode') as scope:
        cx_e = char_rnn_encoder(config, cx, cx_num, max_word_num)
    with tf.variable_scope('syll_encode') as scope:
        sx_e = char_rnn_encoder(config, sx, sx_num, max_word_num)

    wx_e = tf.concat([wx, cx_e, sx_e], axis=2)
    wx_e = fully_connected(wx_e, config.rnn_dim)

    with tf.variable_scope('dropout'):
        wx_e = layers.dropout( wx_e, keep_prob=keep_prob, is_training=is_training,)

    with tf.variable_scope('word_rnn') as scope:
        cell = MultiRNNCell([GRUCell(config.rnn_dim)] * config.cell_stack_count)
        out, _ = bidirectional_rnn(cell, cell, wx_e, wx_num, scope=None)

    attn, loss = struct_self_attn(out, wx_num, 3, [25])
    attn = tf.reshape(attn, [-1, attn.get_shape()[1] * attn.get_shape()[2]])

    return attn, loss

def rr_rnn(config, wx, wx_num, cx, cx_num, sx, sx_num, n_unit, is_training):
    keep_prob = config.keep_prob
    max_word_num = wx.get_shape()[1].value

    with tf.variable_scope('char_neocde') as scope:
        cx_e = char_rnn_encoder(config, cx, cx_num, max_word_num)
    with tf.variable_scope('syll_encode') as scope:
        sx_e = char_rnn_encoder(config, sx, sx_num, max_word_num)
    wx_e = tf.concat([wx, cx_e, sx_e], axis=2)
    wx_e = fully_connected(wx_e, n_unit)

    with tf.variable_scope('dropout'):
        wx_e = layers.dropout(wx_e, keep_prob=keep_prob, is_training=is_training)

    cell = MultiRNNCell([GRUCell(n_unit)] * config.cell_stack_count)
    _, out = bidirectional_rnn(cell, cell, wx_e, wx_num)

    return out[0], None

def rr_swide(config, wx, wx_num, cx, cx_num, sx, sx_num, n_unit, is_training):

    max_word_num = wx.get_shape()[1]
    keep_prob = config.keep_prob

    with tf.variable_scope('char_encode') as scope:
        cx_e = char_rnn_encoder(config, cx, cx_num, max_word_num)
    with tf.variable_scope('syll_encode') as scope:
        sx_e = char_rnn_encoder(config, sx, sx_num, max_word_num)

    wx_e = tf.concat([wx, cx_e, sx_e], axis=2)
    wx_e = fully_connected(wx_e, config.rnn_dim)

    with tf.variable_scope('dropout'):
        wx_e = layers.dropout( wx_e, keep_prob=keep_prob, is_training=is_training,)
    
    with tf.variable_scope('shallow_cnn') as scope:
        swc = shallow_wide_cnn(wx_e, (3, 4, 5), 100)
    
    return swc, None

def tt_tcn_sattn(config, wx, wx_num, cx, cx_num, sx, sx_num, n_unit, is_training):

    max_word_num = wx.get_shape()[1].value
    max_char_num = cx.get_shape()[2].value
    max_syll_num = sx.get_shape()[2].value

    char_dim = cx.get_shape()[3].value
    syll_dim = sx.get_shape()[3].value

    with tf.variable_scope('char_encode') as scope:
        cx_reshape = tf.reshape(cx, [-1, max_char_num, char_dim])
        cx_e = tcn_block(cx_reshape, is_training)
        cx_e = task_specific_attention(cx_e, cx_num, char_dim)
        cx_e = tf.reshape(cx_e, [-1, max_word_num, char_dim])
    with tf.variable_scope('syll_encode') as scope:
        sx_reshape = tf.reshape(sx, [-1, max_syll_num, syll_dim])
        sx_e = tcn_block(sx_reshape, is_training)
        sx_e = task_specific_attention(sx_e, sx_num, syll_dim)
        sx_e = tf.reshape(sx_e, [-1, max_word_num, syll_dim])
    wx_e = tf.concat([wx, cx_e, sx_e], axis=2)
    wx_e = fully_connected(wx_e, n_unit)

    with tf.variable_scope('word_tcn_block') as scope:
        tcn_out = tcn_block(wx_e, is_training)

    attn, loss = struct_self_attn(tcn_out, wx_num, 3, [25])
    output = tf.reshape(attn, [-1, attn.get_shape()[1] * attn.get_shape()[2]])

    return output, loss
