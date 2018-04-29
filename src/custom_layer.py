import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.rnn import MultiRNNCell, GRUCell
from tensorflow.contrib.cudnn_rnn import CudnnGRU, CudnnLSTM
from tensorflow.contrib.keras import activations
from tensorflow.contrib.keras import initializers

try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple

VERY_NEGATIVE_NUMBER = -1e29
    
def exp_mask(val, mask):
    mask = tf.cast(tf.sequence_mask(mask, tf.shape(val)[1]), tf.float32)
    mask = tf.expand_dims(mask,axis=-1)
    return val * mask + (1 - mask) * VERY_NEGATIVE_NUMBER

def join_mask(x_mask, key_mask, x_max_len, key_max_len):
    x_mask = tf.sequence_mask(x_mask, x_max_len)
    mem_mask = tf.sequence_mask(key_mask, key_max_len)
    join_mask = tf.logical_and(tf.expand_dims(x_mask, 2), tf.expand_dims(mem_mask, 1))
    return join_mask

def cudnn_birnn(x, num_units, num_layers, cell_name='gru', is_training=False, scope=None):

    max_length = x.get_shape()[1].value
    input_size = x.get_shape()[2].value

    if cell_name == 'gru':
        cellClass = CudnnGRU
    else:
        cellClass = CudnnLSTM

    with tf.variable_scope(scope or 'cudnn_bigru') as scope:
        with tf.variable_scope('fw') as scope:
            fw_gru = cellClass(num_layers, num_units, input_size)
            fw_h = tf.Variable(
                tf.random_uniform([num_layers,batch_size,num_units], -0.1, 0.1),
                validate_shape=False,
                name='fw_h')
            fw_params = tf.Variable(
                tf.random_uniform([fw_gru.params_size()], -0.1, 0.1),
                validate_shape=False,
                name='fw_params')
            fw_out, fw_outh = fw_gru(x, fw_h, fw_params)
        with tf.variable_scope('bw') as scope:
            bw_gru = cellClass(num_layers, num_units, input_size)
            bw_h = tf.Variable(
                tf.random_uniform([num_layers,batch_size,num_units], -0.1, 0.1),
                validate_shape=False,
                name='bw_h')
            bw_params = tf.Variable(
                tf.random_uniform([bw_gru.params_size()], -0.1, 0.1),
                validate_shape=False,
                name='bw_params')
            bw_out, bw_outh = bw_gru(x, bw_h, bw_params)
        concat_out  = tf.concat([fw_out,  bw_out ], axis=2)
        concat_outh = tf.concat([fw_outh, bw_outh], axis=1)

    return concat_out, concat_outh

def dropout(args, keep_prob, is_train, mode="recurrent"):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args

def cudnn_bigru(x, x_len, num_units, num_layers, keep_prob=0.7, is_training=False, scope=None):
    x = tf.transpose(x, [1, 0, 2])
    input_size = x.get_shape()[2].value
    grus = []
    inits = []
    dropout_mask = []
    for layer in range(num_layers):
        input_size_ = input_size if layer == 0 else 2 * num_units
        gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
        gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
        init_fw = tf.tile(tf.Variable(tf.zeros([1, 1, num_units])), [1, tf.shape(x)[1], 1])
        init_bw = tf.tile(tf.Variable(tf.zeros([1, 1, num_units])), [1, tf.shape(x)[1], 1])
        mask_fw = dropout(tf.ones([1, tf.shape(x)[1], input_size_], dtype=tf.float32), keep_prob=keep_prob, is_train=is_training)
        mask_bw = dropout(tf.ones([1, tf.shape(x)[1], input_size_], dtype=tf.float32), keep_prob=keep_prob, is_train=is_training)
        grus.append((gru_fw, gru_bw,))
        inits.append((init_fw, init_bw,))
        dropout_mask.append((mask_fw, mask_bw,))

    outputs = [x]
    hs = [None]
    for layer in range(num_layers):
        gru_fw, gru_bw = grus[layer]
        init_fw, init_bw = inits[layer]
        mask_fw, mask_bw = dropout_mask[layer]
        with tf.variable_scope("fw_{}".format(layer)):
            out_fw, out_fw_h = gru_fw(outputs[-1] * mask_fw, initial_state=(init_fw,))
        with tf.variable_scope("bw_{}".format(layer)):
            inputs_bw = tf.reverse_sequence(outputs[-1] * mask_bw, seq_lengths=x_len, seq_dim=0, batch_dim=1)
            out_bw, out_bw_h = gru_bw(inputs_bw, initial_state=(init_bw,))
            out_bw = tf.reverse_sequence(out_bw, seq_lengths=x_len, seq_dim=0, batch_dim=1)
        outputs.append(tf.concat([out_fw, out_bw], axis=2))
        hs.append(tf.concat([out_fw_h, out_bw_h], axis=1))
    res = tf.transpose(outputs[-1], [1, 0, 2])
    return res, hs[-1]

def bidirectional_rnn(cell_fw, cell_bw, inputs_embedded, input_lengths,
                      scope=None):
    """Bidirecional RNN with concatenated outputs and states"""
    with tf.variable_scope(scope or "birnn") as scope:
        ((fw_outputs,
          bw_outputs),
         (fw_state,
          bw_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                            cell_bw=cell_bw,
                                            inputs=inputs_embedded,
                                            sequence_length=input_lengths,
                                            dtype=tf.float32,
                                            swap_memory=True,
                                            scope=scope))
        outputs = tf.concat((fw_outputs, bw_outputs), 2)

        def concatenate_state(fw_state, bw_state):
            if isinstance(fw_state, LSTMStateTuple):
                state_c = tf.concat(
                    (fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
                state_h = tf.concat(
                    (fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
                state = LSTMStateTuple(c=state_c, h=state_h)
                return state
            elif isinstance(fw_state, tf.Tensor):
                state = tf.concat((fw_state, bw_state), 1,
                                  name='bidirectional_concat')
                return state
            elif (isinstance(fw_state, tuple) and
                    isinstance(bw_state, tuple) and
                    len(fw_state) == len(bw_state)):
                # multilayer
                state = tuple(concatenate_state(fw, bw)
                              for fw, bw in zip(fw_state, bw_state))
                return state

            else:
                raise ValueError(
                    'unknown state type: {}'.format((fw_state, bw_state)))


        state = concatenate_state(fw_state, bw_state)
        return outputs, state


def task_specific_attention(inputs, inputs_num, output_size,
                            initializer=layers.xavier_initializer(),
                            activation_fn=tf.tanh, scope=None):
    """
    Performs task-specific attention reduction, using learned
    attention context vector (constant within task of interest).

    Args:
        inputs: Tensor of shape [batch_size, units, input_size]
            `input_size` must be static (known)
            `units` axis will be attended over (reduced from output)
            `batch_size` will be preserved
        output_size: Size of output's inner (feature) dimension

    Returns:
        outputs: Tensor of shape [batch_size, output_dim].
    """
    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

    with tf.variable_scope('attention') as scope:
        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[output_size],
                                                   initializer=initializer,
                                                   dtype=tf.float32)
        input_projection = layers.fully_connected(inputs, output_size,
                                                  activation_fn=activation_fn,
                                                  scope=scope)

        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
        exp_mask(vector_attn, inputs_num)
        attention_weights = tf.nn.softmax(vector_attn, dim=1)
        weighted_projection = tf.multiply(input_projection, attention_weights)

        outputs = tf.reduce_sum(weighted_projection, axis=1)
        return outputs

def shallow_cnn(inputs_embedded, inputs_len, max_sequence_length, n_units, filter_sizes = [3,4,5], pool = "max_pool", scope=None):
    embedding_dim = inputs_embedded.get_shape()[-1].value
    inputs_embedded = tf.expand_dims(inputs_embedded, axis=-1)
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope("conv-maxpool-%s" % filter_size) as scope_:
            # Convolution Layer
            filter_shape = [filter_size, embedding_dim, 1, n_units]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[n_units]), name="b")
            conv = tf.nn.conv2d(
                inputs_embedded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            if pool == "max_pool":
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
            elif pool == "attention_pool":
                h = tf.squeeze(h, axis=2)
                pooled = task_specific_attention(h, inputs_len, n_units, scope=scope_)
            elif pool == "sum_pool":
                h = tf.squeeze(h, axis=2)
                pooled = tf.reduce_sum(h, axis=1)
            pooled_outputs.append(pooled)
    output = tf.concat(pooled_outputs, axis=-1)
    return output


def conv_block(x, shortcut, filter_width=3, filter_channel=64, is_training=True, scope=None):
    x_dim = x.get_shape()[2]
    with tf.variable_scope(scope or "conv_block") as scope:
        # Convolution 1 step
        W1 = tf.get_variable(
            name="W1",
            shape=[filter_width, x_dim, filter_channel],
            initializer=initializers.get("glorot_uniform"))
        x = tf.nn.conv1d(x, W1, stride=1, padding="SAME")
        x = layers.dropout(x, keep_prob=0.7, is_training=is_training)
        x = tf.nn.relu(x)
        # Convolution 2 step
        W2 = tf.get_variable(
            name="W2",
            shape=[filter_width, x_dim, filter_channel],
            initializer=initializers.get("glorot_uniform"))
        x = tf.nn.conv1d(x, W2, stride=1, padding="SAME")
        x = layers.dropout(x, keep_prob=0.7, is_training=is_training)
        x = tf.nn.relu(x)
        # Residual connection
        if shortcut != None:
            return shortcut + x
        else:
            return x

def pool_block(x, shortcut, filter_width=3, scope=None):
    with tf.variable_scope(scope or "pool_block") as scope:
        pool = tf.layers.max_pooling1d(inputs=x, pool_size=filter_width, strides=2, padding='same')
        if shortcut != None:
            shortcut = tf.layers.conv1d(inputs=x, filters=shortcut.get_shape()[2], kernel_size=1, strides=2, padding='same', use_bias=False)
            pool += shortcut
        pad_begin = (filter_width-1) // 2
        pad_end = (filter_width-1) - pad_begin
        pool = tf.pad(pool, [[0,0],[pad_begin, pad_end],[0,0]])
        return tf.layers.conv1d(inputs=pool, filters=pool.get_shape()[2]*2, kernel_size=1, strides=1, padding='valid',use_bias=False)
            
def vdcnn(x, filter_width=3, init_channel=64, num_layers=[2,2,2,2], use_shortcut=False, k=8, is_training=True, scope=None):
    layers = []
    x_dim = x.get_shape()[2]

    with tf.variable_scope("temp_conv"):
        filter_shape = [filter_width, x_dim, init_channel]
        W = tf.get_variable(name='temp_1', shape=filter_shape, initializer=initializers.get("glorot_uniform"))
        x = tf.nn.conv1d(x, W, stride=1, padding="SAME")
        layers.append(x)

    now_channel_size = init_channel
    
    for i, num_layer in enumerate(num_layers):
        for j in range(num_layer):
            with tf.variable_scope("%d_layer_%d_cnn" % (i, j)) as scope :
                shortcut = None
                if use_shortcut and i < len(num_layers) -1 :
                    shortcut = layers[-1]
                conv_ = conv_block(layers[-1], shortcut, filter_width, now_channel_size, is_training, scope)
                layers.append(conv_)

        if i == len(num_layers)-1:
            break
                
        with tf.variable_scope("%d_layer_pool" % (i)) as scope:
            shortcut = None
            if use_shortcut :
                shortcut = layers[-1]
            pool_ = pool_block(layers[-1], shortcut, filter_width, scope)
            layers.append(pool_)
            
        now_channel_size *= 2
                    
    k_pooled = tf.nn.top_k(tf.transpose(layers[-1], [0,2,1]), k=k, name='k_pool', sorted=False)[0]
    flatten = tf.reshape(k_pooled, (-1, now_channel_size * k))
    return flatten

def char_cnn_encoder(cx, filter_sizes, t_char_dims, max_word_num):
    max_char_num = cx.get_shape()[1]
    char_dim = cx.get_shape()[2]
    layers[7]
    for i in range(len(filter_sizes)):
        cx = tf.reshape(cx, [-1, max_char_num, char_dim])
        W = tf.get_variable(
            shape=(filter_sizes[i], char_dim, t_char_dims[i]),
            dtype=tf.float32,
            initializer=layers.xavier_initializer(),
        )
        cx = tf.nn.conv1d(
            value=cx,
            filters=W,
            stride=1,
            padding='VALID'
        )
        max_char_num = max_char_num - filter_sizes[i] + 1
        char_dim = t_char_dims[i]
    
    cx = task_specific_attention(cx, char_dim,
        initializer=layers.xavier_initializer(),
        activation_fn=tf.tanh, scope=None)

    return tf.reshape(cx, [-1, max_word_num, char_dim])

def char_rnn_encoder(config, cx, cx_num, max_word_num):
    max_word_num = cx.get_shape()[1].value
    max_char_num = cx.get_shape()[2].value
    char_dim = cx.get_shape()[3].value

    cell = MultiRNNCell([GRUCell(char_dim)] * config.cell_stack_count)
    cx = tf.reshape(cx, [-1, max_char_num, char_dim])
    cx_num = tf.reshape(cx_num, [-1])
    _, embed = bidirectional_rnn(cell, cell, cx, cx_num)
    cx_e = tf.reshape(embed, [-1, max_word_num, char_dim * config.cell_stack_count * 2])
    return cx_e

def char_han_encoder(config, cx, cx_num):
    max_word_num = cx.get_shape()[1].value
    max_char_num = cx.get_shape()[2].value
    char_dim = cx.get_shape()[3].value

    cell = MultiRNNCell([GRUCell(char_dim)] * config.cell_stack_count)
    cx = tf.reshape(cx, [-1, max_char_num, char_dim])
    cx_num = tf.reshape(cx_num, [-1])
    out, _ = bidirectional_rnn(cell, cell, cx, cx_num)
    attn = task_specific_attention(out, cx_num, char_dim)

    res = tf.reshape(attn, [-1, max_word_num, char_dim])

    return res


def dc_bilstm(config, x, x_nums, rnn_cells):
    '''
    Densely connected biLSTM
    '''
    max_x_num = x.get_shape()[1]
    layers = [x]
    layer_num = len(rnn_cells)

    for i in range(layer_num):
        layers.append(
            bidirectional_rnn(
                rnn_cells[i], rnn_cells[i],
                tf.concat(layers, axis=2),
                x_nums,
                scope=None
            )[0]
        )
    
    pooled = tf.layers.average_pooling1d(
        inputs=layers[-1],
        pool_size=[1,max_x_num,1],
        strides=1,
    )

    return tf.reshape(pooled, [-1, pooled.get_shape()[-1]])

def struct_self_attn(x, x_len, r, proj_nums):
    '''
    @param x: [-1, word_num, word_dim]
    @r
    @rnn_cell
    @proj_nums: int[]
    @return: [-1, r, word_dim], float32
    '''

    # A = softmax(W2 * tanh(W1 * H^T))
    attn = x
    for proj_num in proj_nums:
        attn = layers.fully_connected(attn, proj_num, activation_fn=None, biases_initializer=None)
        attn = tf.tanh(attn)
    attn = layers.fully_connected(attn, r, activation_fn=None, biases_initializer=None)

    attn = exp_mask(attn, x_len)
    attn = tf.nn.softmax(attn, dim=1) # [-1, word_num, r]

    M = tf.matmul(tf.transpose(attn, [0,2,1]), x) # [-1, r, word_dim]

    # Loss : ||(AA^T - I)||^2
    loss = tf.square(tf.norm(
        tf.matmul(attn, tf.transpose(attn, [0,2,1])) - tf.eye(attn.get_shape()[1].value),
        axis=(1,2),
        ord='fro',
    ))
    
    return M, loss

class ScaleDotAttention:

    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim

    def apply(self, is_train, x, keys, memories, x_mask=None, mem_mask=None):
        x_len_dim  = x.get_shape()[1]
        key_len_dim = keys.get_shape()[1]

        with tf.variable_scope("attention"):
            with tf.variable_scope("x_proj"):
                x_ = layers.fully_connected(x, self.hidden_dim, activation_fn=None, biases_initializer=None)
            with tf.variable_scope("key_proj"):
                key_ = layers.fully_connected(keys, self.hidden_dim, activation_fn=None, biases_initializer=None)
            dist_matrix = tf.matmul(x_, tf.transpose(key_,[0,2,1])) / (self.hidden_dim ** 0.5)
            joint_mask = join_mask(x_mask, mem_mask, x_len_dim, key_len_dim)
            if joint_mask is not None:
                dist_matrix += VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask, dist_matrix.dtype))
            query_prob = tf.nn.softmax(dist_matrix)
            select_query = tf.matmul(query_prob, memories)
            res = tf.concat([x, select_query], axis=2)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            gate_fc = layers.fully_connected(res, dim, activation_fn=tf.sigmoid, biases_initializer=None)
            gate = gate_fc.apply(is_train, res, x_mask)
            return res * gate

class DynamicCoAttention:

    def apply(self, is_train, x, keys, memories, x_mask=None, mem_mask=None):
        x_len_dim = x.get_shape()[1]
        key_len_dim = keys.get_shape()[1]

        with tf.variable_scope("attention"):
            dist_matrix = tf.matmul(x, tf.transpose(keys, [0, 2, 1]))
            joint_mask = join_mask(x_mask, mem_mask, x_len_dim, key_len_dim)
            if joint_mask is not None:
                dist_matrix += VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask, dist_matrix.dtype))
            query_prob = tf.nn.softmax(dist_matrix)
            passage_prob = tf.nn.sofmax(tf.transpose(dist_matrix, [0, 2, 1]))
            Passaage2Query = tf.matmul(passage_prob, x)
            Query2Passsage = tf.matmul(query_prob, tf.concat([memories, Passaage2Query], axis=2))
            return tf.concat([x, Query2Passsage], axis=2)

def tri_linear(x, keys):
    with tf.variable_scope("tri_linear_attn") as scope:
        key_w = tf.get_variable("key_w", shape=x.shape.as_list()[-1], initializer=initializers.get("glorot_uniform"), dtype=tf.float32)
        key_logits = tf.tensordot(keys, key_w, axes=[[2], [0]])  # (batch, key_len)
        x_w = tf.get_variable("input_w", shape=x.shape.as_list()[-1], initializer=initializers.get("glorot_uniform"), dtype=tf.float32)
        x_logits = tf.tensordot(x, x_w, axes=[[2], [0]])  # (batch, x_len)
        dot_w = tf.get_variable("dot_w", shape=x.shape.as_list()[-1], initializer=initializers.get("glorot_uniform"), dtype=tf.float32)
        x_dots = x * tf.expand_dims(tf.expand_dims(dot_w, 0), 0)
        dot_logits = tf.matmul(x_dots, keys, transpose_b=True)
        return dot_logits + tf.expand_dims(key_logits, 1) + tf.expand_dims(x_logits, 2)

def tri_linear_sattn(x, x_len):
    x_max_num = x.get_shape()[1]
    dist_matrix = tri_linear(x, x)
    dist_matrix += tf.expand_dims(tf.eye(x_max_num) * VERY_NEGATIVE_NUMBER, 0)  # Mask out self
    joint_mask = join_mask(x_len, x_len, x_max_num, x_max_num)
    dist_matrix += VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask, dist_matrix.dtype))
    select_probs = tf.nn.softmax(dist_matrix)
    response = tf.matmul(select_probs, x)
    return response

def shallow_wide_cnn(x, filter_widths, filter_channel):

    layers = []
    x_dim = x.get_shape()[2].value
    x_width = x.get_shape()[1].value
    for idx, filter_width in enumerate(filter_widths):
        with tf.variable_scope('filter_{}'.format(idx)) as scope:
            W = tf.get_variable(
                name='conv_W',
                shape=[filter_width, x_dim, filter_channel],
                initializer=initializers.get("glorot_uniform"))
            conved = tf.nn.conv1d(
                value=x,
                filters=W,
                stride=1,
                padding='VALID'
            )
            pooled = tf.reduce_max(conved, axis=1)
            layers.append(pooled)
    
    return tf.concat(layers, axis=1)

def dilated_causal_conv(x, filter_width=3, dilates=[1,2,4], scope=None):
    x_dim = x.get_shape()[-1].value
    with tf.variable_scope(scope or 'dilated_causal_conv'):
        conved = x
        for idx, dilate in enumerate(dilates):
            W = tf.get_variable(
                name='conv_filter_{}'.format(idx),
                shape=[filter_width, x_dim, x_dim],
                initializer=initializers.get('glorot_uniform')
            )
            W_norm = tf.nn.l2_normalize(W, [1,2])
            conved = tf.nn.convolution(
                input=conved,
                filter=W_norm,
                padding='SAME',
                strides=[1],
                dilation_rate=[dilate],
                name='conved_{}'.format(idx)
            )
    return conved

def tcn_block(x, is_training, filter_width=3, dilates=[1,2,4], keep_prob=0.7):
    '''
    reference: https://arxiv.org/pdf/1803.01271.pdf
    '''
    x_dim = x.get_shape()[-1].value
    with tf.variable_scope('dilated_causal_conv_1') as scope:
        conved_1 = dilated_causal_conv(x, filter_width, dilates, scope=scope)
    with tf.variable_scope('relu_dropout') as scope:
        output_1 = tf.nn.relu(conved_1)
        output_1 = layers.dropout(output_1, keep_prob=keep_prob, is_training=is_training)
    with tf.variable_scope('dilated_causal_conv_2') as scope:
        conved_2 = dilated_causal_conv(output_1, filter_width, dilates, scope=scope)
    with tf.variable_scope('relu_dropout') as scope:
        output_2 = tf.nn.relu(conved_2)
        output_2 = layers.dropout(output_2, keep_prob=keep_prob, is_training=is_training)
    
    conv_11_W = tf.get_variable(
        name='conv_11_filter',
        shape=[1,x_dim,x_dim],
        initializer=initializers.get('glorot_uniform')
    )
    conved_11 = tf.nn.convolution(
        input=x,
        filter=conv_11_W,
        padding='SAME',
        strides=[1],
        name='conved_11'
    )

    return tf.nn.relu(output_2 + conved_11)
