#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.python.framework import dtypes


""""""""""""""""""""""""""""""" General Purpose Functions """""""""""""""""""""""""""""""


def filter_indexed_slices(in_indexed_slices, split_idx, filter_before=True, name=None):
    """
    Filter the indexed slices so that some designated indices will disappear.
    Usually used to control embedding update.
    :param in_indexed_slices: original IndexedSlices instance
    :param split_idx: the idx to split the whole index range
    :param filter_before: filter out all the indices before "split_idx" (not included),
            otherwise filter out all the indices after "split_idx" (not included)
    :param name:    name of this scope
    :return:
    """
    with tf.name_scope(name, default_name="FilterIndexedSlices", values=[in_indexed_slices]):
        if filter_before:
            mask = tf.sign(in_indexed_slices.indices - split_idx) + 1   # 0 means filter; 1 and 2 means keep
        else:
            mask = -tf.sign(in_indexed_slices - split_idx) + 1   # 0 means filter; 1 and 2 means keep
        mask = tf.cast(mask, tf.bool)

        inner_indices = tf.squeeze(tf.where(mask), squeeze_dims=[1])
        indices = tf.gather(in_indexed_slices.indices, inner_indices)
        values = tf.gather(in_indexed_slices.values, inner_indices)

        return tf.IndexedSlices(values, indices)


def scalar_array_to_tensor(array, name=None):
    with tf.name_scope(name, default_name="ScalarArrayToTensor", values=[array]):
        for i in range(len(array)):
            if type(array[i]) == int:
                array[i] = tf.constant([array[i]])
            else:   # must be rank 0 tensor
                array[i] = tf.expand_dims(array[i], 0)
        return tf.concat(0, array)


def expand_dim_for_tensor_list(tensor_list, dim_array):
    """
    Expand dimension wrapper for tensorlist
    :param tensor_list  a list of tensor to apply tf.expand_dims()
    :param dim_array    dimensions to expand
    :return processed tensor_list
    """
    res_tensor_list = []
    for tensor in tensor_list:
        res_tensor = tensor
        for dim in dim_array:
            res_tensor = tf.expand_dims(res_tensor, dim)
        res_tensor_list.append(res_tensor)

    return res_tensor_list


def embed_index_array(input_data, embedding, seq_len, is_training=False, keep_prob=1):
    with tf.device("/cpu:0"):
        # Convert lookup result (3-dimension) into a list of matrices (batch_size * word_vec_size)
        word_embeddings = tf.split(1, seq_len, tf.nn.embedding_lookup(embedding, input_data))
        word_embeddings = [tf.squeeze(embed_, [1]) for embed_ in word_embeddings]

    if is_training and keep_prob < 1:
        word_embeddings = [tf.nn.dropout(input_, keep_prob) for input_ in word_embeddings]
    return word_embeddings


def matmul_3d_by_2d(in_tensor, in_matrix, name):
    """
    Multiply each piece of a 3D tensor by a 2D matrix
    :param in_tensor: input tensor, size (batch_size, i, j)
    :param in_matrix: input matrix, size (j, k)
    :return: 3D tensor, size (batch_size, i, k)
    """
    with tf.name_scope(name, default_name="Matmul_3D_by_2D", values=[in_tensor, in_matrix]):
        tensor_shape = tf.shape(in_tensor)
        matrix_shape = tf.shape(in_matrix)
        return tf.reshape(tf.matmul(
            tf.reshape(in_tensor, [tensor_shape[0]*tensor_shape[1], tensor_shape[2]]), in_matrix),
            [tensor_shape[0], tensor_shape[1], matrix_shape[1]])


def batch_diag_part(in_tensor, batch_size):
    """
    Return the diagonal part of each matrix in a batch
    :param in_tensor: 3D tensor, size: (batch_size, i, i)
    :param batch_size: batch size
    :return: 2D tensor, size: (batch_size, i)
    """
    tensor_list = tf.split(split_dim=0, num_split=batch_size, value=in_tensor)
    tensor_list = [tf.expand_dims(tf.diag_part(tf.squeeze(t, [0])), 0) for t in tensor_list]
    return tf.concat(0, tensor_list)


def batch_trace(in_tensor, batch_size, absolute_value=False):
    """
    Return the trace of each matrix in a batch
    :param in_tensor: 3D tensor, size: (batch_size, i, i)
    :param batch_size: batch size
    :param absolute_value: use the absolute value of the diagonals or not
    :return: batch_trace, size: (batch_size)
    """
    # mask = tf.constant(value=np.identity(diag_num))
    # mask = tf.tile(mask, [batch_size, 1, 1])
    diag_matrix = batch_diag_part(in_tensor, batch_size)
    if absolute_value:
        diag_matrix = tf.abs(diag_matrix)
    return tf.reduce_sum(diag_matrix, reduction_indices=[1], keep_dims=False)


def normalize_to_sum_one(in_tensor, tensor_rank, sum_one_indices_cnt=0):
    """
    Normalize the input tensor so that the designated dimensions sum to one
    :param in_tensor: input tensor
    :param tensor_rank: rank of the input tensor
    :param sum_one_indices_cnt: 'sum_one_indices_cnt' indices to sum to one (should be consecutive from left to right)
    :return: normalized in_tensor, same shape
    """
    if sum_one_indices_cnt == 0:
        total_sum = tf.reduce_sum(in_tensor)
        return in_tensor / total_sum

    tensor_shape = tf.shape(in_tensor)
    sum_tensor = tf.reduce_sum(in_tensor, reduction_indices=range(sum_one_indices_cnt, tensor_rank), keep_dims=True)
    denominator = tf.tile(sum_tensor, tf.concat(0, [tf.ones([sum_one_indices_cnt], dtype=dtypes.int32),
                                                    tensor_shape[sum_one_indices_cnt:]]))
    return in_tensor / denominator


""""""""""""""""""""""""""""""" MLP """""""""""""""""""""""""""""""


def mlp(in_tensor, hidden_size_array, activation=tf.nn.relu, bias_initializer=tf.zeros_initializer,
        batch_norm=False, keep_prob=1.0, is_training=False, scope_name="Full_Connect"):
    """
    Mlp layer
    :param in_tensor:           (batch_size * vec_len)
    :param hidden_size_array:   [input_vec_len, 1st_layer_len, 2nd_layer_len, ...]
    :param activation:          activation function name (e.g. tf.nn.relu)
    :param bias_initializer:    bias initializer (e.g. tf.zeros_initializer, useless if batch_norm=True)
    :param batch_norm:          use batch normalization or not
    :param keep_prob:           dropout keep probability for each layer (except for the input layer, including output)
    :param is_training:         is training or not (used for dropout)
    :param scope_name:          scope name (default Full_Connect)
    :return mlp result tensor
    """
    if batch_norm:
        bias_initializer = None
    with tf.variable_scope(scope_name):
        temp_mlp_input = in_tensor
        for i in range(len(hidden_size_array)-1):
            with tf.variable_scope("Layer_{0}".format(i)):
                # Old sytle
                # temp_w = tf.get_variable(
                #     name="W",  shape=[hidden_size_array[i], hidden_size_array[i+1]],
                #     initializer=tf.contrib.layers.xavier_initializer(uniform=False)
                # )
                # temp_b = tf.get_variable(
                #     name="bias", initializer=tf.zeros_initializer(shape=[hidden_size_array[i+1]]),
                # )
                # temp_output = tf.nn.xw_plus_b(temp_mlp_input, temp_w, temp_b)
                # temp_output = activation(temp_output)

                # new style
                temp_output = tf.contrib.layers.fully_connected(
                    inputs=temp_mlp_input, num_outputs=hidden_size_array[i+1], activation_fn=activation,
                    normalizer_fn=None, biases_initializer=bias_initializer,
                    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                )
                if batch_norm:
                    temp_output = tf.contrib.layers.batch_norm(
                        temp_output, decay=0.999, center=True, scale=True, updates_collections=None,
                        is_training=is_training, reuse=tf.get_variable_scope().reuse, trainable=True, scope="Batch_Norm"
                    )

                temp_mlp_input = temp_output

                if is_training and keep_prob < 1:
                    temp_mlp_input = tf.nn.dropout(temp_mlp_input, keep_prob)
    return temp_mlp_input


""""""""""""""""""""""""""""""" RNN """""""""""""""""""""""""""""""


def get_rnn(X, rnn_size, seq_len, batch_size, num_layers=1, input_keep_prob=1.0, output_keep_prob=1.0, is_training=False,
            cell_name="BasicLSTM", bidirectional=False):
    """
    Get rnn given input data.
    Note that if use bidirectional RNN, the outputs contains only h rather than (h, c)
    :param X:                   batch_size * seq_len * vec_len
    :param rnn_size:            number of cells in each layer of rnn
    :param seq_len:             sequence length
    :param batch_size:          batch size
    :param num_layers:          number of layers
    :param input_keep_prob:     input keep probability in dropout
    :param output_keep_prob:    input keep probability in dropout
    :param is_training:         is training or not (used for dropout)
    :param cell_name:           cell to use (GRU, LSTM)
    :param bidirectional:       use bidirectional RNN or not
    :return:    rnn object
    """
    with tf.device("/cpu:0"):
        # Convert input tensor to python list (along the sequence length dimention)
        word_embeddings = tf.split(1, seq_len, X)
        word_embeddings = [tf.squeeze(embed_, [1]) for embed_ in word_embeddings]

    # if is_training and keep_prob < 1:
    #     word_embeddings = [tf.nn.dropout(input_, keep_prob) for input_ in word_embeddings]

    def get_cell():
        if cell_name == "GRU":      # GRU
            cell = rnn_cell.GRUCell(rnn_size)
        elif cell_name == "LSTM":   # LSTM
            cell = rnn_cell.LSTMCell(rnn_size, tf.shape(X)[2])
        else:
            cell = rnn_cell.BasicLSTMCell(rnn_size)
        if is_training and (input_keep_prob < 1 or output_keep_prob < 1):
            cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)
        cell = rnn_cell.MultiRNNCell([cell] * num_layers)
        initial_state = cell.zero_state(batch_size, tf.float32)
        return cell, initial_state

    if bidirectional:
        with tf.variable_scope("forward"):
            cell_fw, initial_state_fw = get_cell()
        with tf.variable_scope("backward"):
            cell_bw, initial_state_bw = get_cell()
        return rnn.bidirectional_rnn(cell_fw, cell_bw, word_embeddings,
                                     initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw)
    else:
        cell, initial_state = get_cell()
        return rnn.rnn(cell, word_embeddings, initial_state=initial_state)


def get_dynamic_rnn(X, rnn_size, seq_len, num_layers=1, input_keep_prob=1.0, output_keep_prob=1.0,
                    is_training=False, cell_name="GRU", bidirectional=False, swap_memory=False):
    """
    Get rnn given input data.
    Note that if use bidirectional RNN, the outputs contains only h rather than (h, c)
    :param X:                   batch_size * seq_len * vec_len
    :param rnn_size:            number of cells in each layer of rnn
    :param seq_len:             list of sequence length of each sequence
    :param num_layers:          number of layers
    :param input_keep_prob:     input keep probability in dropout
    :param output_keep_prob:    input keep probability in dropout
    :param is_training:         is training or not (used for dropout)
    :param cell_name:           cell to use (GRU, LSTM)
    :param bidirectional:       use bidirectional RNN or not
    :param swap_memory:         Swap the tensors produced in forward inference but needed for back prop from GPU to CPU.
    :return:    rnn object
    """
    batch_size = tf.shape(X)[0]
    if bidirectional:
        rnn_size /= 2

    def get_cell():
        if cell_name == "GRU":      # GRU
            _cell = rnn_cell.GRUCell(rnn_size)
        elif cell_name == "LSTM":   # LSTM
            _cell = rnn_cell.LSTMCell(rnn_size, tf.shape(X)[2])
            # cell = rnn_cell.BasicLSTMCell(rnn_size)
        else:
            raise ValueError("Unrecognized Cell Name: {0}".format(cell_name))
        if is_training and (input_keep_prob < 1 or output_keep_prob < 1):
            _cell = rnn_cell.DropoutWrapper(_cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)
        _cell = rnn_cell.MultiRNNCell([_cell] * num_layers)
        _initial_state = _cell.zero_state(batch_size, tf.float32)
        return _cell, _initial_state

    if bidirectional:
        with tf.variable_scope("forward"):
            cell_fw, initial_state_fw = get_cell()
        with tf.variable_scope("backward"):
            cell_bw, initial_state_bw = get_cell()
        return rnn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, X, sequence_length=seq_len,
            initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw,
            dtype=None, parallel_iterations=None, swap_memory=swap_memory, time_major=False, scope=None)
    else:
        cell, initial_state = get_cell()
        return rnn.dynamic_rnn(
            cell, X, sequence_length=seq_len, initial_state=initial_state,
            dtype=None, parallel_iterations=None, swap_memory=swap_memory, time_major=False, scope=None)



""""""""""""""""""""""""""""""" CNN """""""""""""""""""""""""""""""


def _prepare_input_for_conv1d_on_2d(input, window, stride=1, padding=None):
    input_shape = tf.shape(input)
    if padding:
        padding_tensor = tf.expand_dims(tf.expand_dims(padding, 0), 0)   # (1, 1, len(padding))
        padding_tensor = tf.tile(padding_tensor, scalar_array_to_tensor([input_shape[0], int(math.floor(window/2)), 1]))
        input = tf.concat(1, [padding_tensor, input])
        input = tf.concat(1, [input, padding_tensor])
        input_shape = tf.shape(input)

    # Convert original input to shape (batch, width-window+1, height*window)
    width_first_input = tf.transpose(input, [1, 2, 0])  # (width, height, batch)
    to_concat = []
    for i in range(window):
        temp_index = tf.range(start=i, limit=input_shape[1]-window+i+1, delta=stride)
        to_concat.append(tf.gather(width_first_input, temp_index))
    new_input = tf.concat(1, to_concat)
    new_input = tf.transpose(new_input, [2, 0, 1])  # (batch, width-window+1, height*window)
    return new_input


def conv1d_on_2d(input, filter, window, stride=1, b=None, padding=None, name=None):
    """
    Convolution on 2d matrix, but the filter has the same height as the input.
    Mainly used for text.
    :param input:   (batch, width, height), width is sentence length, height is word vector length
    :param window:  window size of the filter
    :param filter:  (width, height, out_channel), should has the same height as input, width should be odd number
    :param stride:  stride of the filter
    :param b:       offset vector, length is out_channel
    :param padding: padding vector (None for no padding)
    :param name:    operator name
    :return:    convolved input, (batch, width-window+1, out_channel)
    """
    with tf.name_scope(name, default_name="TextConvolution", values=[input]):
        input_shape = tf.shape(input)
        filter_shape = tf.shape(filter)

        new_input = _prepare_input_for_conv1d_on_2d(input, window, stride, padding)

        # Prepare filter
        new_filter = tf.reshape(filter, scalar_array_to_tensor([filter_shape[0]*filter_shape[1], filter_shape[2]]))
        new_filter = tf.expand_dims(new_filter, 0)
        new_filter = tf.tile(new_filter, scalar_array_to_tensor([input_shape[0], 1, 1]))

        output = tf.batch_matmul(new_input, new_filter)
        output_shape = tf.shape(output)
        if b:
            new_b = tf.expand_dims(tf.expand_dims(b, 0), 0)
            new_b = tf.tile(new_b, scalar_array_to_tensor([output_shape[0], output_shape[1], 1]))
            output = output + new_b

        return output


def pooling1d_on_2d(input_data, window, stride, padding=None, method="max", name=None):
    """
    Pooling on 2d matrix, but only pooling along the line.
    Mainly used for text.
    :param input_data:  (batch, width, height), width is sentence length, height is word vector length
    :param window:      window size of the filter
    :param stride:      stride of the filter
    :param padding:     padding vector (None for no padding)
    :param name:        operator name
    :param method:      max, avg
    :return:    pooled data, (batch, width-window+1, height)
    """
    with tf.name_scope(name, default_name="Pooling", values=[input_data]):
        # Change input to size: (batch, width-window+1, height*window)
        new_input = _prepare_input_for_conv1d_on_2d(input_data, window, stride, padding)
        input_shape = tf.shape(new_input)
        new_input = tf.reshape(new_input, scalar_array_to_tensor([
            input_shape[0], input_shape[1], tf.to_int32(input_shape[2]/window), window]))

        if method == "max":
            output = tf.reduce_max(new_input, reduction_indices=3)
        return output


def get_cnn_for_text(input_data, conv_window, filter_num, pooling_window, activation_fn=tf.nn.relu,
                     padding_vec=None, use_batch_norm=False, bias_initializer=tf.zeros_initializer,
                     keep_prob=1, is_training=False):
    """
    Get a CNN model for text data
    :param input_data:  (batch, width, height), width is sequence length, height is word embedding length
    :param conv_window: list of integers to control window size in each layer
    :param filter_num:  list of integers to control filter number in each layer (first element is input vec length)
    :param pooling_window:  list of integers to control pooling window size in each layer
    (-1 for pool them all, 0 for no pooling)
    :param padding_vec: padding vector (if not None, pad the sequence so that output has the same length as the input)
    :param use_batch_norm:  use batch normalization or not
    :param bias_initializer: bias initializer (e.g. tf.zeros_initializer, useless if batch_norm=True)
    :param keep_prob:   dropout keep probability
    :param is_training: is training or not (used for dropout)
    :return:    (batch, output_vec_length)
    """
    # Do padding
    input_shape = tf.shape(input_data)
    if padding_vec is not None:
        padding_tensor = tf.expand_dims(tf.expand_dims(padding_vec, 0), 0)   # (1, 1, len(padding))
        padding_tensor = tf.tile(padding_tensor, scalar_array_to_tensor([input_shape[0], int(math.floor(conv_window[0]/2)), 1]))
        input_data = tf.concat(1, [padding_tensor, input_data])
        input_data = tf.concat(1, [input_data, padding_tensor])

    if use_batch_norm:
        bias_initializer = None

    temp_input = tf.expand_dims(input_data, -1)     # expand the channel dimension (last dimension)
    num_layers = len(conv_window)
    for i in range(num_layers):
        with tf.variable_scope("Layer_{0}".format(i)):
            # Convolution
            if i == 0:
                filter_shape = [conv_window[i], filter_num[i], 1, filter_num[i+1]]
            else:
                filter_shape = [conv_window[i], 1, filter_num[i], filter_num[i+1]]  # height is 1 after the first conv

            # New style
            # temp_conv_output is of size (batch_size, curr_seq_len, 1, curr_filter_num)
            temp_conv_output = tf.contrib.layers.convolution2d(
                    inputs=temp_input, num_outputs=filter_num[i+1], kernel_size=filter_shape[:2], stride=[1, 1],
                    padding='VALID', activation_fn=activation_fn,
                    normalizer_fn=None,
                    biases_initializer=bias_initializer,
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                    # biases_initializer=tf.random_uniform_initializer(-config.conv_init_scale, config.conv_init_scale),
                    # weights_initializer=tf.random_uniform_initializer(-config.conv_init_scale, config.conv_init_scale),
                )
            if use_batch_norm:
                temp_conv_output = tf.contrib.layers.batch_norm(
                    temp_conv_output, decay=0.999, center=True, scale=True, updates_collections=None, epsilon=0.001,
                    is_training=is_training, trainable=True, scope="Batch_Norm")

            # Old style
            # temp_filter = tf.get_variable(
            #     name="filter", shape=filter_shape,
            #     initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
            # )
            # b = tf.get_variable(
            #     name="bias", initializer=tf.zeros_initializer(shape=[filter_num[i + 1]])
            # )
            # temp_conv_output = tf.nn.conv2d(temp_input, temp_filter, strides=[1, 1, 1, 1], padding="VALID")

            # tmp_seq_len = tmp_seq_len - conv_window[i] + 1

            # Pooling
            # final temp_pooling_output is of size (batch_size, 1, 1, final_conv_kernel_num)
            if pooling_window[i] == -1:
                # temp_pooling_output = tf.nn.max_pool(
                #     temp_conv_output, ksize=[1, tmp_seq_len, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
                temp_pooling_output = tf.reduce_max(temp_conv_output, reduction_indices=[1], keep_dims=True)
            elif pooling_window[i] > 0:
                temp_pooling_output = tf.nn.max_pool(
                    temp_conv_output, ksize=[1, pooling_window[i], 1, 1], strides=[1, 1, 1, 1], padding="VALID")
            else:       # no pooling
                temp_pooling_output = temp_conv_output


            # Old style
            # temp_pooling_output = tf.nn.bias_add(temp_pooling_output, b)
            # temp_input = activation_fn(temp_pooling_output)

            # New style
            temp_input = temp_pooling_output

            if is_training and keep_prob < 1:
                temp_input = tf.nn.dropout(temp_input, keep_prob)

    return tf.squeeze(temp_input, squeeze_dims=[1, 2])


""""""""""""""""""""""""""""""" Classifier """""""""""""""""""""""""""""""


def softmax_classifier(in_tensor, label_num, scope_name="Softmax_Classifier"):
    """
    Softmax Classifier
    :param in_tensor: input feature tensor (size: (batch_size, feature_vec_length))
    :param label_num: number of labels
    :param scope_name: scope name (default Softmax_Classifier)
    :return: softmax_outputs, logits (both in size: (batch_size, label_num))
    """
    with tf.variable_scope(scope_name):
        tensor_shape = in_tensor.get_shape()
        softmax_w = tf.get_variable(
            name="softmax_w", shape=[tensor_shape[-1], label_num],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        )
        softmax_b = tf.get_variable(name="softmax_b", initializer=tf.zeros_initializer(shape=[label_num]))
        logits = tf.nn.xw_plus_b(in_tensor, weights=softmax_w, biases=softmax_b)
        softmax_outputs = tf.nn.softmax(logits)
    return softmax_outputs, logits


def sigmoid_classifier(in_tensor, label_num, scope_name="Sigmoid_Classifier"):
    """
    Sigmoid Classifier
    :param in_tensor: input feature tensor (size: (batch_size, feature_vec_length))
    :param label_num: number of labels
    :param scope_name: scope name (default Sigmoid_Classifier)
    :return: softmax_outputs, logits (both in size: (batch_size, label_num))
    """
    with tf.variable_scope(scope_name):
        tensor_shape = in_tensor.get_shape()
        sigmoid_w = tf.get_variable(
            name="sigmoid_w", shape=[tensor_shape[-1], label_num],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        )
        sigmoid_b = tf.get_variable(name="sigmoid_b", initializer=tf.zeros_initializer(shape=[label_num]))
        logits = tf.nn.xw_plus_b(in_tensor, weights=sigmoid_w, biases=sigmoid_b)
        sigmoid_outputs = tf.nn.sigmoid(logits)
    return sigmoid_outputs, logits





""""""""""""""""""""""""""""""" Loss """""""""""""""""""""""""""""""


def cross_entropy_with_probs(pred_probs, target_probs, name=None):
    """
    Cross entropy loss function
    :param pred_probs: probability tensor produced by softmax, size: [batch_size, label_num]
    :param target_probs: gold labels, size: [batch_size, label_num]
    :param name: scope name
    :return: loss vector, size: [batch_size]
    """
    with tf.name_scope(name, "SoftmaxCrossEntropyWithProb", [pred_probs, target_probs]):
        crossent_loss = -tf.reduce_sum(target_probs * tf.log(pred_probs), reduction_indices=[1], keep_dims=False)
        return crossent_loss


def sparse_cross_entropy_with_probs(probs, labels, name=None):
    """
    Sparse cross entropy loss function
    :param probs: probability tensor produced by softmax, size: [batch_size, label_num]
    :param labels: gold labels, size: [batch_size]
    :param name: scope name
    :return: loss vector, size: [batch_size]
    """
    with tf.name_scope(name, "SparseSoftmaxCrossEntropyWithProb", [probs, labels]):
        probs_shape = tf.shape(probs)
        one_hot_labels = tf.one_hot(labels, probs_shape[1])
        crossent_loss = cross_entropy_with_probs(probs, one_hot_labels, name=name)
        return crossent_loss


def sigmoid_cross_entropy_with_probs(pred_probs, target_probs, eps=1e-6, target_mask=None, name=None):
    """
    Sigmoid cross entropy loss function
    :param pred_probs: probability tensor produced by sigmoid, size: [batch_size, label_num]
    :param target_probs: gold labels, size: [batch_size, label_num]
    :param eps: epsilon used to avoid overflow in tf.log
    :param target_mask: target mask to do pseudo candidate sampling
    :param name: scope name
    :return: loss vector, size: [batch_size, label_num]
    """
    with tf.name_scope(name, "SigmoidCrossEntropyWithProb", [pred_probs, target_probs]):
        pred_probs = tf.minimum(pred_probs, 1)
        # crossent_loss = -tf.reduce_sum(
        #     target_probs * tf.log(tf.maximum(pred_probs, eps)) + (1-target_probs) * tf.log(tf.maximum(1-pred_probs, eps)),
        #     reduction_indices=1, keep_dims=False
        # )
        crossent_loss = target_probs * (-tf.log(tf.maximum(pred_probs, eps))) + \
                        (1-target_probs) * (-tf.log(tf.maximum(1-pred_probs, eps)))
        return crossent_loss


def entropy(probs, name=None):
    """
    Calculate the entropy for a batch of distributions
    :param probs: one row corresponding to one distribution, size: [batch_size, dist_pt_num]
    :param name: scope name
    :return: entropy vector, size: [batch_size]
    """
    with tf.name_scope(name, "Entropy", [probs]):
        entropy_matrix = - probs * tf.log(probs)
        entropy_vec = tf.reduce_sum(entropy_matrix, reduction_indices=[1], keep_dims=False)
        return entropy_vec



