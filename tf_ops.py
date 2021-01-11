"""
Functions for building tensorflow computational graph models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def fan_scale(initrange, activation, tensor_in):
    """
    Creates a scaling factor for weight initialization according to best practices.

    :param initrange: Scaling in addition to fan_in scale.
    :param activation: A tensorflow non-linear activation function
    :param tensor_in: Input tensor to layer of network to scale weights for.
    :return: (float) scaling factor for weight initialization.
    """
    if activation == tf.nn.relu:
        initrange *= np.sqrt(2.0 / float(tensor_in.get_shape().as_list()[1]))
    else:
        initrange *= (1.0 / np.sqrt(float(tensor_in.get_shape().as_list()[1])))
    return initrange


def ident(tensor_in, name='ident'):
    """
    The identity function

    :param tensor_in: Input to operation.
    :return: tensor_in
    """
    return tensor_in


def weights(distribution, shape, dtype=tf.float32, initrange=1e-5,
            seed=None, l2=0.0, name='weights'):
    """
    Wrapper parameterizing common constructions of tf.Variables.

    :param distribution: A string identifying distribution 'tnorm' for truncated normal, 'rnorm' for random normal, 'constant' for constant, 'uniform' for uniform.
    :param shape: Shape of weight tensor.
    :param dtype: dtype for weights
    :param initrange: Scales standard normal and trunctated normal, value of constant dist., and range of uniform dist. [-initrange, initrange].
    :param seed: For reproducible results.
    :param l2: Floating point number determining degree of of l2 regularization for these weights in gradient descent update.
    :param name: For variable scope.
    :return: A tf.Variable.
    """
    with tf.variable_scope(name):
        if distribution == 'norm':
            wghts = tf.Variable(initrange * tf.random_normal(shape, 0, 1, dtype, seed))
        elif distribution == 'tnorm':
            wghts = tf.Variable(initrange * tf.truncated_normal(shape, 0, 1, dtype, seed))
        elif distribution == 'uniform':
            wghts = tf.Variable(tf.random_uniform(shape, -initrange, initrange, dtype, seed))
        elif distribution == 'constant':
            wghts = tf.Variable(tf.constant(initrange, dtype=dtype, shape=shape))
        else:
            raise ValueError("Argument 'distribution takes values 'norm', 'tnorm', 'uniform', 'constant', "
                             "Received %s" % distribution)
        if l2 != 0.0:
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(wghts), l2, name=name + 'weight_loss'))
        return wghts


def batch_normalize(tensor_in, epsilon=1e-5, decay=0.999):
    """
    Batch Normalization:
    `Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift`_

    An exponential moving average of means and variances in calculated to estimate sample mean
    and sample variance for evaluations. For testing pair placeholder is_training
    with [0] in feed_dict. For training pair placeholder is_training
    with [1] in feed_dict. Example:

    Let **train = 1** for training and **train = 0** for evaluation

    .. code-block:: python

        bn_deciders = {decider:[train] for decider in tf.get_collection('bn_deciders')}
        feed_dict.update(bn_deciders)

    During training the running statistics are updated, and batch statistics are used for normalization.
    During testing the running statistics are not updated, and running statistics are used for normalization.

    :param tensor_in: (tf.Tensor) Input Tensor.
    :param epsilon: (float) A float number to avoid being divided by 0.
    :param decay: (float) For exponential decay estimate of running mean and variance.
    :return: (tf.Tensor) Tensor with variance bounded by a unit and mean of zero according to the batch.
    """

    is_training = tf.placeholder(tf.int32, shape=[None])  # [1] or [0], Using a placeholder to decide which
    # statistics to use for normalization allows
    # either the running stats or the batch stats to
    # be used without rebuilding the graph.
    tf.add_to_collection('bn_deciders', is_training)

    pop_mean = tf.Variable(tf.zeros([tensor_in.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([tensor_in.get_shape()[-1]]), trainable=False)

    # calculate batch mean/var and running mean/var
    batch_mean, batch_variance = tf.nn.moments(tensor_in, [0])

    # The running mean/variance is updated when is_training == 1.
    running_mean = tf.assign(pop_mean,
                             pop_mean * (decay + (1.0 - decay) * (1.0 - tf.to_float(is_training))) +
                             batch_mean * (1.0 - decay) * tf.to_float(is_training))
    running_var = tf.assign(pop_var,
                            pop_var * (decay + (1.0 - decay) * (1.0 - tf.to_float(is_training))) +
                            batch_variance * (1.0 - decay) * tf.to_float(is_training))

    # Choose statistic
    mean = tf.nn.embedding_lookup(tf.stack([running_mean, batch_mean]), is_training)
    variance = tf.nn.embedding_lookup(tf.stack([running_var, batch_variance]), is_training)

    shape = tensor_in.get_shape().as_list()
    gamma = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[shape[1]], name='gamma'))
    beta = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[shape[1]], name='beta'))

    # Batch Norm Transform
    inv = tf.rsqrt(epsilon + variance)
    tensor_in = beta * (tensor_in - mean) * inv + gamma

    return tensor_in


def dropout(tensor_in, prob):
    """
    Adds dropout node.
    `Dropout A Simple Way to Prevent Neural Networks from Overfitting`_

    :param tensor_in: Input tensor.
    :param prob: The percent of units to keep.
    :return: Tensor of the same shape of *tensor_in*.
    """

    keep_prob = tf.placeholder(tf.float32)
    tf.add_to_collection('dropout_prob_placeholder', keep_prob)
    tf.add_to_collection('dropout_prob_value', prob)
    return tf.nn.dropout(tensor_in, keep_prob)


def layer_norm(h):
    """

    :param h: (tensor) Hidden layer of neural network
    :return: (tensor) Hidden layer after layer_norm transform
    """
    dim = h.get_shape().as_list()
    bias = tf.Variable(tf.zeros([1, dim[1]], dtype=tf.float32))
    gain = tf.Variable(tf.ones([1, dim[1]], dtype=tf.float32))
    mu, variance = tf.nn.moments(h, [1], keep_dims=True)
    return (gain / tf.sqrt(variance)) * (h - mu) + bias


def dnn(x, layers=[100, 408], act=tf.nn.relu, scale_range=1.0, norm=None, keep_prob=None, name='nnet'):
    """
    An arbitrarily deep neural network. Output has non-linear activation.

    :param x: Input to the network.
    :param layers: List of sizes of network layers.
    :param act: Activation function to produce hidden layers of neural network.
    :param scale_range: Scaling factor for initial range of weights (Set to 1/sqrt(fan_in) for tanh, sqrt(2/fan_in) for relu.
    :param norm: Normalization function. Could be layer_norm or other function that retains shape of tensor.
    :param keep_prob: The percent of nodes to keep in dropout layers.
    :param name: For naming and variable scope.
    :return: (tf.Tensor) Output of neural net. This will be just following a non linear transform, so that final activation has not been applied.
    """
    if type(scale_range) is not list:
        scale_range = [scale_range] * len(layers)
    assert len(layers) == len(scale_range)

    for ind, hidden_size in enumerate(layers):
        with tf.variable_scope('layer_%s' % ind):

            fan_in = x.get_shape().as_list()[1]
            W = tf.Variable(fan_scale(scale_range[ind], act, x) * tf.truncated_normal([fan_in, hidden_size],
                                                                                      mean=0.0, stddev=1.0,
                                                                                      dtype=tf.float32, seed=None,
                                                                                      name='W'))
            tf.add_to_collection(name + '_weights', W)
            b = tf.Variable(tf.zeros([hidden_size])) + 0.1 * (float(act == tf.nn.relu))
            tf.add_to_collection(name + '_bias', b)
            x = tf.matmul(x, W) + b
            if norm is not None:
                x = norm(x)
            x = act(x, name='h' + str(ind))  # The hidden layer
            tf.add_to_collection(name + '_activation', x)
            if keep_prob is not None:
                x = dropout(x, keep_prob)
    return x


