import tensorflow as tf
import os

def conv2d(input_, output_dim, name, k_h=5, k_w=5, s_h=2, s_w=2, stddev=0.02, padding='SAME', activation='relu',
           batch_norm=True, training=True, constraint=None):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        kernel = tf.get_variable('kernel', [k_h, k_w, input_dim, output_dim],
                                 initializer=tf.initializers.truncated_normal(stddev=stddev), constraint=constraint)
        conv = tf.nn.conv2d(input_, kernel, [1, s_h, s_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.initializers.constant(0.0))
        conv = tf.nn.bias_add(conv, biases)
        if batch_norm:
            conv = tf.layers.batch_normalization(conv, training=training)  # training is False when test
        if activation == 'relu':
            conv = tf.nn.relu(conv)
        elif activation == 'tanh':
            conv = tf.nn.tanh(conv)
        elif activation == 'sigmoid':
            conv = tf.nn.sigmoid(conv)
        elif activation == 'lrelu':
            conv = tf.nn.leaky_relu(conv)

        return conv


def de_conv(input_, output_shape, name, k_h=5, k_w=5, s_h=2, s_w=2, stddev=0.02, padding='SAME', activation='relu',
            batch_norm=True, training=True, constraint=None):

    # filter : [height, width, output_channels, in_channels]
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        kernel = tf.get_variable('kernel', [k_h, k_w, output_shape[-1], input_dim],
                                 initializer=tf.initializers.random_normal(stddev=stddev), constraint=constraint)
        deconv = tf.nn.conv2d_transpose(input_, kernel, output_shape, [1, s_h, s_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.initializers.constant(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
        deconv = tf.reshape(deconv, deconv.get_shape())

        if batch_norm:
            deconv = tf.layers.batch_normalization(deconv, training=training)
        if activation == 'relu':
            deconv = tf.nn.relu(deconv)
        elif activation == 'tanh':
            deconv = tf.nn.tanh(deconv)
        elif activation == 'sigmoid':
            deconv = tf.nn.sigmoid(deconv)

        return deconv


def fully_connected(input_, output_dim, name, bias_init=0.0, activation='relu', batch_norm=True,
                    training=True, keep_prob=1.0, constraint=None):
    input_dim = input_.get_shape().as_list()[1]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('weights', [input_dim, output_dim], tf.float32,
                                  tf.initializers.random_normal(stddev=1./(tf.sqrt(input_dim/2.))), constraint=constraint)
        bias = tf.get_variable('bias', [output_dim],
                               initializer=tf.initializers.constant(bias_init))

        fc = tf.matmul(input_, weights) + bias

        if batch_norm:
            fc = tf.layers.batch_normalization(fc, training=training)  # training is False when test

        if activation == 'relu':
            fc = tf.nn.relu(fc)
        elif activation == 'tanh':
            fc = tf.nn.tanh(fc)
        elif activation == 'sigmoid':
            fc = tf.nn.sigmoid(fc)
        elif activation == 'softmax':
            fc = tf.nn.softmax(fc)

        fc = tf.nn.dropout(fc, keep_prob=keep_prob)

        return fc

def check_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)