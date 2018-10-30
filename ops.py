#-*- coding: utf-8 -*-
"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops
import tensorflow.contrib as tf_contrib

from utils import *

weight_init = tf_contrib.layers.xavier_initializer()
weight_regularizer = None

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)

def bn(x, is_training, scope):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, name="conv2d", use_bias=False, sn=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=weight_init)
        if sn:
            conv = tf.nn.conv2d(input_, filter=spectral_norm(w), strides=[1, d_h, d_w, 1], padding='SAME')
        else:
            conv = tf.nn.conv2d(input_, filter=w, strides=[1, d_h, d_w, 1], padding='SAME')
        if use_bias:
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="deconv2d", with_w=False, use_bias=False, sn=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=weight_init)

        if sn:
            deconv = tf.nn.conv2d_transpose(input_, filter=spectral_norm(w), output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1], padding='SAME')
        else:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1], padding='SAME')

        if use_bias:
            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        else:
            biases = 0

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def linear(input_, output_size, scope=None, with_w=False, sn=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 initializer=weight_init, regularizer=weight_regularizer)
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(0.0))
        if with_w:
            if sn:
                return tf.matmul(input_, spectral_norm(matrix)) + bias, matrix, bias
            else:
                return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            if sn:
                return tf.matmul(input_, spectral_norm(matrix)) + bias
            else:
                return tf.matmul(input_, matrix) + bias

def MinibatchLayer(dim_b, dim_c, inputs, name):
    # input: batch_size, n_in
    # M: batch_size, dim_b, dim_c
    m = linear(inputs, dim_b * dim_c, scope=name, sn=True)
    m = tf.reshape(m, [-1, dim_b, dim_c])
    # c: batch_size, batch_size, dim_b
    c = tf.abs(tf.expand_dims(m, 0) - tf.expand_dims(m, 1))
    c = tf.reduce_sum(c, reduction_indices=[3])
    c = tf.exp(-c)
    # o: batch_size, dim_b
    o = tf.reduce_mean(c, reduction_indices=[1])
    o -= 1  # to account for the zero L1 distance of each example with itself
    # result: batch_size, n_in+dim_b
    return tf.concat([o, inputs], axis=1)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

##################################################################################
# Loss function
##################################################################################

def discriminator_loss(loss_func, real, fake):
    real_loss = 0
    fake_loss = 0

    if loss_func == 'wgan':
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)

    if loss_func == 'lsgan':
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))

    if loss_func == 'sgan' or loss_func == 'dragan':
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real)+1e-9)
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake)+1e-9)

    if loss_func == 'hinge':
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake))

    loss = real_loss + fake_loss

    return loss


def generator_loss(loss_func, real, fake):
    real_loss = 0
    fake_loss = 0

    if loss_func == 'wgan':
        real_loss = tf.reduce_mean(real)
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan':
        real_loss = tf.reduce_mean(tf.square(real))
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if loss_func == 'sgan' or loss_func == 'dragan':
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real), logits=real)+1e-9)
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake)+1e-9)

    if loss_func == 'hinge':
        fake_loss = -tf.reduce_mean(fake)

    loss = real_loss + fake_loss

    return loss

def classifier_loss(predition, label):
    return tf.reduce_mean(-tf.reduce_sum(label * tf.log(predition), reduction_indices=[1]))

def LL_loss(fake, real):
    c = -0.5 * tf.log(2 * np.pi)
    multiplier = 1.0 / (2.0 * 1)
    tmp = tf.square(fake - real)
    tmp *= -multiplier
    tmp += c
    return tf.reduce_mean(tf.reduce_sum(tmp, [1,2,3]))
