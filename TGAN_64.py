# -*- coding: utf-8 -*-
from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch

from ops import *
from utils import *


class TGAN_64(object):
    # 初始化各类定义
    def __init__(self, sess, args):
        self.sess = sess
        self.dataset_name = args.dataset
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.test_dir = args.test_dir
        self.n_critic = 2
        self.input_fname_pattern = '*.png'
        self.custom_dataset = True
        self.model_name = "TGAN_" + args.loss_type  # name for checkpoint
        self.path = "/media/media/9EA2104DA2102BF1/creat-TwistedW/f_VAEs/CelebA-HQ/train"
        self.loss_type = args.loss_type

        if self.dataset_name == 'celebA':
            # parameters
            self.input_height = 256
            self.input_width = 256
            self.output_height = 64
            self.output_width = 64

            self.z_dim = args.z_dim  # dimension of noise-vector
            self.c_dim = 3

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load
            self.data = glob(os.path.join(self.path, self.input_fname_pattern))

            # get number of batches for a single epoch
            self.num_batches = 1000

            self.dataset_num = len(self.data)


        else:
            raise NotImplementedError

    def encoder(self, x, is_training=True, reuse=False, sn=True):
        with tf.variable_scope("encoder", reuse=reuse):
            if self.dataset_name == 'celebA':
                net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='en_conv1', sn=sn))
                net = lrelu(
                    bn(conv2d(net, 128, 4, 4, 2, 2, name='en_conv2', sn=sn), is_training=is_training, scope='en_bn2'))
                net = lrelu(
                    bn(conv2d(net, 256, 4, 4, 2, 2, name='en_conv3', sn=sn), is_training=is_training, scope='en_bn3'))
                net = lrelu(
                    bn(conv2d(net, 256, 4, 4, 1, 1, name='en_conv4', sn=sn), is_training=is_training, scope='en_bn4'))
                return net

    def discriminator(self, x, is_training=True, reuse=False, sn=True):
        with tf.variable_scope("discriminator", reuse=reuse):
            if self.dataset_name == 'celebA':
                net = tf.reshape(x, [self.batch_size, -1])
                net = MinibatchLayer(32, 32, net, 'd_fc1')
                net = lrelu(bn(linear(net, 512, scope='d_fc2', sn=sn), is_training=is_training, scope='d_bn2'))
                net = lrelu(bn(linear(net, 64, scope='d_fc3', sn=sn), is_training=is_training, scope='d_bn3'))
                out = linear(net, 1, scope='d_fc4', sn=sn)
                return out

    def generator(self, z, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            if self.dataset_name == 'celebA':
                net = tf.nn.relu(bn(linear(z, 448 * 4 * 4, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
                net = tf.reshape(net, [self.batch_size, 4, 4, 448])
                net = tf.nn.relu(bn(deconv2d(net, [self.batch_size, 8, 8, 256], 4, 4, 2, 2, name='g_dc2'),
                                    is_training=is_training, scope='g_bn2'))
                net = tf.nn.relu(bn(deconv2d(net, [self.batch_size, 8, 8, 256], 4, 4, 1, 1, name='g_dc3'),
                                    is_training=is_training, scope='g_bn3'))
                net = tf.nn.relu(bn(deconv2d(net, [self.batch_size, 16, 16, 128], 4, 4, 2, 2, name='g_dc4'),
                                    is_training=is_training, scope='g_bn4'))
                net = tf.nn.relu(bn(deconv2d(net, [self.batch_size, 32, 32, 64], 4, 4, 2, 2, name='g_dc5'),
                                    is_training=is_training, scope='g_bn5'))
                out = tf.nn.tanh(deconv2d(net, [self.batch_size, 64, 64, 3], 4, 4, 2, 2, name='g_dc6'))
                return out

    def build_model(self):
        # some parameters
        bs = self.batch_size

        """ Graph Input """
        # images
        if self.custom_dataset:
            Image_Data_Class = ImageData(self.output_height, self.c_dim)
            inputs = tf.data.Dataset.from_tensor_slices(self.data)

            gpu_device = '/gpu:0'
            inputs = inputs.apply(shuffle_and_repeat(self.dataset_num)).apply(
                map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=8,
                              drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))

            inputs_iterator = inputs.make_one_shot_iterator()

            self.inputs = inputs_iterator.get_next()

        else:
            self.inputs = tf.placeholder(tf.float32,
                                         [self.batch_size, self.output_height, self.output_height, self.c_dim],
                                         name='real_images')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """
        x_fake = self.generator(self.z, is_training=True, reuse=False)
        x_real_encoder = self.encoder(self.inputs, is_training=True, reuse=False, sn=True)
        x_fake_encoder = self.encoder(x_fake, is_training=True, reuse=True, sn=True)
        x_real_fake = tf.subtract(x_real_encoder, x_fake_encoder)
        x_fake_real = tf.subtract(x_fake_encoder, x_real_encoder)
        x_real_fake_score = self.discriminator(x_real_fake, is_training=True, reuse=False, sn=True)
        x_fake_real_score = self.discriminator(x_fake_real, is_training=True, reuse=True, sn=True)

        # get loss for discriminator
        self.d_loss = discriminator_loss(self.loss_type, real=x_real_fake_score, fake=x_fake_real_score)

        # get loss for generator
        self.g_loss = generator_loss(self.loss_type, real=x_real_fake_score, fake=x_fake_real_score)

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name or 'encoder' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)

        """ Summary """
        self.d_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_sum = tf.summary.scalar("g_loss", self.g_loss)

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = int(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        past_d_loss = -1
        for epoch in range(start_epoch, self.epoch):
            for idx in range(start_batch_id, self.num_batches):
                batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                if self.dataset_name == 'celebA':
                    if self.custom_dataset:
                        train_feed_dict = {
                            self.z: batch_z
                        }
                    else:
                        random_index = np.random.choice(self.dataset_num, size=self.batch_size, replace=False)
                        batch_images = get_pix_image(self.data, random_index, ih=256, iw=256, oh=128, ow=128)
                        train_feed_dict = {
                            self.z: batch_z,
                            self.inputs:batch_images
                        }

                # update D network
                d_loss = None
                if (counter - 1) % self.n_critic == 0:
                    _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                           feed_dict=train_feed_dict)
                    self.writer.add_summary(summary_str, counter)
                    past_d_loss = d_loss

                # update G network
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                       feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                if d_loss is None:
                    d_loss = past_d_loss

                if np.mod(counter, 50) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss= %.8f, g_loss= %.8f" \
                          % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images, feed_dict={self.z: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                                '_train_{:02d}_{:04d}.png'.format(epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                    '_epoch%03d' % epoch + '_test_all_classes.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def train_check(self):
        import re
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir, self.model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            start_epoch = (int)(counter / self.num_batches)
        if start_epoch == self.epoch:
            print(" [*] Training already finished! Begin to test your model")
