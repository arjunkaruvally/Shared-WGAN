'''
An example of distribution approximation using Generative Adversarial Networks
in TensorFlow.

Based on the blog post by Eric Jang:
http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html,

and of course the original GAN paper by Ian Goodfellow et. al.:
https://arxiv.org/abs/1406.2661.

The minibatch discrimination technique is taken from Tim Salimans et. al.:
https://arxiv.org/abs/1606.03498.
'''
from __future__ import division
from ops import *
from tensorflow.contrib import losses
from data_distributions import *

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# import seaborn as s

from glob import glob
from matplotlib import animation

import sys
import os
import json
import scipy

# sns.set(color_codes=True)

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

d0_bn = batch_norm(name="d_bn0")
dt_bn = batch_norm(name="d_bnt")
d1_bn = batch_norm(name="d_bn1")
d2_bn = batch_norm(name="d_bn2")
d3_bn = batch_norm(name="d_bn3")
d4_bn = batch_norm(name="d_bn4")

class GeneratorDistribution(object):
    def sample(self, N):
        return (np.random.uniform(-1, 1, size=[N, 100]))


def generator(input, h_dim, params):
    gl0_bn = batch_norm(name='gl_bn0')
    g0_bn = batch_norm(name='g_bn0')
    g1_bn = batch_norm(name='g_bn1')
    g2_bn = batch_norm(name='g_bn2')
    gt2_bn = batch_norm(name='g_bnt2')
    gt3_bn = batch_norm(name='g_bnt3')
    
    g3_bn = batch_norm(name='g_bn3')

    print input.shape

    h0t = lrelu(gl0_bn(linear(input, params.image_height*params.image_width*params.image_channels, scope='g0t')))

    h0t = tf.reshape(h0t, tf.TensorShape([h0t.shape[0], params.image_height, params.image_width, params.image_channels ]))

    h0 = lrelu(g0_bn(conv2d(h0t, h_dim, d_h=1, d_w=1, name='g0')))
    h1 = lrelu(g1_bn(conv2d(h0, h_dim, d_h=1, d_w=1, name='gshared')))
    h2 = lrelu(g2_bn(conv2d(h1, h_dim, d_h=1, d_w=1, name='g2')))
    ht3 = lrelu(gt3_bn(conv2d(h2, h_dim, d_h=1, d_w=1, name='gt3')))
    ht1 = lrelu(g3_bn(conv2d(ht3, h_dim, d_h=1, d_w=1, name='gt1')))
    h3 = tf.nn.sigmoid(conv2d(ht1, params.image_channels, d_h=1, d_w=1, name='g3'))

    print h3
    
    return h3


def discriminator(input, h_dim, minibatch_layer=True):
    h0 = lrelu(d0_bn(conv2d(input, h_dim, name='d0')))
    h1 = lrelu(d1_bn(conv2d(h0, h_dim, name='dshared')))
    h2 = lrelu(d2_bn(conv2d(h1, h_dim/2, name='d2')))
    h3 = lrelu(d3_bn(conv2d(h2, h_dim/4, name='d3')))

    h4 = tf.reshape(h3, tf.TensorShape([h3.shape[0], h3.shape[1]*h3.shape[2]*h3.shape[3]]))

    h5 = tf.nn.tanh(linear(h4, 1, scope="d4"))

    return h5


def minibatch(input, num_kernels=5, kernel_dim=3):
    x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - \
        tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], 1)


def optimizer(loss, var_list, learning_rate, beta1):
    # learning_rate = 0.001
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(
        loss,
        global_step=step,
        var_list=var_list
    )
    return optimizer


def copy_disc_shared():
    shared_weights_g_w = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.endswith('gshared/w:0')]
    shared_weights_g_b = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.endswith('gshared/b:0')]

    shared_weights_d_w = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.endswith('dshared/w:0')]
    shared_weights_d_b = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.endswith('gshared/w:0')]

    shared_weights_g_w[0].assign(shared_weights_d_w[0])
    shared_weights_g_b[0].assign(shared_weights_d_b[0])


def log(x):
    '''
    Sometimes discriminiator outputs can reach values close to
    (or even slightly less than) zero due to numerical rounding.
    This just makes sure that we exclude those values so that we don't
    end up with NaNs during optimisation.
    '''
    return tf.log(tf.maximum(x, 1e-5))


class GAN(object):
    def __init__(self, params):
        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.

        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(params.batch_size, 100))
            self.G = generator(self.z, params.hidden_size, params)

        # The discriminator tries to tell the difference between samples from
        # the true data distribution (self.x) and the generated samples
        # (self.z).
        #
        # Here we create two copies of the discriminator network
        # that share parameters, as you cannot use the same network with
        # different inputs in TensorFlow.
        self.x = tf.placeholder(tf.float32, shape=(params.batch_size, params.image_height, params.image_width, params.image_channels))

        with tf.variable_scope('D'):
            self.D1 = discriminator(
                self.x,
                params.hidden_size,
                params.minibatch
            )
        with tf.variable_scope('D', reuse=True):
            self.D2 = discriminator(
                self.G,
                params.hidden_size,
                params.minibatch
            )

        # Define the loss for discriminator and generator networks
        # (see the original paper for details), and create optimizers for both

        self.D1 = tf.reshape(self.D1, (-1, 1))
        self.D2 = tf.reshape(self.D2, (-1, 1))

        print "D1: ", self.D1
        print "D2: ", self.D2

        real_loss = losses.sigmoid_cross_entropy(self.D1, tf.ones([params.batch_size, 1]))
        fake_loss = losses.sigmoid_cross_entropy(self.D2, tf.fill([params.batch_size, 1], -1))
        self.loss_d = real_loss + fake_loss
        self.loss_g = losses.sigmoid_cross_entropy(self.D2, tf.ones([params.batch_size, 1]))

        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        ## Clip Ops

        self.clip_discriminator = []

        for var in self.d_params:
            self.clip_discriminator.append(tf.assign(var, tf.clip_by_value(var, -params.clip_value, params.clip_value)))

        if params.shared:
            self.g_params.remove([v for v in vars if v.name.startswith('G/gshared/w')][0])
            self.g_params.remove([v for v in vars if v.name.startswith('G/gshared/b')][0])
            
        print self.loss_d
        print self.loss_g

        self.opt_d = optimizer(self.loss_d, self.d_params, params.d_learning_rate, params.beta1)
        self.opt_g = optimizer(self.loss_g, self.g_params, params.g_learning_rate, params.beta1)

        d_loss_summary = tf.summary.scalar('d_loss', self.loss_d)
        g_loss_summary = tf.summary.scalar('g_loss', self.loss_g)

        g_out_summary = tf.summary.histogram('g_out', self.G)
        d_real_out_summary = tf.summary.histogram('d_real_out', self.D1)
        d_fake_out_summary = tf.summary.histogram('d_fake_out', self.D2)

        self.d_summary = tf.summary.merge([ d_loss_summary ])
        self.g_summary = tf.summary.merge([ g_loss_summary ])

        self.test_summary = tf.summary.merge([ d_real_out_summary ])

        # copy_gen_shared()

        shared_weights_g = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('gshared')]

        shared_weights_g_w = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.endswith('gshared/w:0')]
        shared_weights_g_b = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.endswith('gshared/b:0')]

        shared_weights_d_w = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.endswith('dshared/w:0')]
        shared_weights_d_b = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.endswith('dshared/b:0')]

        print shared_weights_g_w[0]
        print shared_weights_d_w[0]
        # sys.exit()

        self.copy_d_w_g = shared_weights_g_w[0].assign(shared_weights_d_w[0])
        self.copy_d_b_g = shared_weights_g_b[0].assign(shared_weights_d_b[0])


def train(model, data, gen, params, index=0):
    anim_frames = [] 

    with tf.Session() as session:
        train_writer = tf.summary.FileWriter('log/'+str(index), session.graph)

        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        saver = tf.train.Saver()

        step_start = 0

        if params.restore_session:
            print "Checking for saved sessions"
            if os.path.isfile('checkpoints/checkpoint'):
                saver.restore(session, "checkpoints/model.ckpt")
                    
                with open('checkpoints/global_step.json', 'r') as fp:
                    step_start = json.load(fp)
                    step_start = step_start["step"]

                print "Session resored"
            else:
                print "No Session Found"

        for step in range(step_start, params.num_steps):
            # update discriminator

            if step < 25 or step%500 == 0:
                # discriminator_train = 100
                discriminator_train = params.discriminator_train
            else:
                discriminator_train = params.discriminator_train

            for diter in range(discriminator_train):
                session.run(model.clip_discriminator)
                x = data.sample(params.batch_size)
                z = gen.sample(params.batch_size)
                summary, loss_d, _, = session.run([model.d_summary, model.loss_d, model.opt_d], {
                    model.x: x,
                    model.z: z
                })

                train_writer.add_summary(summary, step)

            if params.shared:
                session.run([model.copy_d_w_g, model.copy_d_b_g])

            # update generator
            z1 = gen.sample(params.batch_size)
            
            summary, loss_g, _ = session.run([model.g_summary, model.loss_g, model.opt_g], {
                model.z: z1
            })

            train_writer.add_summary(summary, step)
            
            if step % params.log_every == 0:
                print('Epoch:{} step:{}: d:{}\tg:{}'.format(int(step*params.batch_size/data.N), step, loss_d, loss_g))

                if params.show_output:
                    plot_gen_out(session, model, gen, params.batch_size, str(step), z1, params)

            if step % 1000 == 0:
                save_path = saver.save(session, "checkpoints/model.ckpt")
                print("Model saved in file: %s" % save_path)
                with open('checkpoints/global_step.json', 'w') as fp:
                    json.dump({ "step": step }, fp)


def test(model, gen, params):
    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        vars = tf.trainable_variables()
        analyze_g = [v for v in vars if v.name.startswith('G/g2/w')]

        saver = tf.train.Saver()

        step_start = 0

        print "Checking for saved sessions"
        if os.path.isfile('checkpoints/checkpoint'):
            saver.restore(session, "checkpoints/model.ckpt")    
            with open('checkpoints/global_step.json', 'r') as fp:
                step_start = json.load(fp)
                step_start = step_start["step"]

                print "Session resored"
        else:
            print "No Session Found - exiting"
            return

        for i in range(params.test_count):
            print i+1,"/",params.test_count
            plot_gen_out(session, model, gen, params.batch_size, "test/test_"+str(i))



def plot_gen_out(sess, model, gen_distribution, N, filename, ginput=None, params=None):
    if ginput == None:
        ginput = gen_distribution.sample(N)

    gen_out = sess.run(model.G, { model.z: ginput })

    for x in range(25):
        plt.subplot(5, 5, x+1)

        if params.image_channels == 1:
            img = gen_out[x].reshape([gen_out[x].shape[0], gen_out[x].shape[1]])
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(gen_out[x])

    plt.tight_layout()
    plt.savefig("plots/"+filename+".png", dpi=200)
    # plt.clf()


def main(args):

    if args.dataset == 'mnist':
        data = MNIST()
    elif args.dataset == 'imagenet':
        data = ImageNet()
    elif args.dataset == 'celebA':
        data = CelebA()
    else:
        print "Dataset does not exist"
        sys.exit()

    d = vars(args)

    d['image_height'] = data.image_height
    d['image_width'] = data.image_width
    d['image_channels'] = data.image_channels

    model = GAN(args)

    if args.mode=="train":
        train(model, data, GeneratorDistribution(), args)

    elif args.mode == "test":
        test(model, GeneratorDistribution(), args)

    else:
        print "Mode not found"