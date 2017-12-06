import tensorflow as tf

from mnist_classifier import *
from scipy.stats import entropy

import experimental_wgan
import experimental_wgan_shared

import matplotlib.pyplot as plt
import pickle as pkl

import argparse
import os
import sys

parser = argparse.ArgumentParser()

parser.add_argument('--hidden-size', type=int, default=64,
                    help='MLP hidden size')
parser.add_argument('--batch-size', type=int, default=64,
                    help='the batch size')
parser.add_argument('--d_learning_rate', type=float, default=5e-5,
                    help='Change learning rate of generator')
parser.add_argument('--g_learning_rate', type=float, default=5e-5,
                    help='Change learning rate of discriminator')
parser.add_argument('--mode', type=str, default="test",
                    help='mode of the system(test/train)')

## Unecessary except for creating graph
parser.add_argument('--minibatch', action='store_false',
                    help='use minibatch discrimination')
parser.add_argument('--clip_value', type=float, default=0.01,
                        help='Clip value of discriminator')
parser.add_argument('--beta1', type=float, default=0.5,
                        help='Beta1 momentum adam')

args = parser.parse_args()

baseline_graph = tf.Graph()

with baseline_graph.as_default():
	model = experimental_wgan_shared.GAN(args)
	model_file = "completed/checkpoints_baseline/model.ckpt"
	saver = tf.train.Saver()

	tbg_shared = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.endswith('gshared/w:0')]
	tbd_shared = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.endswith('dshared/w:0')]

	with tf.Session() as sess:
		saver.restore(sess, model_file)

		d_weights, g_weights = sess.run([ tbd_shared[0], tbg_shared[0] ])

		with open('completed/weights/baseline_discriminator.pkl', 'w') as f:
			np.save(f, d_weights)

		with open('completed/weights/baseline_generator.pkl', 'w') as f:
			np.save(f, g_weights)

		print d_weights[:, :, 0, 0]
		print g_weights[:, :, 0, 0]

		print "Baseline Saved"

shared_graph = tf.Graph()

with shared_graph.as_default():
	model = experimental_wgan_shared.GAN(args)
	model_file = "completed/checkpoints_shared/model.ckpt"
	saver = tf.train.Saver()

	tbg_shared = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.endswith('gshared/w:0')]
	tbd_shared = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.endswith('dshared/w:0')]

	with tf.Session() as sess:
		saver.restore(sess, model_file)

		d_weights, g_weights = sess.run([ tbd_shared[0], tbg_shared[0] ])

		with open('completed/weights/shared_discriminator.pkl', 'w') as f:
			np.save(f, d_weights)

		with open('completed/weights/shared_generator.pkl', 'w') as f:
			np.save(f, g_weights)

		print d_weights[:, :, 0, 0]
		print g_weights[:, :, 0, 0]

		print "Shared Saved"