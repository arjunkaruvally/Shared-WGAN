import tensorflow as tf

from mnist_classifier import *
from scipy.stats import entropy

import experimental_wgan
import experimental_wgan_shared

import matplotlib.pyplot as plt

import argparse
import os
import sys

def create_test_framework():
  x = tf.placeholder(tf.float32, [None, 784], name="input")
  
  conv_out, keep_prob, _ = deepnn(x)

  output_distribution = tf.nn.softmax(conv_out)

  return x, output_distribution, keep_prob


def normalize_scores(scores):
  # scores = np.exp(scores)

  # scores = scores/np.sum(scores, axis=1).reshape([scores.shape[0], 1])

  return scores


def plot_distribution(x):
  plt.hist(x, bins=100)
  plt.show()


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
parser.add_argument('--sample_count', type=int, default=10,
                    help='sample size used for estimation')

## Unecessary except for creating graph
parser.add_argument('--minibatch', action='store_false',
                    help='use minibatch discrimination')
parser.add_argument('--clip_value', type=float, default=0.01,
                        help='Clip value of discriminator')
parser.add_argument('--beta1', type=float, default=0.5,
                        help='Beta1 momentum adam')

args = parser.parse_args()

## Create the test framework

test_graph = tf.Graph()
with test_graph.as_default():
  x, output_distribution, keep_prob = create_test_framework()
  saver = tf.train.Saver()

test_session = tf.Session(graph=test_graph)
entropy_list = []

if args.mode == 'dataset':
  model = experimental_wgan.DataDistribution()

  with test_session.as_default():
    # with test_graph.as_default():
    # init = tf.global_variables_initializer()
    saver.restore(test_session, "checkpoints/classifier.ckpt")

    for i in range(args.sample_count):
      batch = model.sample(args.batch_size)

      batch = batch.reshape([batch.shape[0], 784])

      scores = test_session.run(output_distribution, { x: batch, keep_prob: 1 })

      # print scores.shape

      # print scores[0]

      # print np.sum(scores[0])

      scores = normalize_scores(scores)

      for score in scores:
        escore = entropy(score)
        entropy_list.append(escore)

  test_session.close()

  # print entropy_list
  print "Mean Entropy: ",np.mean(entropy_list)
  print "Standard Deviation: ",np.std(entropy_list)

  plot_distribution(entropy_list)

  sys.exit()


if args.mode == 'perfect':
  model = experimental_wgan_shared.DataDistribution()

  saver.restore(test_session, "checkpoints/classifier.ckpt")

  for i in range(args.sample_count):
    batch, correct_pred = model.sample(args.batch_size, out_y=True)

    # batch = batch.reshape([batch.shape[0], 784])

    scores = np.zeros([batch.shape[0], 10])
    print correct_pred.shape
    
    for i in range(scores.shape[0]):
      scores[i][int(correct_pred[i])] = 1

    # print scores.shape

    # print scores[0]

    # print np.sum(scores[0])

    scores = normalize_scores(scores)

    for score in scores:
      escore = entropy(score)
      entropy_list.append(escore)

  # print entropy_list
  print "Mean Entropy: ",np.mean(entropy_list)
  print "Standard Deviation: ",np.std(entropy_list)

  sys.exit()



if args.mode == 'uniform_outputs':
  model = experimental_wgan_shared.DataDistribution()

  for i in range(args.sample_count):
    batch, correct_pred = model.sample(args.batch_size, out_y=True)

    # batch = batch.reshape([batch.shape[0], 784])

    # scores = np.random.uniform(0, 1, [batch.shape[0], 10])
    scores = 0.1*np.ones([batch.shape[0], 10])
    
    # print scores.shape

    # print scores[0]

    # print np.sum(scores[0])

    scores = normalize_scores(scores)

    for score in scores:
      escore = entropy(score)
      entropy_list.append(escore)

  # print entropy_list
  print "Mean Entropy: ",np.mean(entropy_list)
  print "Standard Deviation: ",np.std(entropy_list)

  sys.exit()


elif args.mode == 'baseline':
  
  model_graph = tf.Graph()  
  with model_graph.as_default():
    model = experimental_wgan.GAN(args)
    model_file = "completed/checkpoints_baseline/model.ckpt"
    model_saver = tf.train.Saver()

elif args.mode == 'shared':

  model_graph = tf.Graph()
  with model_graph.as_default():
    model = experimental_wgan_shared.GAN(args)
    model_file = "completed/checkpoints_shared/model.ckpt"
    model_saver = tf.train.Saver()

else:
  model_graph = tf.Graph()
  with model_graph.as_default():
    model = experimental_wgan_shared.GAN(args)

model_session = tf.Session(graph=model_graph)

input_distribution = experimental_wgan.GeneratorDistribution()

with model_session.as_default():
  if args.mode == 'random':
    with model_graph.as_default():
      tf.global_variables_initializer().run()
  else:
    saver.restore(model_session, model_file)


with test_session.as_default():
  saver.restore(test_session, "checkpoints/classifier.ckpt")

for i in range(args.sample_count):
  with model_session.as_default():
    gen_input = input_distribution.sample(args.batch_size)

    gen_image = model_session.run(model.G, { model.z: gen_input })

  with test_session.as_default():
    gen_image = gen_image.reshape([gen_image.shape[0], 784])
    scores = test_session.run(output_distribution, { x: gen_image, keep_prob: 1 })
    scores = normalize_scores(scores)      

    i1 = 0 
    for score in scores:
      # print score
      escore = entropy(score)
      entropy_list.append(escore)

      # plt.imshow(gen_image[i1].reshape([28, 28]), cmap='gray')
      # plt.show()
      i1+=1

  print i

# print entropy_list
print "Mean Entropy: ",np.mean(entropy_list)
print "Standard Deviation: ",np.std(entropy_list)
print "Minimum: ", np.min(entropy_list)
print "Maximum: ", np.max(entropy_list)

model_session.close()
test_session.close()

plot_distribution(entropy_list)