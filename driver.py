import tensorflow as tf
import numpy as np

import os
import json

seed = 100
np.random.seed(seed)
tf.set_random_seed(seed)

class ModelDistribution(object):
  def __init__(self, mean=4, stdev=0.5):
    self.mean = mean
    self.stdev = stdev


  def sample(self, N):
    samples = np.random.normal(self.mean, self.stdev, N)
    # samples.sort()
    return samples




class GeneratorNoise(object):
  def __init__(self, n):
    self.range = n


  def sample(self, N):
    return np.linspace(-self.range, self.range, N) + \
        np.random.random(N) * 0.01


def optimizer(loss, learning_rate, var_list):
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=step,
        var_list=var_list
    )
    return optimizer


class GAN_Network():
  def __init__(self, params):
    self.input_size = params['input_size']
    with tf.name_scope('generator'):
      self.g_in = tf.placeholder(tf.float32, shape=[ params['batch_size'], params['input_size'] ])
      self.g_out = self.generator(self.g_in, params['input_size'], params['gen_hidden_units'], params['output_size'])

    with tf.name_scope('discriminator'):
      self.d_fake_out = self.discriminator(self.g_out, params['input_size'], params['disc_hidden_units1'], params['disc_hidden_units2'], params['output_size'])

    with tf.name_scope('discriminator'):
      self.d_in_real = tf.placeholder(tf.float32, shape=[ params['batch_size'], params['output_size'] ])
      self.d_real_out = self.discriminator(self.d_in_real, params['input_size'], params['disc_hidden_units1'], params['disc_hidden_units2'], params['output_size'])

    vars = tf.trainable_variables()
    self.d_params = [v for v in vars if v.name.startswith('discriminator/')]
    self.g_params = [v for v in vars if v.name.startswith('generator/')]

    self.d_loss = tf.reduce_mean(-tf.log(self.d_real_out)-tf.log(1-self.d_fake_out))
    self.g_loss = tf.reduce_mean(-tf.log(self.d_fake_out))

    d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
    d_real_out_summary = tf.summary.histogram('d_real_out', self.d_real_out)
    d_fake_out_summary = tf.summary.histogram('d_fake_out', self.d_fake_out)
    g_in_summary = tf.summary.histogram('model_distribution', self.d_in_real)
    seed_summary = tf.summary.histogram('generator_seed', self.g_in)
    g_out_summary = tf.summary.histogram('g_out', self.g_out)

    g_out_test = tf.summary.scalar('g_out_test', self.g_out)
    d_out_test = tf.summary.scalar('d_out_test', self.d_real_out)

    g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)

    self.opt_g = optimizer(self.g_loss, params['g_learning_rate'], self.g_params)
    # self.opt_d = optimizer(self.d_loss, params['d_learning_rate'], self.d_params)

    step = tf.Variable(0, trainable=False)
    self.opt_d = tf.train.GradientDescentOptimizer(params['d_learning_rate']).minimize(
        self.d_loss,
        global_step=step,
        var_list=self.d_params
    )

    self.d_summary = tf.summary.merge([d_loss_summary, d_real_out_summary, g_in_summary])
    self.g_summary = tf.summary.merge([g_loss_summary, d_fake_out_summary, g_out_summary, seed_summary])
    
    self.test_summary = tf.summary.merge([g_out_test, d_out_test])


  def generator(self, X, input_size, hidden_units, output_size, alpha=0.2):
    alpha = tf.constant(alpha, tf.float32)
    with tf.name_scope('hidden_layer'):    
      weight = tf.Variable(tf.truncated_normal([input_size, hidden_units], stddev=1.0 ), name='weight')
      bias = tf.Variable(tf.zeros([hidden_units]), name='bias')
      # hidden_layer1 = tf.nn.relu(tf.matmul(X, weight) + bias)
      features = tf.nn.relu(tf.matmul(X, weight) + bias)
      hidden_layer1 = tf.maximum(features, alpha*features)

    with tf.name_scope('output_layer'):
      weight = tf.Variable(tf.truncated_normal([hidden_units, output_size], stddev=1.0 ), name='weight')
      bias = tf.Variable(tf.zeros([output_size]), name='bias')
      output = tf.matmul(hidden_layer1, weight) + bias

    return output


  def discriminator(self, X, input_size, hidden_units1, hidden_units2, output_size, alpha=0.2):
    alpha = tf.constant(alpha, tf.float32)
    with tf.name_scope('hidden_layer1'):
      weight = tf.Variable(tf.truncated_normal([input_size, hidden_units1], stddev=1.0 ), name='weight')
      bias = tf.Variable(tf.zeros([hidden_units1]), name='bias')
      features = tf.matmul(X, weight) + bias
      hidden_layer1 = tf.maximum(features, alpha*features)

    with tf.name_scope('hidden_layer2'):
      weight = tf.Variable(tf.truncated_normal([hidden_units1, hidden_units2], stddev=1.0 ), name='weight')
      bias = tf.Variable(tf.zeros([hidden_units2]), name='bias')
      features = tf.matmul(hidden_layer1, weight) + bias
      hidden_layer2 = tf.maximum(features, alpha*features)

      # hidden_layer2 = tf.nn.leaky_relu(tf.matmul(hidden_layer1, weight) + bias)

    with tf.name_scope('output_layer'):
      weight = tf.Variable(tf.truncated_normal([hidden_units2, 1], stddev=1.0 ), name='weight')
      bias = tf.Variable(tf.zeros([1]), name='bias')
      output = tf.matmul(hidden_layer2, weight) + bias

      activation = tf.sigmoid(output)

    return activation


  def train(self, sample, seed, params):

    index = 0
    while True:
      if not os.path.exists('log/'+str(index)):
        os.makedirs('log/'+str(index))
        break
      index+=1

    saver = tf.train.Saver()

    with tf.Session() as sess:
      train_writer = tf.summary.FileWriter('log/'+str(index), sess.graph)  

      tf.local_variables_initializer().run()
      tf.global_variables_initializer().run()

      disc_switch = False

      loss_g = 0
      loss_d = 0

      for step in range(params['number_of_steps']):
        real_x = sample.sample(params['batch_size'])
        gen_seed = seed.sample(params['batch_size'])

        #Update discriminator
        # loss_d, _, = sess.run([ model.d_loss, model.opt_d ],{ model.d_in_real: np.reshape(real_x, (params['batch_size'], 1)), model.g_in: np.reshape(gen_seed, (params['batch_size'], 1)) })
        
        if step%params['discriminator_skip'] != 0:
          summary, loss_d, _, = sess.run([ model.d_summary, model.d_loss, model.opt_d ],{ model.d_in_real: np.reshape(real_x, (params['batch_size'], 1)), model.g_in: np.reshape(gen_seed, (params['batch_size'], 1)) })
          train_writer.add_summary(summary, step)
        else:
          summary, loss_d = sess.run([ model.d_summary, model.d_loss ],{ model.d_in_real: np.reshape(real_x, (params['batch_size'], 1)), model.g_in: np.reshape(gen_seed, (params['batch_size'], 1)) })
          train_writer.add_summary(summary, step)
          gen_seed = seed.sample(params['batch_size'])
          summary, loss_g, _ = sess.run([ model.g_summary, model.g_loss, model.opt_g ],{ model.g_in: np.reshape(gen_seed, (params['batch_size'], 1)) })

        #Update generator
        # gen_seed = seed.sample(params['batch_size'])
        # summary, loss_g, _ = sess.run([ model.g_summary, model.g_loss, model.opt_g ],{ model.g_in: np.reshape(gen_seed, (params['batch_size'], 1)) })
        # loss_g, _ = sess.run([ model.g_loss, model.opt_g ],{ model.g_in: np.reshape(gen_seed, (params['batch_size'], 1)) })

        train_writer.add_summary(summary, step)

        if step % params['log_every'] == 0:
          print('{}: d {:.4f}\tg {:.4f}, '.format(step, loss_d, loss_g))

      with open('log/'+str(index)+"/params.json", 'w') as fp:
        json.dump(params, fp)

      save_path = saver.save(sess, "models/model.ckpt")
      print("Model saved in file: %s" % save_path)

      gen_out, d_out = sess.run([ model.g_out, model.d_real_out ], { model.g_in: np.reshape(seed.sample(['test_batch'])), model.d_in_real: np.reshape(sample.sample(['test_batch'])) })
      


# params = {
#   'g_learning_rate': 0.006,
#   'd_learning_rate': 0.003,
#   'input_size': 1,
#   'output_size': 1,
#   'batch_size': 100,
#   'gen_hidden_units': 4,
#   'disc_hidden_units1': 2,
#   'disc_hidden_units2': 1,
#   'log_every': 10,
#   'discriminator_skip': 5,
#   'number_of_steps': 10000
# }
params = {
  'g_learning_rate': 0.001,
  'd_learning_rate': 0.001,
  'input_size': 1,
  'output_size': 1,
  'batch_size': 100,
  'test_batch': 10000,
  'gen_hidden_units': 4,
  'disc_hidden_units1': 2,
  'disc_hidden_units2': 2,
  'log_every': 10,
  'discriminator_skip': 5,
  'number_of_steps': 5000
}

model = GAN_Network(params)

model.train(ModelDistribution(mean=1, stdev=0.5), ModelDistribution(mean= 0, stdev=1), params)
# model.train(ModelDistribution(mean=1, stdev=0.5), ModelDistribution(mean=1, stdev=0.5), params)