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

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from matplotlib import animation
# import seaborn as sns

# sns.set(color_codes=True)

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01


def linear(input, output_dim, scope=None, stddev=1.0, reuse=None):
    with tf.variable_scope(scope or 'linear', reuse=reuse):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input, w) + b


def common_layer(output_dim, input=None, scope=None, stddev=1.0, reuse=None):
    # input = tf.placeholder(tf.float32, [None, output_dim])
    with tf.variable_scope('shared_knowledge', reuse=reuse):
        w = tf.get_variable(
            'w',
            [output_dim, output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input, w) + b


def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = tf.nn.softplus(linear(h0, h_dim, 'gt'))
    h2 = linear(h1, 1, 'g1')
    return h2


def discriminator(input, h_dim, minibatch_layer=True):
    h0 = tf.nn.relu(linear(input, h_dim, 'd0'))
    h1 = tf.nn.relu(linear(h0, h_dim, 'dt'))
    h2 = tf.nn.relu(linear(h1, h_dim * 2, 'd1'))

    # without the minibatch layer, the discriminator needs an additional layer
    # to have enough capacity to separate the two distributions correctly
    if minibatch_layer:
        h3 = minibatch(h2)
    else:
        h3 = tf.nn.relu(linear(h2, h_dim * 2, scope='d2'))

    h4 = tf.sigmoid(linear(h3, 1, scope='d3'))
    return h4


def minibatch(input, num_kernels=5, kernel_dim=3):
    x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - \
        tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], 1)


def optimizer(loss, var_list, learning_rate):
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=step,
        var_list=var_list
    )
    return optimizer


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
        h_dim = params.hidden_size

        ## Generator
        self.z = tf.placeholder(tf.float32, shape=(params.batch_size, 1))
        input = self.z
        h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
        h1 = common_layer(h_dim, h0)
        h2 = linear(h1, 1, 'g1')
        self.G = h2

        # The discriminator tries to tell the difference between samples from
        # the true data distribution (self.x) and the generated samples
        # (self.z).
        #
        # Here we create two copies of the discriminator network
        # that share parameters, as you cannot use the same network with
        # different inputs in TensorFlow.
        self.x = tf.placeholder(tf.float32, shape=(params.batch_size, 1))

        minibatch_layer = params.minibatch

        # Discriminator1

        h0 = tf.nn.relu(linear(self.x, h_dim, 'd0'))
        h1 = common_layer(h_dim, h0, reuse=True)
        # h1 = tf.nn.relu(linear(h0, h_dim, 'dt'))
        h2 = tf.nn.relu(linear(h1, h_dim * 2, 'd1'))

        # without the minibatch layer, the discriminator needs an additional layer
        # to have enough capacity to separate the two distributions correctly
        if minibatch_layer:
            h3 = minibatch(h2)
        else:
            h3 = tf.nn.relu(linear(h2, h_dim * 2, scope='d2'))

        self.D1 = tf.sigmoid(linear(h3, 1, scope='d3'))
    
        ## Second Discriminator Network

        h0 = tf.nn.relu(linear(self.G, h_dim, 'd0', reuse=True))
        h1 = common_layer(h_dim, h0, reuse=True)

        # h1 = tf.nn.relu(linear(h0, h_dim, 'dt'))
        h2 = tf.nn.relu(linear(h1, h_dim * 2, 'd1', reuse=True))

        # without the minibatch layer, the discriminator needs an additional layer
        # to have enough capacity to separate the two distributions correctly
        if minibatch_layer:
            h3 = minibatch(h2)
        else:
            h3 = tf.nn.relu(linear(h2, h_dim * 2, scope='d2', reuse=True))

        self.D2 = tf.sigmoid(linear(h3, 1, scope='d3', reuse=True))

        # Define the loss for discriminator and generator networks
        # (see the original paper for details), and create optimizers for both
        self.loss_d = tf.reduce_mean(-log(self.D1) - log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-log(self.D2))

        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('d')]
        self.g_params = [v for v in vars if v.name.startswith('g')]

        self.d_params.extend([v for v in vars if v.name.startswith('shared_knowledge')])
        self.g_params.extend([v for v in vars if v.name.startswith('shared_knowledge')])

        print self.d_params
        print self.g_params

        self.opt_d = optimizer(self.loss_d, self.d_params, params.d_learning_rate)
        self.opt_g = optimizer(self.loss_g, self.g_params, params.g_learning_rate)

        d_loss_summary = tf.summary.scalar('d_loss', self.loss_d)
        g_loss_summary = tf.summary.scalar('g_loss', self.loss_g)

        g_out_summary = tf.summary.histogram('g_out', self.G)
        d_real_out_summary = tf.summary.histogram('d_real_out', self.D1)
        d_fake_out_summary = tf.summary.histogram('d_fake_out', self.D2)

        self.d_summary = tf.summary.merge([ d_loss_summary, d_real_out_summary ])
        self.g_summary = tf.summary.merge([ g_loss_summary, d_fake_out_summary, g_out_summary ])

        self.test_summary = tf.summary.merge([ g_out_summary, d_real_out_summary ])


def train(model, data, gen, params, index=0):
    anim_frames = [] 

    with tf.Session() as session:
        train_writer = tf.summary.FileWriter('log/'+str(index), session.graph)

        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        for step in range(params.num_steps + 1):

            if step%params.generator_skip != 0:
                # update discriminator
                x = data.sample(params.batch_size)
                z = gen.sample(params.batch_size)
                summary, loss_d, _, = session.run([model.d_summary, model.loss_d, model.opt_d], {
                    model.x: np.reshape(x, (params.batch_size, 1)),
                    model.z: np.reshape(z, (params.batch_size, 1))
                })

                train_writer.add_summary(summary, step)
            else:
                # update generator
                x = data.sample(params.batch_size)
                z = gen.sample(params.batch_size)
                summary, loss_d = session.run([model.d_summary, model.loss_d], {
                    model.x: np.reshape(x, (params.batch_size, 1)),
                    model.z: np.reshape(z, (params.batch_size, 1))
                })

                train_writer.add_summary(summary, step)

                z = gen.sample(params.batch_size)
                summary, loss_g, _ = session.run([model.g_summary, model.loss_g, model.opt_g], {
                    model.z: np.reshape(z, (params.batch_size, 1))
                })

                train_writer.add_summary(summary, step)

            if step % params.log_every == 0:
                print('{}: {:.4f}\t{:.4f}'.format(step, loss_d, loss_g))

            if params.anim_path and (step % params.anim_every == 0):
                anim_frames.append(
                    samples(model, session, data, gen.range, params.batch_size)
                )

        if params.anim_path:
            save_animation(anim_frames, params.anim_path, gen.range)
        else:
            samps = samples(model, session, data, gen.range, params.batch_size)
            plot_distributions(samps, gen.range)


def samples(
    model,
    session,
    data,
    sample_range,
    batch_size,
    num_points=10000,
    num_bins=100
):
    '''
    Return a tuple (db, pd, pg), where db is the current decision
    boundary, pd is a histogram of samples from the data distribution,
    and pg is a histogram of generated samples.
    '''
    xs = np.linspace(-sample_range, sample_range, num_points)
    bins = np.linspace(-sample_range, sample_range, num_bins)

    # decision boundary
    db = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        db[batch_size * i:batch_size * (i + 1)] = session.run(
            model.D1,
            {
                model.x: np.reshape(
                    xs[batch_size * i:batch_size * (i + 1)],
                    (batch_size, 1)
                )
            }
        )

    # data distribution
    d = data.sample(num_points)
    pd, _ = np.histogram(d, bins=bins, density=True)

    # generated samples
    zs = np.linspace(-sample_range, sample_range, num_points)
    g = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        g[batch_size * i:batch_size * (i + 1)] = session.run(
            model.G,
            {
                model.z: np.reshape(
                    zs[batch_size * i:batch_size * (i + 1)],
                    (batch_size, 1)
                )
            }
        )
    pg, _ = np.histogram(g, bins=bins, density=True)

    return db, pd, pg


def plot_distributions(samps, sample_range):
    db, pd, pg = samps
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))
    f, ax = plt.subplots(1)
    ax.plot(db_x, db, label='decision boundary')
    ax.set_ylim(0, 1)
    plt.plot(p_x, pd, label='real data')
    plt.plot(p_x, pg, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()


def save_animation(anim_frames, anim_path, sample_range):
    f, ax = plt.subplots(figsize=(6, 4))
    f.suptitle('1D Generative Adversarial Network', fontsize=15)
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1.4)
    line_db, = ax.plot([], [], label='decision boundary')
    line_pd, = ax.plot([], [], label='real data')
    line_pg, = ax.plot([], [], label='generated data')
    frame_number = ax.text(
        0.02,
        0.95,
        '',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes
    )
    ax.legend()

    db, pd, _ = anim_frames[0]
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))

    def init():
        line_db.set_data([], [])
        line_pd.set_data([], [])
        line_pg.set_data([], [])
        frame_number.set_text('')
        return (line_db, line_pd, line_pg, frame_number)

    def animate(i):
        frame_number.set_text(
            'Frame: {}/{}'.format(i, len(anim_frames))
        )
        db, pd, pg = anim_frames[i]
        line_db.set_data(db_x, db)
        line_pd.set_data(p_x, pd)
        line_pg.set_data(p_x, pg)
        return (line_db, line_pd, line_pg, frame_number)

    anim = animation.FuncAnimation(
        f,
        animate,
        init_func=init,
        frames=len(anim_frames),
        blit=True
    )
    anim.save(anim_path, fps=30, extra_args=['-vcodec', 'libx264'])


def main(args):

    index = 0
    while True:
      if not os.path.exists('log/'+str(index)):
        os.makedirs('log/'+str(index))
        break
      index+=1

    model = GAN(args)
    train(model, DataDistribution(), GeneratorDistribution(range=8), args, index)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=10000,
                        help='the number of training steps to take')
    parser.add_argument('--hidden-size', type=int, default=2,
                        help='MLP hidden size')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='the batch size')
    parser.add_argument('--minibatch', action='store_true',
                        help='use minibatch discrimination')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--anim-path', type=str, default=None,
                        help='path to the output animation file')
    parser.add_argument('--anim-every', type=int, default=1,
                        help='save every Nth frame for animation')
    parser.add_argument('--use-common-layer', type=bool, default=False,
                        help='Use the modified common layer')
    parser.add_argument('--g_learning_rate', type=float, default=0.0,
                        help='Change learning rate of generator')
    parser.add_argument('--d_learning_rate', type=float, default=0.001,
                        help='Change learning rate of discriminator')
    parser.add_argument('--generator_skip', type=float, default=1,
                        help='Skip generator updations')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
