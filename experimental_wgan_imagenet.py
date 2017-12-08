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

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
# import seaborn as sns

import os
import json

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

class DataDistribution(object):
    def __init__(self):
        data_dir = "./data/"
        
        fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

        # fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
        # loaded = np.fromfile(file=fd,dtype=np.uint8)
        # trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

        # fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
        # loaded = np.fromfile(file=fd,dtype=np.uint8)
        # teY = loaded[8:].reshape((10000)).astype(np.float)

        # trY = np.asarray(trY)
        # teY = np.asarray(teY)
        
        X = np.concatenate((trX, teX), axis=0)
        
        np.random.shuffle(X)
        
        self.X = X/255.0


    def sample(self, N):
        indices = np.random.choice(range(self.X.shape[0]), N)
        # return self.X[indices].reshape([N, 28, 28, 1])
        return self.X[indices]


class GeneratorDistribution(object):
    def __init__(self):
        pass

    def sample(self, N):
        return (np.random.uniform(-1, 1, size=[N, 100]))


def generator(input, h_dim):
    # h0 = lrelu(linear(input, h_dim, 'g0'))
    # h1 = lrelu(linear(h0, h_dim, 'g1'))
    # h2 = lrelu(linear(h1, h_dim, 'gshared'))
    # # h3 = tf.nn.softplus(linear(h2, h_dim, 'g2'))
    # h4 = lrelu(linear(h2, h_dim, 'g3'))
    # h5 = linear(h4, 28*28, 'g4')

    gl0_bn = batch_norm(name='gl_bn0')
    g0_bn = batch_norm(name='g_bn0')
    g1_bn = batch_norm(name='g_bn1')
    g2_bn = batch_norm(name='g_bn2')
    gt2_bn = batch_norm(name='g_bnt2')
    gt3_bn = batch_norm(name='g_bnt3')
    
    g3_bn = batch_norm(name='g_bn3')

    print input.shape

    h0t = lrelu(gl0_bn(linear(input, 784, scope='g0t')))

    h0t = tf.reshape(h0t, tf.TensorShape([h0t.shape[0], 28, 28, 1 ]))

    h0 = lrelu(g0_bn(conv2d(h0t, h_dim, d_h=1, d_w=1, name='g0')))
    h1 = lrelu(g1_bn(conv2d(h0, h_dim, d_h=1, d_w=1, name='gshared')))
    h2 = lrelu(g2_bn(conv2d(h1, h_dim, d_h=1, d_w=1, name='g2')))
    # ht2 = lrelu(gt2_bn(conv2d(h2, h_dim, d_h=1, d_w=1, name='gt2')))
    ht3 = lrelu(gt3_bn(conv2d(h2, h_dim, d_h=1, d_w=1, name='gt3')))
    ht1 = lrelu(g3_bn(conv2d(ht3, h_dim, d_h=1, d_w=1, name='gt1')))
    h3 = tf.nn.sigmoid(conv2d(ht1, 1, d_h=1, d_w=1, name='g3'))

    print h3
    
    return h3


def discriminator(input, h_dim, minibatch_layer=True):
    # h0 = lrelu(linear(input, h_dim, 'd0'))
    # h1 = lrelu(linear(h0, h_dim, 'd1'))
    # # h2 = tf.nn.relu(linear(h1, h_dim, 'd2'))
    # h3 = lrelu(linear(h1, h_dim, 'dshared'))

    # # without the minibatch layer, the discriminator needs an additional layer
    # # to have enough capacity to separate the two distributions correctly
    # if minibatch_layer:
    #     h4 = minibatch(h3)
    # else:
    #     h4 = tf.nn.relu(linear(h3, h_dim , scope='d3'))

    # h5 = tf.sigmoid(linear(h4, 1, scope='d4'))
    # h6 = tf.sigmoid(linear(h5, h_dim, 'd5'))
    # h7 = tf.sigmoid(linear(h6, h_dim, 'd6'))

    h0 = lrelu(d0_bn(conv2d(input, h_dim, name='d0')))
    # ht = lrelu(dt_bn(conv2d(h0, h_dim, name='dt')))
    h1 = lrelu(d1_bn(conv2d(h0, h_dim, name='dshared')))
    h2 = lrelu(d2_bn(conv2d(h1, h_dim/2, name='d2')))
    h3 = lrelu(d3_bn(conv2d(h2, h_dim/4, name='d3')))
    # h4 = conv2d(h3, 1, name='d4')

    print h3

    h4 = tf.reshape(h3, tf.TensorShape([h3.shape[0], h3.shape[1]*h3.shape[2]*h3.shape[3]]))

    print h4

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
            self.G = generator(self.z, params.hidden_size)

        # The discriminator tries to tell the difference between samples from
        # the true data distribution (self.x) and the generated samples
        # (self.z).
        #
        # Here we create two copies of the discriminator network
        # that share parameters, as you cannot use the same network with
        # different inputs in TensorFlow.
        self.x = tf.placeholder(tf.float32, shape=(params.batch_size, 28, 28, 1))

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

        # self.loss_d = tf.reduce_mean(-log(self.D1) - log(1 - self.D2))
        # self.loss_g = tf.reduce_mean(-log(self.D2))

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

        # print "REmove attemp"
        # print self.g_params
        # print [v for v in vars if v.name.startswith('G/gshared/')][0]
        # print [v for v in vars if v.name.startswith('G/dshared/')]

        # self.g_params.remove([v for v in vars if v.name.startswith('G/gshared/w')][0])
        # self.g_params.remove([v for v in vars if v.name.startswith('G/gshared/b')][0])

        # self.d_params.remove([v for v in vars if v.name.startswith('D/dshared/w')][0])
        # self.d_params.remove([v for v in vars if v.name.startswith('D/dshared/b')][0])

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

        self.copy_d_w_g = shared_weights_g_w[0].assign(shared_weights_d_w[0])
        self.copy_d_b_g = shared_weights_g_b[0].assign(shared_weights_d_b[0])

        # self.copy_g_w_d = shared_weights_d_w[0].assign(shared_weights_g_w[0])
        # self.copy_g_b_d = shared_weights_d_b[0].assign(shared_weights_g_b[0])


def train(model, data, gen, params, index=0):
    anim_frames = [] 

    with tf.Session() as session:
        train_writer = tf.summary.FileWriter('log/'+str(index), session.graph)

        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        vars = tf.trainable_variables()
        analyze_g = [v for v in vars if v.name.startswith('G/g2/w')]

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
                gout, summary, loss_d, _, = session.run([analyze_g, model.d_summary, model.loss_d, model.opt_d], {
                    model.x: np.reshape(x, (params.batch_size, 28, 28, 1)),
                    # model.z: np.reshape(z, (params.batch_size, 28, 28, 1))
                    model.z: z
                })

                train_writer.add_summary(summary, step)

            # session.run([model.copy_d_w_g, model.copy_d_b_g])
            # session.run([model.copy_d_w_g])

            # update generator
            z1 = gen.sample(params.batch_size)
            
            summary, loss_g, _ = session.run([model.g_summary, model.loss_g, model.opt_g], {
                # model.z: np.reshape(z1, (params.batch_size, 28, 28, 1))
                model.z: z1
            })

            train_writer.add_summary(summary, step)
            
            if step % params.log_every == 0:
                print('Epoch:{} step:{}: d:{}\tg:{}'.format(int(step*params.batch_size/70000), step, loss_d, loss_g))

                if params.show_output:
                    plot_gen_out(session, model, gen, params.batch_size, str(step), z1)

            if step % 1000 == 0:
                save_path = saver.save(session, "checkpoints/model.ckpt")
                print("Model saved in file: %s" % save_path)
                with open('checkpoints/global_step.json', 'w') as fp:
                    json.dump({ "step": step }, fp)

        plot_gen_out(session, model, gen, params.batch_size, "final_out")


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



def plot_gen_out(sess, model, gen_distribution, N, filename, ginput=None):
    if ginput == None:
        ginput = gen_distribution.sample(N)

    gen_out = sess.run(model.G, { model.z: ginput })

    for x in range(25):
        plt.subplot(5, 5, x+1)
        plt.imshow(np.reshape(gen_out[x, :, :, 0], [28, 28]), cmap="gray")

    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()

    # plt.grid(True)

    plt.tight_layout()
    plt.savefig("plots/"+filename+".png", dpi=200)
    # plt.clf()



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

    print g
    print "mean ", np.mean(np.array(g))
    print "stddev", np.std(np.array(g))

    return db, pd, pg


def plot_distributions(samps, sample_range):
    db, pd, pg = samps
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))
    # f, ax = plt.subplots(1)
    # ax.plot(db_x, db, label='decision boundary')
    # ax.set_ylim(0, 1)
    # plt.plot(p_x, pd, label='real data')
    # plt.plot(p_x, pg, label='generated data')
    # plt.title('1D Generative Adversarial Network')
    # plt.xlabel('Data values')
    # plt.ylabel('Probability density')
    # plt.legend()
    # plt.show()


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
    model = GAN(args)

    mnist = DataDistribution()

    print args

    if args.mode=="train":
        train(model, DataDistribution(), GeneratorDistribution(range=8), args)

    elif args.mode == "test":
        test(model, GeneratorDistribution(), args)

    else:
        print "Mode not found"


def parse_args():

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0', 'True'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_session', type=str2bool, default=True,
                        help='restore session')
    parser.add_argument('--num-steps', type=int, default=50000,
                        help='the number of training steps to take')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='MLP hidden size')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='the batch size')
    parser.add_argument('--minibatch', action='store_false',
                        help='use minibatch discrimination')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--anim-path', type=str, default=None,
                        help='path to the output animation file')
    parser.add_argument('--anim-every', type=int, default=1,
                        help='save every Nth frame for animation')
    parser.add_argument('--use-common-layer', type=str2bool, default=False,
                        help='Use the modified common layer')
    parser.add_argument('--d_learning_rate', type=float, default=5e-5,
                        help='Change learning rate of generator')
    parser.add_argument('--g_learning_rate', type=float, default=5e-5,
                        help='Change learning rate of discriminator')
    parser.add_argument('--generator_train', type=float, default=10,
                        help='Change generator training time')
    parser.add_argument('--discriminator_train', type=float, default=5,
                        help='Change discriminator training time')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Beta1 momentum adam')
    parser.add_argument('--clip_value', type=float, default=0.01,
                        help='Clip value of discriminator')
    parser.add_argument('--show_output', type=str2bool, default=True,
                        help='show output as plot')
    parser.add_argument('--mode', type=str, default="test",
                        help='mode of the system(test/train)')
    parser.add_argument('--test_count', type=int, default=1,
                        help='number of tests to be done')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())