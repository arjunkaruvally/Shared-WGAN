import sys
import os
import json
import scipy

from models_experimental import *

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
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='MLP hidden size')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--minibatch', action='store_false',
                        help='use minibatch discrimination')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--anim-path', type=str, default=None,
                        help='path to the output animation file')
    parser.add_argument('--anim-every', type=int, default=1,
                        help='save every Nth frame for animation')
    parser.add_argument('--shared', type=str2bool, default=True,
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
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset to train on')
    parser.add_argument('--input_dim', type=int, default=100,
                        help='Dimension of input')
    return parser.parse_args()


if __name__ == '__main__':
    a = parse_args()
    # print dict(a)
    main(parse_args())