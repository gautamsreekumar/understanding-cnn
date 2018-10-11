import argparse
import os
import numpy as np

from model import cifar
import tensorflow as tf

parser = argparse.ArgumentParser()

"""
sess : TF session in use
f : list of filter sizes
c_layers : number of convolutional layers
fc_u : number of units in fully connected layers
fc_layers : number of fully connected layers
s : list of filter strides in convolutional layers
mp_s : list of max pool filter strides
c : list of channels in each convolutional layers

length : side length of each image
tr_sample : number of samples in training set
te_sample : number of samples in testing set
va_sample : number of samples in validation set (made separately)
batch_size : batch size to be used
epochs : number of epochs to be trained for
lr : learning rate for the optimizer
"""

parser.add_argument('--epochs', dest='epochs', type=int, default=40,
	help='# of epochs')
parser.add_argument('--f', dest='f', type=int, nargs="*", default=[3, 3],
	help='filter side')
parser.add_argument('--c_layers', dest='c_layers', type=int, default=2,
	help='number of convolutional layers')
parser.add_argument('--fc_u', dest='fc_u', type=int, nargs="*", default=[128, 64],
	help='number of units in fc layers')
parser.add_argument('--fc_layers', dest='fc_layers', type=int, default=2,
	help='number of fc layers')
parser.add_argument('--s', dest='s', type=int, nargs="*", default=[1, 1],
	help='filter strides in convolutional layers')
parser.add_argument('--mp', dest='mp', type=int, nargs="*", default=[1, 1],
	help='filter sides in max pool layers')
parser.add_argument('--mp_s', dest='mp_s', type=int, nargs="*", default=[1, 1],
	help='filter strides in max pool layers')
parser.add_argument('--c', dest='c', type=int, nargs="*", default=[16, 32],
	help='number of channels in convolutional layers')
parser.add_argument('--tr_sample', dest='tr_sample', default=50000, type=int,
    help='number of data points in training set')
parser.add_argument('--te_sample', dest='te_sample', default=10000, type=int,
    help='number of data points in testing set')
parser.add_argument('--va_sample', dest='va_sample', default=10000, type=int,
    help='number of data points in validation sets. reduce this number from training set to get accurate results')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=10,
	help='# images in batch')
parser.add_argument('--phase', dest='phase', default='train',
	help='train, test, show_misclassified, occlusion_sensitivity, filter_analysis, filter_modification')
parser.add_argument('--lr', dest='lr', default=0.0001, type=float,
    help='learning rate')
parser.add_argument('--dropout', dest='dropout', default=False, type=bool,
    help='whether to add dropout or not')
parser.add_argument('--batch_norm', dest='batch_norm', default=False, type=bool,
    help='whether to add batch normalization or not')
parser.add_argument('--continue_training', dest='continue_training', default=False, type=bool,
    help='whether to continue training with saved model or not')
parser.add_argument('--init_mean', dest='init_mean', default=0.0, type=float,
    help='mean for normal initialization of parameters')
parser.add_argument('--init_stddev', dest='init_stddev', default=0.01, type=float,
    help='standard deviation for normal initialization of parameters')
parser.add_argument('--seed', dest='seed', default=0, type=int,
    help='random seed for tensorflow initialization')
parser.add_argument('--reg_factor', dest='reg_factor', default=0.0000001, type=float,
    help='regularization factor')
parser.add_argument('--p_size', dest='p_size', default=3, type=int,
    help='occlusion patch size')

args = parser.parse_args()

def main(_):
    with tf.Session() as sess:
        model = cifar(sess=sess,
                        f=args.f,
                        c_layers=args.c_layers,
                        fc_u=args.fc_u,
                        fc_layers=args.fc_layers,
                        s=args.s,
                        mp=args.mp,
                        mp_s=args.mp_s,
                        c=args.c,
                        length=32,
                        tr_sample=args.tr_sample,
                        te_sample=args.te_sample,
                        va_sample=args.va_sample,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        lr=args.lr,
                        dropout=args.dropout,
                        batch_norm=args.batch_norm,
                        continue_training=args.continue_training,
                        reg_factor=args.reg_factor,
                        init_mean=args.init_mean,
                        init_stddev=args.init_stddev,
                        seed=args.seed,
                        p_size=args.p_size)

        if args.phase == 'train':
            model.train_model()
        elif args.phase == 'test':
            model.test_model()
        elif args.phase == 'misclassified':
            model.show_misclassified()
        elif args.phase == 'occlusion_sensitivity':
            model.occlusion_sensitivity()
        elif args.phase == 'filter_analysis':
            model.filter_analysis()
        elif args.phase == 'filter_modification':
            model.filter_modification()
        else:
        	print "Invalid argument for --phase"

if __name__ == '__main__':
    tf.app.run()