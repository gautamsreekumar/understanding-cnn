# understanding-cnn
Partial implementation of Visualizing and Understanding CNN paper by Matthew Zeiler and Rob Fergus

Classification of CIFAR10 dataset using convolutional neural networks.
Parts of the paper Visualizing and Understanding Neural Networks are also implemented

The program is easily scalable. All the hyperparameters can be set at runtime. Only one model (at the end
of training) will be saved.

Architecture used

5 layers with filters of size 5, 5, 5, 3, and 3.
Each filter has a stride of 1.
Number of channels in each layer go as 40, 64, 64, 128, 128.
Max pooling is carried out with filter size 2, 2, 1, 1, 1 and
with strides 1, 2, 1, 1, 1.
Two fully connected layers are used, each with 256 neurons

Run the main file as the following:

python main.py -h

to know how to set the hyperparameters

There are provisions for:

1. identifying filters that have learnt meaningful features
2. understanding how greying out parts will affect classification
