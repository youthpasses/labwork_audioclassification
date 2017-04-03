# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("data/", one_hot=True)


TRAIN_PATH = 'data/traindata/'
TEST_PATH = 'data/testdata/'
TRAIN_DATA_PATH = 'savedData/train_data.p'
TEST_DATA_PATH = 'savedData/test_data.p'

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 3

# Network Parameters
n_hidden_1 = 1024 # 1st layer num features
n_hidden_2 = 800 # 2nd layer num features
n_input = 2600 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()
train_data = np.zeros((1, n_input))
if os.path.exists(TRAIN_DATA_PATH):
    train_data = pickle.load(file(TRAIN_DATA_PATH, 'rb'))
else:
    train_data = getMFCCs(TRAIN_PATH)
    pickle.dump(train_data, file(TRAIN_DATA_PATH, 'wb'))
for i in xrange(0, 50):
    print('train_data:', i, train_data[i][:5])

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(len(train_data) / batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        # lastepoch = 0
        for i in range(total_batch):
            # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = train_data[i * batch_size : (i + 1) * batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, y = sess.run([optimizer, cost, decoder_op], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))
            # if lastepoch != epoch:
            #     print(y)
            #     lastepoch = epoch


    print("Optimization Finished!")
    test_data = np.zeros((1, n_input))
    if os.path.exists(TEST_DATA_PATH):
        test_data = pickle.load(file(TEST_DATA_PATH, 'rb'))
    else:
        test_data = getMFCCs(TEST_PATH)
        pickle.dump(test_data, file(TEST_DATA_PATH, 'wb'))
    encode_decode = sess.run(y_pred, feed_dict={X: test_data})
    for i in xrange(0, 50):
        print('test_data:', i, test_data[i][:5])
    for i in xrange(0, 50):
        print('pred_data:', i, encode_decode[i][:5])

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: test_data[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 3, figsize=(3, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(test_data[i], (20, 130)))
        a[1][i].imshow(np.reshape(encode_decode[i], (20, 130)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
