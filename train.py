#	coding:utf-8
#
#	Copyright @makai
#
#	16/10/25
#

from __future__ import division, print_function, absolute_import
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle


TRAIN_PATH = 'data/traindata/'
TEST_PATH = 'data/testdata/'
TRAIN_DATA_PATH = 'savedData/train_data.p'
TEST_DATA_PATH = 'savedData/test_data.p'
ENCODER_H1_PATH = 'savedData/encoder_h1.p'
# ENCODER_H2_PATH = 'savedData/encoder_h2.p'
ENCODER_B1_PATH = 'savedData/encoder_b1.p'
# ENCODER_B2_PATH = 'savedData/encoder_b2.p'
DECODER_H1_PATH = 'savedData/decoder_h1.p'
# DECODER_H2_PATH = 'savedData/decoder_h2.p'
DECODER_B1_PATH = 'savedData/decoder_b1.p'
# DECODER_B2_PATH = 'savedData/decoder_b2.p'


learning_rate = 0.01
traning_epochs = 20
batch_size = 256

n_hidden_1 = 1024
# n_hidden_2 = 512
n_input = 2600
a = 1

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    # 'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    # 'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    # 'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_input])),
    # 'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    # layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   # biases['encoder_b2']))
    return layer_1


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    # layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
    #                                biases['decoder_b2']))
    return layer_1


def getMFCCs(dirpath):
	filelist = os.listdir(dirpath)
	data = np.zeros((len(filelist), n_input))
	for i, filename in enumerate(filelist):
		filepath = dirpath + filename
 		y, sr = librosa.load(filepath)
	 	mfcc = np.array(librosa.feature.mfcc(y=y, sr=sr))
	 	mfcc = np.reshape(mfcc, (2600))
	 	data[i] = mfcc
	 	if i % 50 == 0:
	 		print(filepath)
	return data


def autoencoder():
	X = tf.placeholder('float', [None, n_input])

	encoder_op = encoder(X)
	decoder_op = decoder(encoder_op)
	y_pred = decoder_op
	y_true = X

	cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	init = tf.initialize_all_variables()
	train_data = np.zeros((1, n_input))
	if os.path.exists(TRAIN_DATA_PATH):
		train_data = pickle.load(file(TRAIN_DATA_PATH, 'rb'))
	else:
		train_data = getMFCCs(TRAIN_PATH)
		pickle.dump(train_data, file(TRAIN_DATA_PATH, 'wb'))
	# print('train_data:', train_data)
	for i in xrange(0, 50):
		print('train_data:', i, train_data[:5])

	with tf.Session() as sess:
		sess.run(init)
		n_batch = int(len(train_data) / batch_size)
		lastepoch = 0
		for epoch in range(traning_epochs):
			for i in range(n_batch):
				batch_xs = train_data[i * batch_size : (i + 1) * batch_size]
				_, c ,y = sess.run([optimizer, cost, decoder_op], feed_dict={X: batch_xs})
				# print 'Epoch: ' + str(epoch) + '  cost = ' + str(c)
        	if epoch % a == 0:
            	print "Epoch:" + str(epoch) + "cost= " + str(c)
		test_data = np.zeros((1, n_input))
		if os.path.exists(TEST_DATA_PATH):
			test_data = pickle.load(file(TEST_DATA_PATH, 'rb'))
		else:
			test_data = getMFCCs(TEST_PATH)
			pickle.dump(test_data, file(TEST_DATA_PATH, 'wb'))

		encode_decode = sess.run(y_pred, feed_dict={X: test_data})
		for i in xrange(0, 50):
			print('test_data:', i, test_data[:5])
		print(encode_decode)
		f, a = plt.subplots(2, 2, figsize=(2, 2))
		for i in range(2):
			a[0][i].imshow(np.reshape(test_data[i], (20, 130)))
			a[0][i].imshow(np.reshape(encode_decode[i], (20, 130)))
		f.show()
		plt.draw()
		plt.waitforbuttonpress()


autoencoder()