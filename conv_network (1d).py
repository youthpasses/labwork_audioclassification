
from __future__ import print_function
import numpy as np
import os
import Image
import tensorflow as tf


TRAIN_DATA_DIR = 'data/image_train/'
TEST_DATA_DIR = 'data/image_test/'

# Parameters
learning_rate = 0.001
training_iters = 30000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 4096
n_classes = 10
dropout = 0.90

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


def getData(filedir):
    imagelist = os.listdir(filedir)
    imagecount = len(imagelist)
    data = []
    label = []
    for i, imagename in enumerate(imagelist):
        if imagename.split('.')[-1] == 'png':
            imagepath = filedir + imagename
            im = np.array(Image.open(imagepath))
            image = np.reshape(im, (n_input))
            data.append(image)
            label.append(int(imagename.split('_')[0]))
    data = np.array(data)
    data = data / 255.
    imagecount = data.shape[0]
    label1 = np.zeros((imagecount, n_classes))
    label1[np.arange(imagecount), label] = 1
    return data, label1

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, 1, 1], strides=[1, k, 1, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 64, 64, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    fc1 = tf.reshape(conv2, [-1, 16 * 64 * 64])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 16*16*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([16 * 64 * 64, 1024])),
    # 1024 inputs, 5 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    train_data, train_label = getData(TRAIN_DATA_DIR)
    test_data, test_label = getData(TEST_DATA_DIR)
    j = 0
    while step * batch_size < training_iters:
        if j >= train_data.shape[0] / batch_size:
            j = 0
        batch_x = train_data[j * batch_size : (j + 1) * batch_size]
        batch_y = train_label[j * batch_size : (j + 1) * batch_size]
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            test_acc = sess.run(accuracy, feed_dict={x: test_data, y: test_label, keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc) + ", Test Accuracy= " + "{:0.5f}".format(test_acc))
        step += 1
        j += 1
    print("Optimization Finished!")
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label, keep_prob: 1.}))