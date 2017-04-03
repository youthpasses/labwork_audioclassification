#Inspired by https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/recurrent_network.py
import tensorflow as tf
import numpy as np
import input_data

# configuration
#                        O * W + b -> 10 labels for each image, O[? 28], W[28 10], B[10]
#                       ^ (O: output 28 vec from 28 vec input)
#                       |
#      +-+  +-+       +--+
#      |1|->|2|-> ... |28| time_step_size = 28
#      +-+  +-+       +--+
#       ^    ^    ...  ^
#       |    |         |
# img1:[28] [28]  ... [28]
# img2:[28] [28]  ... [28]
# img3:[28] [28]  ... [28]
# ...
# img128 or img256 (batch_size or test_size 256)
#      each input size = input_vec_size=lstm_size=28

# configuration variables

time_step_size = 20 #128
batch_size = 128
test_size = 128

input_vec_size = lstm_size = 130 #216 #130
n_classes = 10
n_rnnlayers = 2
n_iter = 100

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, W, B, lstm_size):
    # X, input shape: (batch_size, time_step_size, input_vec_size)
    XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size
    # XT shape: (time_step_size, batch_size, input_vec_size)
    XR = tf.reshape(XT, [-1, lstm_size]) # each row has input for each lstm cell (lstm_size=input_vec_size)
    # XR shape: (time_step_size * batch_size, input_vec_size)
    X_split = tf.split(0, time_step_size, XR) # split them to time_step_size (28 arrays)
    # Each array shape: (batch_size, input_vec_size)

    # Make lstm with lstm_size (each input vector size)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.75)
    lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * n_rnnlayers)
    
    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    outputs, _states = tf.nn.rnn(lstm, X_split, dtype=tf.float32)

    # Linear activation
    # Get the last output
    return tf.matmul(outputs[-1], W) + B, lstm.state_size # State size to initialize the stat

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# print trX.shape, trY.shape
trX, trY, vaX, vaY, teX, teY = input_data.getMFCCFeatures()
# trX, trY, teX, teY = input_data.getLMSFeatures()
print trX.shape, trY.shape, vaX.shape, vaY.shape, teX.shape, teY.shape
# trX = trX.reshape(-1, 28, 28)
# teX = teX.reshape(-1, 28, 28)

X = tf.placeholder("float", [None, time_step_size, lstm_size])
Y = tf.placeholder("float", [None, n_classes])

# get lstm_size and output 10 labels
W = init_weights([lstm_size, n_classes])
B = init_weights([n_classes])

py_x, state_size = model(X, W, B, lstm_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)
correct_pred = tf.equal(predict_op, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True

# Launch the graph in a session
with tf.Session(config=session_conf) as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(n_iter):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
            _, loss, acc = sess.run([train_op, cost, accuracy], feed_dict={X: trX[start:end], Y: trY[start:end]})
	validation_acc = sess.run(accuracy, feed_dict={X:vaX, Y:vaY})
	print 'Training ' + str(i) + ': loss=' + str(loss) + ', acc=' + str(acc) + ', vali_acc=' + str(validation_acc)
    print 'Optimization Finished!'
    test_acc = sess.run(accuracy, feed_dict={X:teX, Y:teY})
    print 'Test acc = ' + str(test_acc)


