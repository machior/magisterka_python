import csv
import numpy as np
import tensorflow as tf

# X = np.array([
#     [0, 0, 1],
#     [0, 1, 1],
#     [1, 0, 1],
#     [1, 1, 1]
#     ])
#
# Y = np.array([[0, 0, 1, 1]])
def get_data(path):
    l = []

    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        spamreader.__next__()
        for row in spamreader:
            separated = row[0].split(',')
            l.append([float(val) for val in separated])

    return np.array(l)

forgery_path = '/media/bartek/120887D50887B5EF/POLITECHNIKA/Magisterka/SUSigP/DataProcessed/BlindSubCorpus/FORGERY/001_f.csv'
genuine_path = '/media/bartek/120887D50887B5EF/POLITECHNIKA/Magisterka/SUSigP/DataProcessed/BlindSubCorpus/GENUINE/001_g.csv'

forg = get_data(forgery_path)
outs = len(forg) * [[0.]]
gen = get_data(genuine_path)
outs.extend(len(gen) * [[1.]])
outs = np.array(outs)

trX = np.concatenate((forg, gen))
trY = np.array(outs)


X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
XOR_X = np.array(trX)
XOR_Y = np.array(trY)

LEARNING_RATE = 0.01
EPOCHS = int(1e5)
EPOCH_RAPORT = int(1e3)

H_LAYER = 20

x_ = tf.placeholder(tf.float32, shape=XOR_X.shape, name="x-input")
y_ = tf.placeholder(tf.float32, shape=XOR_Y.shape, name="y-input")

Theta1 = tf.Variable(tf.random_uniform([XOR_X.shape[1], H_LAYER], -1, 1), name="Theta1")
Theta2 = tf.Variable(tf.random_uniform([H_LAYER, XOR_Y.shape[1]], -1, 1), name="Theta2")

Bias1 = tf.Variable(tf.zeros(H_LAYER), name="Bias1")
Bias2 = tf.Variable(tf.zeros(XOR_Y.shape[1]), name="Bias2")

A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)
Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

cost = tf.reduce_mean(((y_ * tf.log(Hypothesis)) +
                       ((1 - y_) * tf.log(1.0 - Hypothesis))) * -1)

train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()

writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph_def)

sess.run(init)

for i in range(EPOCHS):
    sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})

    if i % 1000 == 0:
        print('Epoch ', i)
        print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y}))
        print('Theta1 ', sess.run(Theta1))
        print('Bias1 ', sess.run(Bias1))
        print('Theta2 ', sess.run(Theta2))
        print('Bias2 ', sess.run(Bias2))
        print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
