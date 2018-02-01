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

teX = trX[::2]
teY = trY[::2]
trX = np.array(trX[1::2])
trY = np.array(trY[1::2])


X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
XOR_X = np.array(trX)
XOR_Y = np.array(trY)
INPUTS_AMOUNT = XOR_X.shape[-1]
OUTPUTS_AMOUNT = XOR_Y.shape[-1]


LEARNING_RATE = 0.01
EPOCHS = int(1e4)
EPOCH_RAPORT = int(1e3)

H_LAYER = 20

COST = 'cost3'

x_ = tf.placeholder(tf.float32, shape=XOR_X.shape, name="x-input")
y_ = tf.placeholder(tf.float32, shape=XOR_Y.shape, name="y-input")


def init_weights_biases(shape, summary_name=None):
    weight = tf.Variable(tf.random_uniform(shape, -1, 1), name=summary_name)
    # weight = tf.cast(weight, tf.float64)

    if type(summary_name) is str:
        tf.summary.histogram(summary_name, weight)

    biases = tf.Variable(tf.zeros([shape[-1]]))
    # biases = tf.cast(biases, tf.float64)

    return weight, biases


def create_layer(in_data, n_amount, activation, summary_name=None):
    in_amount = in_data._shape_as_list()[-1]
    weights, biases = init_weights_biases([in_amount, n_amount], summary_name)

    data = tf.matmul(in_data, weights)
    data = tf.add(data, biases)
    data = activation(data)

    return data


def neural_network_model(data):
    hl1 = create_layer(
        in_data=data,
        n_amount=H_LAYER,
        activation=tf.sigmoid,
        summary_name='h_l'
    )
    output = create_layer(
        in_data=hl1,
        n_amount=XOR_Y.shape[-1],
        activation=tf.sigmoid,
        summary_name='out'
    )

    return output



# Theta1 = tf.Variable(tf.random_uniform([XOR_X.shape[1], H_LAYER], -1, 1), name="Theta1")
# Theta2 = tf.Variable(tf.random_uniform([H_LAYER, XOR_Y.shape[1]], -1, 1), name="Theta2")
#
# Bias1 = tf.Variable(tf.zeros(H_LAYER), name="Bias1")
# Bias2 = tf.Variable(tf.zeros(XOR_Y.shape[1]), name="Bias2")
#
# A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)
# Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

Hypothesis = neural_network_model(x_)

cost = tf.reduce_mean(((y_ * tf.log(Hypothesis)) +
                       ((1 - y_) * tf.log(1.0 - Hypothesis))) * -1)
tf.summary.scalar(COST, cost)

train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

with tf.Session() as sess:

    writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph_def)
    merged = tf.summary.merge_all()

    tf.global_variables_initializer().run()

    for i in range(EPOCHS):
        sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
        # TODO TEST DATA
        summary, acc = sess.run([merged, cost], feed_dict={x_: teX, y_: teY})
        writer.add_summary(summary, i)  # Write summary

        if i % 1000 == 0:
            print('Epoch ', i)
            print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: teX, y_: teY}))
            # print('Theta1 ', sess.run(Theta1))
            # print('Bias1 ', sess.run(Bias1))
            # print('Theta2 ', sess.run(Theta2))
            # print('Bias2 ', sess.run(Bias2))
            print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
