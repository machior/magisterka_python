from parameters import *


x_ = tf.placeholder(tf.float32, shape=XOR_X.shape, name="x-input")
y_ = tf.placeholder(tf.float32, shape=XOR_Y.shape, name="y-input")


def init_weights_biases(shape, summary_name=None):
    weight = tf.Variable(tf.random_uniform(shape, -1, 1), name=summary_name)

    if type(summary_name) is str:
        tf.summary.histogram(summary_name, weight)

    biases = tf.Variable(tf.zeros([shape[-1]]))

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
        summary_name='hidden1_' + TRIAL_NAME
    )
    output = create_layer(
        in_data=hl1,
        n_amount=XOR_Y.shape[-1],
        activation=tf.sigmoid,
        summary_name='out_' + TRIAL_NAME
    )

    return output


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
        summary, acc = sess.run([merged, cost], feed_dict={x_: teX, y_: teY})
        writer.add_summary(summary, i)  # Write summary

        if i % EPOCH_RAPORT == 0:
            print('Epoch ', i)
            # print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: teX, y_: teY}))
            # print('Theta1 ', sess.run(Theta1))
            # print('Bias1 ', sess.run(Bias1))
            # print('Theta2 ', sess.run(Theta2))
            # print('Bias2 ', sess.run(Bias2))
            print('cost train ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
            print('cost test  ', sess.run(cost, feed_dict={x_: teX, y_: teY}))
