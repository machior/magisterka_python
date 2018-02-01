from features.ExtractFeatures import extract_features
from tkinter.filedialog import askopenfilename

from features.GroupSigns import get_files
import tkinter as tk
from tkinter import filedialog


class SelectionWindow:
    out_path = None

    def __init__(self, in_path):
        self.in_path = in_path
        root = tk.Tk()

        self.in_path_string = tk.StringVar(value=in_path)
        self.out_path_string = tk.StringVar()
        self.recursively = tk.BooleanVar(root, value=False)
        tk.Button(root, text="Source Directory", command=self.get_source_dir).pack(side=tk.TOP)
        tk.Label(None, textvariable=self.in_path_string, fg='black').pack()
        tk.Button(root, text="Destination Directory", command=self.get_destination_dir).pack(side=tk.TOP)
        tk.Label(None, textvariable=self.out_path_string, fg='black').pack()

        tk.Checkbutton(root, text="Recursively", variable=self.recursively).pack()

        tk.Button(root, text="Start", command=self.get_files).pack(side=tk.TOP)
        root.mainloop()

    def get_source_dir(self):
        self.in_path = filedialog.askdirectory(initialdir=self.in_path)
        self.in_path_string.set(self.in_path)

    def get_destination_dir(self):
        self.out_path = filedialog.askdirectory(initialdir=self.out_path)
        self.out_path_string.set(self.out_path)

    def get_files(self):
        if self.in_path is not None and self.out_path is not None:
            get_files(
                file_name='all',
                in_dir_path=self.in_path,
                out_dir_path=self.out_path,
                recursively=self.recursively.get()
            )


if __name__ == '__main__':
    # dir_path = '/media/bartek/120887D50887B5EF/POLITECHNIKA/Magisterka/SUSigP/Data/BlindSubCorpus/FORGERY'
    #
    # SelectionWindow(dir_path)

    extract_features(
        in_dir_path='/media/bartek/120887D50887B5EF/POLITECHNIKA/Magisterka/SUSigP/Data/BlindSubCorpus',
        out_dir_path='/media/bartek/120887D50887B5EF/POLITECHNIKA/Magisterka/SUSigP/results',
        signatory_nr='039',
        files_urls=['/FORGERY/039_f_9.sig']
    )

    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    n_inputs = 784

    n_nodes_hl1 = 500
    n_nodes_hl2 = 500
    n_nodes_hl3 = 500

    n_classes = 10
    batch_size = 100

    # height x width
    x = tf.placeholder(dtype='float', shape=[None, n_inputs])
    y = tf.placeholder(dtype='float')


    def neural_network_model(data):
        hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_inputs, n_nodes_hl1])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

        # hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        #                   'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

        hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

        output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                        'biases': tf.Variable(tf.random_normal([n_classes]))}

        # (input_data * weights) + biases

        l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        # l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        # l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l1, hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.relu(l3)

        output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

        return output


    def train_neural_network(x):
        prediction = neural_network_model(x)
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
        )

        # learning rate
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        hm_epochs = 10

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in range(int(mnist.train.num_examples/batch_size)):
                    epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c
                print('Epoch ', epoch, ' completed out of ', hm_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    train_neural_network(x)
