import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import numpy as np


class Dnn1D:
    def __init__(self, feature_size, output_size):
        self.antibiotic_name = None
        self.feature_size = feature_size
        self.output_size = output_size
        self.X = None
        self.class_weights = None
        self.Y = None
        self.layers_outputs = {}
        self.dropouts = {}
        self.prediction = None
        self.unweighted_losses = None
        self.weighted_losses = None
        self.cost = None
        self.optimizer = None
        self.accuracy = None
        self.summary = None

    def create_model(self,
                     hidden_units=[128],
                     activation_functions=['relu'],
                     dropout_rate=0,
                     learning_rate=0.001,
                     optimizer_method='Adam'):

        # Input layer
        with tf.variable_scope('input'):
            self.X = tf.placeholder(tf.float32, shape=(None, self.feature_size))
            self.class_weights = tf.placeholder(tf.float32, shape=(self.output_size,))

        # Feet forward layers
        index = 1
        with tf.variable_scope('dl_1'):
            weights = tf.get_variable('weights_1',
                                      shape=[self.feature_size, hidden_units[0]],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('bias_1',
                                     shape=[hidden_units[0]],
                                     initializer=tf.zeros_initializer())

            if activation_functions[0] == 'relu':
                self.layers_outputs[index] = tf.nn.relu(tf.matmul(self.X, weights) + biases)
            elif activation_functions[0] == 'elu':
                self.layers_outputs[index] = tf.nn.elu(tf.matmul(self.X, weights) + biases)
            elif activation_functions[0] == 'leaky_relu':
                self.layers_outputs[index] = tf.nn.leaky_relu(tf.matmul(self.X, weights) + biases)
            elif activation_functions[0] == 'tanh':
                self.layers_outputs[index] = tf.math.tanh(tf.matmul(self.X, weights) + biases)
            else:
                raise Exception

        with tf.variable_scope('do_' + str(index)):
            self.dropouts[index] = tf.nn.dropout(self.layers_outputs[index], rate=dropout_rate)

        # Output layer
        with tf.variable_scope('output'):
            weights = tf.get_variable('weights_out',
                                      shape=[hidden_units[index-1], self.output_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('bias_out',
                                     shape=[self.output_size],
                                     initializer=tf.zeros_initializer())
            self.prediction = tf.matmul(self.dropouts[index], weights) + biases

        with tf.variable_scope('cost'):
            self.Y = tf.placeholder(tf.float32,
                                    shape=(None, self.output_size))

            weights = tf.reduce_sum(self.class_weights * self.Y, axis=1)

            self.unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.prediction)
            # apply the weights, relying on broadcasting of the multiplication
            self.weighted_losses = self.unweighted_losses * weights
            # reduce the result to get your final loss
            self.cost = tf.reduce_mean(self.weighted_losses)

        with tf.variable_scope('train'):
            if optimizer_method == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
            elif optimizer_method == 'RMSprop':
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.cost)
            elif optimizer_method == 'Adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.cost)
            elif optimizer_method == 'Adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(self.cost)
            else:
                raise Exception

        with tf.variable_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.Y, axis=1), tf.argmax(self.prediction, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Logging results
        with tf.variable_scope("logging"):
            # tf.summary.scalar('unweighted_current_cost', unweighted_cost)
            tf.summary.scalar('current_cost', self.cost)
            tf.summary.scalar('current_accuacy', self.accuracy)
            self.summary = tf.summary.merge_all()

        self.saver = tf.train.Saver()

    def clear_model(self):
        self.X = None
        self.class_weights = None
        self.Y = None
        self.layers_outputs = {}
        self.dropouts = {}
        self.prediction = None
        self.unweighted_losses = None
        self.weighted_losses = None
        self.cost = None
        self.optimizer = None
        self.accuracy = None
        self.summary = None

    def train_model(self, session, x, y_one_hot_encoded, class_weights):
        session.run(self.optimizer,
                    feed_dict={self.X: x,
                               self.Y: y_one_hot_encoded,
                               self.class_weights: class_weights})

    def get_cost(self, session, x, y_one_hot_encoded, class_weights):
        return session.run(self.cost,
                           feed_dict={self.X: x,
                                      self.Y: y_one_hot_encoded,
                                      self.class_weights: class_weights})

    def get_summary(self, session, x, y_one_hot_encoded, class_weights):
        return session.run(self.summary,
                           feed_dict={self.X: x,
                                      self.Y: y_one_hot_encoded,
                                      self.class_weights: class_weights})

    def get_accuracy(self, session, x, y_one_hot_encoded, class_weights):
        return session.run(self.accuracy,
                           feed_dict={self.X: x,
                                      self.Y: y_one_hot_encoded,
                                      self.class_weights: class_weights})

    def set_antibiotic_name(self, antibiotic_name):
        self.antibiotic_name = antibiotic_name


class Dnn2D:
    def __init__(self, feature_size, output_size):
        self.feature_size = feature_size
        self.output_size = output_size
        self.X = None
        self.class_weights = None
        self.Y = None
        self.layers_outputs = {}
        self.dropouts = {}
        self.prediction = None
        self.unweighted_losses = None
        self.weighted_losses = None
        self.cost = None
        self.optimizer = None
        self.accuracy = None
        self.summary = None

    def create_model(self,
                     hidden_units=[128, 64],
                     activation_functions=['relu', 'relu'],
                     dropout_rate=0,
                     learning_rate=0.001,
                     optimizer_method='Adam'):

        # Input layer
        with tf.variable_scope('input'):
            self.X = tf.placeholder(tf.float32, shape=(None, self.feature_size))
            self.class_weights = tf.placeholder(tf.float32, shape=(self.output_size,))

        # Feet forward layers
        index = 1
        with tf.variable_scope('dl_1'):
            weights = tf.get_variable('weights_1',
                                      shape=[self.feature_size, hidden_units[0]],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('bias_1',
                                     shape=[hidden_units[0]],
                                     initializer=tf.zeros_initializer())

            if activation_functions[0] == 'relu':
                self.layers_outputs[index] = tf.nn.relu(tf.matmul(self.X, weights) + biases)
            elif activation_functions[0] == 'elu':
                self.layers_outputs[index] = tf.nn.elu(tf.matmul(self.X, weights) + biases)
            elif activation_functions[0] == 'leaky_relu':
                self.layers_outputs[index] = tf.nn.leaky_relu(tf.matmul(self.X, weights) + biases)
            elif activation_functions[0] == 'tanh':
                self.layers_outputs[index] = tf.math.tanh(tf.matmul(self.X, weights) + biases)
            else:
                raise Exception

        with tf.variable_scope('do_' + str(index)):
            self.dropouts[index] = tf.nn.dropout(self.layers_outputs[index], rate=dropout_rate)

        index = index + 1
        with tf.variable_scope('dl_2'):
            weights = tf.get_variable('weights_2',
                                      shape=[self.feature_size, hidden_units[1]],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('bias_2',
                                     shape=[hidden_units[1]],
                                     initializer=tf.zeros_initializer())

            if activation_functions[1] == 'relu':
                self.layers_outputs[index] = tf.nn.relu(tf.matmul(self.X, weights) + biases)
            elif activation_functions[1] == 'elu':
                self.layers_outputs[index] = tf.nn.elu(tf.matmul(self.X, weights) + biases)
            elif activation_functions[1] == 'leaky_relu':
                self.layers_outputs[index] = tf.nn.leaky_relu(tf.matmul(self.X, weights) + biases)
            elif activation_functions[1] == 'tanh':
                self.layers_outputs[index] = tf.math.tanh(tf.matmul(self.X, weights) + biases)
            else:
                raise Exception

        with tf.variable_scope('do_' + str(index)):
            self.dropouts[index] = tf.nn.dropout(self.layers_outputs[index], rate=dropout_rate)

        # Output layer
        with tf.variable_scope('output'):
            weights = tf.get_variable('weights_out',
                                      shape=[hidden_units[-1], self.output_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('bias_out',
                                     shape=[self.output_size],
                                     initializer=tf.zeros_initializer())
            self.prediction = tf.matmul(self.dropouts[-1], weights) + biases

        with tf.variable_scope('cost'):
            self.Y = tf.placeholder(tf.float32,
                                    shape=(None, self.output_size))

            weights = tf.reduce_sum(self.class_weights * self.Y, axis=1)

            self.unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.prediction)
            # apply the weights, relying on broadcasting of the multiplication
            self.weighted_losses = self.unweighted_losses * weights
            # reduce the result to get your final loss
            self.cost = tf.reduce_mean(self.weighted_losses)

        with tf.variable_scope('train'):
            if optimizer_method == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
            elif optimizer_method == 'RMSprop':
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.cost)
            elif optimizer_method == 'Adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.cost)
            elif optimizer_method == 'Adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(self.cost)
            else:
                raise Exception

        with tf.variable_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.Y, axis=1), tf.argmax(self.prediction, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Logging results
        with tf.variable_scope("logging"):
            # tf.summary.scalar('unweighted_current_cost', unweighted_cost)
            tf.summary.scalar('current_cost', self.cost)
            tf.summary.scalar('current_accuacy', self.accuracy)
            self.summary = tf.summary.merge_all()


class DnnND:
    def __init__(self, feature_size, output_size, hidden_layer_size):
        self.feature_size = feature_size
        self.output_size = output_size
        self.N = hidden_layer_size
        self.X = None
        self.class_weights = None
        self.Y = None
        self.layers_outputs = {}
        self.dropouts = {}
        self.prediction = None
        self.unweighted_losses = None
        self.weighted_losses = None
        self.cost = None
        self.optimizer = None
        self.accuracy = None
        self.summary = None

    def create_model(self,
                     hidden_units=None,
                     activation_functions=None,
                     dropout_rate=0,
                     learning_rate=0.001,
                     optimizer_method='Adam'):

        if self.N != len(hidden_units) and len(hidden_units) != len(activation_functions):
            raise Exception

        # Input layer
        with tf.variable_scope('input'):
            self.X = tf.placeholder(tf.float32, shape=(None, self.feature_size))
            self.class_weights = tf.placeholder(tf.float32, shape=(self.output_size,))

        # Feet forward layers
        index = 1
        with tf.variable_scope('dl_1'):
            weights = tf.get_variable('weights_1',
                                      shape=[self.feature_size, hidden_units[0]],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('bias_1',
                                     shape=[hidden_units[0]],
                                     initializer=tf.zeros_initializer())

            if activation_functions[0] == 'relu':
                self.layers_outputs[index] = tf.nn.relu(tf.matmul(self.X, weights) + biases)
            elif activation_functions[0] == 'elu':
                self.layers_outputs[index] = tf.nn.elu(tf.matmul(self.X, weights) + biases)
            elif activation_functions[0] == 'leaky_relu':
                self.layers_outputs[index] = tf.nn.leaky_relu(tf.matmul(self.X, weights) + biases)
            elif activation_functions[0] == 'tanh':
                self.layers_outputs[index] = tf.math.tanh(tf.matmul(self.X, weights) + biases)
            else:
                raise Exception

        with tf.variable_scope('do_' + str(index)):
            self.dropouts[index] = tf.nn.dropout(self.layers_outputs[index], rate=dropout_rate)

        for i in range(1, len(hidden_units)):
            index = index + 1
            with tf.variable_scope('dl_' + str(index)):
                weights = tf.get_variable('weights_' + str(index),
                                          shape=[hidden_units[index - 1], hidden_units[index]],
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable('bias_' + str(index),
                                         shape=[hidden_units[index]],
                                         initializer=tf.zeros_initializer())
                if activation_functions[0] == 'relu':
                    self.layers_outputs[index] = tf.nn.relu(tf.matmul(self.dropouts[index - 1], weights) + biases)
                elif activation_functions[0] == 'elu':
                    self.layers_outputs[index] = tf.nn.elu(tf.matmul(self.dropouts[index - 1], weights) + biases)
                elif activation_functions[0] == 'leaky_relu':
                    self.layers_outputs[index] = tf.nn.leaky_relu(tf.matmul(self.dropouts[index - 1], weights) + biases)
                elif activation_functions[0] == 'tanh':
                    self.layers_outputs[index] = tf.math.tanh(tf.matmul(self.dropouts[index - 1], weights) + biases)
                else:
                    raise Exception

            with tf.variable_scope('do_' + str(index)):
                self.dropouts[index] = tf.nn.dropout(self.layers_outputs[index], rate=dropout_rate)

        with tf.variable_scope('cost'):
            self.Y = tf.placeholder(tf.float32,
                                    shape=(None, self.output_size))

            weights = tf.reduce_sum(self.class_weights * self.Y, axis=1)

            self.unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.prediction)
            # apply the weights, relying on broadcasting of the multiplication
            self.weighted_losses = self.unweighted_losses * weights
            # reduce the result to get your final loss
            self.cost = tf.reduce_mean(self.weighted_losses)

        with tf.variable_scope('train'):
            if optimizer_method == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
            elif optimizer_method == 'RMSprop':
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.cost)
            elif optimizer_method == 'Adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.cost)
            elif optimizer_method == 'Adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(self.cost)
            else:
                raise Exception

        with tf.variable_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.Y, axis=1), tf.argmax(self.prediction, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Logging results
        with tf.variable_scope("logging"):
            # tf.summary.scalar('unweighted_current_cost', unweighted_cost)
            tf.summary.scalar('current_cost', self.cost)
            tf.summary.scalar('current_accuacy', self.accuracy)
            self.summary = tf.summary.merge_all()