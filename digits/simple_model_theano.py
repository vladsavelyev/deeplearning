import itertools
import json
import math
import sys
from typing import List, Dict

import numpy as np
import theano
import theano.tensor as T
from theano.compile import SharedVariable

from digits.mnist_loader import load_data, digits_to_binary_arrays
from digits.run_benchmarks import run_benchmarks

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid


def main():
    all_hparams = [
        dict(inercia=0,   subset=5000, hidden_layers=(100,), learning_rate=3.0,
             batch_size=10, regul_param=1.0, early_stop=5),
        # dict(inercia=0.01, subset=5000, hidden_layers=(100,), epochs=100, learning_rate=3.0,
        #      batch_size=10, regul_param=1.0, early_stop=1),
        # dict(inercia=0.01, subset=5000, hidden_layers=(30,),  epochs=100, learning_rate=3.0,
        #      batch_size=10, regul_param=1.0, early_stop=1),
        # dict(inercia=0.01, subset=5000, hidden_layers=(30,),  epochs=100, learning_rate=1.0,
        #      batch_size=10, regul_param=1.0, early_stop=1),
        # dict(inercia=0.01, subset=1000, hidden_layers=(30,),  epochs=100, learning_rate=3.0,
        #      batch_size=10, regul_param=1.0, early_stop=1),
        # dict(inercia=0.01, subset=5000, hidden_layers=(30,),  epochs=100, learning_rate=3.0,
        #      batch_size=50, regul_param=1.0, early_stop=1),
    ]
    """
    - the best results achieved with:
      inercia=0.01, subset=5000, learning_rate=3.0, hidden_layers=100
      inercia=0.01, subset=5000, learning_rate=1.0, hidden_layers=30
    - however learning_rate=1.0, hidden_layers=30 is faster
    - inercia doesn't affect accuracy, only the runtime
    """
    def make_nn(tr_d, **hparams):
        hparams = hparams.copy()
        print(f'Training the NN with hyperparams {hparams}')

        tr_x, tr_y = tr_d
        pixels_in_picture = tr_x.get_value().shape[1]
        output_classes = max(tr_y.eval()) + 1
        hidden_layers = list(hparams.get('hidden_layers'))
        return NeuralNetwork([
            FullyConnectedLayer(n_in=pixels_in_picture, n_out=hidden_layers[0]),
            FullyConnectedLayer(n_in=hidden_layers[0], n_out=output_classes)
        ], hparams)

    def train_nn(nn, tr_d, va_d):
        return nn.learn(tr_d, va_d)

    def eval_nn(nn, te_d):
        return nn.evaluate(te_d)

    def _load_data(hparams):
        (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = load_data()
        if 'subset' in hparams:
            subset_to = hparams.get('subset')
            tr_x = tr_x[:subset_to]
            tr_y = tr_y[:subset_to]
        tr_x = shared(tr_x)
        tr_y = shared(tr_y)
        va_x = shared(va_x)
        va_y = shared(va_y)
        te_x = shared(te_x)
        te_y = shared(te_y)
        # num_y_classes = max(tr_y) + 1
        tr_y = T.cast(tr_y, "int32")
        va_y = T.cast(va_y, "int32")
        te_y = T.cast(te_y, "int32")
        # tr_y = digits_to_binary_arrays(tr_y, num_y_classes)
        # va_y = digits_to_binary_arrays(va_y, num_y_classes)
        # te_y = digits_to_binary_arrays(te_y, num_y_classes)
        return (tr_x, tr_y), (va_x, va_y), (te_x, te_y)

    run_benchmarks(make_nn, train_nn, eval_nn,
                   _load_data, all_hparams)


class Layer:
    def __init__(self):
        self.n_in = None
        self.n_out = None
        self.activation_fn = None

        self.params = []
        self.w = None
        self.b = None

        self.input = None
        self.output = None

    def set_input(self, inpt, mini_batch_size):
        pass

    def predict(self):
        pass

    def accuracy(self, y):
        pass


class FullyConnectedLayer(Layer):
    def __init__(self, n_in, n_out, activation_fn=sigmoid):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(np.random.normal(
                loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX
            ), name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(
                loc=0.0, scale=1.0, size=(n_out,)),
                dtype=theano.config.floatX
            ), name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_input(self, inpt, batch_size):
        self.input = inpt.reshape((batch_size, self.n_in))
        self.output = self.activation_fn(T.dot(self.input, self.w) + self.b)

    def predict(self):
        return binary_arrays_to_digits(self.output)

    def accuracy(self, y):
        y_pred_flat = binary_arrays_to_digits(self.output)
        # y_test_flat = binary_arrays_to_digits(y)
        return T.mean(T.eq(y_pred_flat, y))


class NeuralNetwork:
    def __init__(self, layers: List[Layer], hparams: Dict):
        """
        Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.
        """
        self.hparams = dict(
            epochs=hparams.get('epochs', 100),
            learning_rate=hparams.get('learning_rate', 3.),
            batch_size=hparams.get('batch_size', 10),
            regul_param=hparams.get('regul_param', .01),
            early_stop=hparams.get('early_stop', 10),
            inercia=hparams.get('inercia', .01),
        )

        self.layers = layers
        self.params = []
        for layer in layers:
            self.params.extend(layer.params)
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        init_layer = layers[0]
        init_layer.set_input(self.x, self.hparams.get('batch_size'))
        for prev_layer, layer in zip(layers[:-1], layers[1:]):
            layer.set_input(prev_layer.output, self.hparams.get('batch_size'))
        self.output = self.layers[-1].output

    def predict(self, x_data):
        self.make_predict_fn(x_data)
        return self.layers[-1].predict()

    def evaluate(self, test_data, batch_size=None):
        test_x, test_y = test_data
        batch_size = batch_size or self.hparams.get('batch_size')
        num_batches = math.ceil(test_x.get_value().shape[0] / batch_size)
        accuracy_fn = self.make_accuracy_fn(test_x, test_y)
        return np.mean([accuracy_fn(j) for j in range(num_batches)])

    def make_predict_fn(self, data_x, batch_size=None):
        batch_size = batch_size or self.hparams.get('batch_size')
        i = T.lscalar()  # batch index
        return theano.function(
            [i], self.layers[-1].predict(),
            givens={
                self.x: data_x[i * batch_size: (i+1) * batch_size]
            })

    def make_accuracy_fn(self, data_x, data_y, batch_size=None):
        batch_size = batch_size or self.hparams.get('batch_size')
        i = T.lscalar()  # batch index
        return theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: data_x[i * batch_size: (i+1) * batch_size],
                self.y: data_y[i * batch_size: (i+1) * batch_size]
            })

    def make_train_fn(self, data_x, data_y,
                      learning_rate, regularisation, batch_size):
        regul_param = 0.5 * (regularisation / batch_size) * \
                      sum((l.w ** 2).sum() for l in self.layers)
        # note on syntax: T.arange(y.shape[0]) is a vector of integers [0,1,2,...,len(y)].
        # Indexing a matrix M by the two vectors [0,1,...,K], [a,b,...,k] returns the
        # elements M[0,a], M[1,b], ..., M[K,k] as a vector.  Here, we use this
        # syntax to retrieve the log-probability of the correct labels, y.
        log = T.log(self.layers[-1].output)
        ind = T.arange(self.y.shape[0])
        ind = T.cast(ind, 'int32')
        log_ind = log[ind, self.y]
        loss = -T.mean(log_ind)
        cost = loss + regul_param
        grads = T.grad(cost, self.params)

        updates = [(param, param - learning_rate * grad)
                   for param, grad in zip(self.params, grads)]

        i = T.lscalar()  # batch index
        return theano.function(
            [i], cost, updates=updates,
            givens={
                self.x: data_x[i * batch_size: (i+1) * batch_size],
                self.y: data_y[i * batch_size: (i+1) * batch_size]
            })

    # def make_validate_accuracy_fn(self, data_x, data_y, batch_size=None):
    #     batch_size = batch_size or self.batch_size
    #     i = T.lscalar()  # batch index
    #     return theano.function(
    #         [i], self.accuracy(self.y),
    #         givens={
    #             self.x: data_x[i * batch_size: (i+1) * batch_size],
    #             self.y: data_y[i * batch_size: (i+1) * batch_size]
    #         })

    def learn(self, training_data, validation_data,
              epochs=None,
              batch_size=None,
              learning_rate=None,
              regularisation=None,
              inercia=None,
              early_stop=None,
              monitor_frequency=None):
        """ Train the network using mini-batch stochastic gradient descent
        """
        epochs = epochs or self.hparams.get('epochs')
        batch_size = batch_size or self.hparams.get('batch_size')
        learning_rate = learning_rate or self.hparams.get('learning_rate')
        regularisation = regularisation or self.hparams.get('regul_param')
        inercia = inercia or self.hparams.get('inercia')
        early_stop = early_stop or self.hparams.get('early_stop')

        training_x, training_y = training_data
        validation_x, validation_y = validation_data

        # Partition into smaller batches
        n_training_batches = math.ceil(training_x.get_value().shape[0] / batch_size)
        num_validation_batches = math.ceil(validation_x.get_value().shape[0] / batch_size)

        train_fn = self.make_train_fn(training_x, training_y,
            learning_rate, regularisation, batch_size)
        accuracy_validation_fn = self.make_accuracy_fn(validation_x, validation_y, batch_size)

        # Do the actual training
        best_validation_accuracy = 0.0
        best_iteration = 0
        accuracies = []
        costs = []
        ini_learning_rate = learning_rate
        early_stopped = False
        for epoch_i in range(epochs):
            for batch_i in range(n_training_batches):
                if early_stopped:
                    break
                n_batches_trained = n_training_batches * epoch_i + batch_i
                monitor_frequency = monitor_frequency or n_training_batches
                if n_batches_trained % monitor_frequency == 0:
                    print(f'Epoch {epoch_i + 1}. '
                          f'Training mini-batch number {n_batches_trained}', end='')

                current_cost = train_fn(batch_i)

                if n_batches_trained % monitor_frequency == 0:
                    if n_batches_trained == 0:
                        print()
                    else:
                        # validate
                        val_acc = np.mean([accuracy_validation_fn(j) for j in
                                           range(num_validation_batches)])
                        print(f", current cost: {current_cost}", end='')
                        print(f", validation accuracy: {val_acc:.2%}", end='')
                        print()
                        costs.append(current_cost)
                        accuracies.append(val_acc)
                        # if val_acc >= best_validation_accuracy:
                        #     print('  this is the best validation accuracy to date.')
                        #     best_validation_accuracy = val_acc
                        #     best_iteration = n_batches_trained

                        if early_stop and \
                                len(costs) > early_stop and \
                                all(val_acc < a for a in accuracies[-early_stop - 1: -1]):
                            learning_rate /= 2
                            print(f'Decreasing learning rate to {learning_rate}')
                            if learning_rate < ini_learning_rate / 128:
                                print(f'Learning rate dropped down to '
                                      f'{learning_rate}<{ini_learning_rate}/128,'
                                      f'stopping at iteration {n_batches_trained}')
                                break

        print('Finished training network.')
        # print(f'Best validation accuracy of {best_validation_accuracy:.2%} '
        #       f'obtained at iteration {best_iteration}')
        return accuracies, costs


def cross_entropy_cost(a, y):
    """ Return the cost associated with an output ``a`` and desired output ``y``
    """
    return -T.sum(y * T.log(a) + (1 - y) * T.log(1 - a))


def data_size(data):
    """ Return the size of the dataset `data`
    """
    return data[0].get_value(borrow=True).shape[0]


def binary_arrays_to_digits(y_pred):
    # convert 10-long arrays of 0/1 into a flat shape (arrays of digits 0-9)
    return T.argmax(y_pred, axis=1)


def shared(data):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    return theano.shared(np.asarray(data, dtype=theano.config.floatX))


if __name__ == '__main__':
    main()
