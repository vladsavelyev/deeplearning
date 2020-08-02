import itertools
import json
import math
import sys
from typing import List, Dict

import numpy as np
import theano
import theano.tensor as T

from digits.run_benchmarks import run_benchmarks

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid


def main():
    all_hparams = [
        # dict(inercia=0.01, subset=5000, hidden_layers=(30,),  epochs=100,
        #      learning_rate=3.0, batch_size=10, regul_param=1.0, early_stop=1),
        dict(inercia=0.01, subset=5000, hidden_layers=(100,), epochs=100,
             learning_rate=3.0, batch_size=10, regul_param=1.0, early_stop=1),
        dict(inercia=0.01, subset=5000, hidden_layers=(30,),  epochs=100,
             learning_rate=1.0, batch_size=10, regul_param=1.0, early_stop=1),
        # dict(inercia=0.01, subset=1000, hidden_layers=(30,),  epochs=100,
        #      learning_rate=3.0, batch_size=10, regul_param=1.0, early_stop=1),
        # dict(inercia=0,    subset=5000, hidden_layers=(30,),  epochs=100,
        #      learning_rate=3.0, batch_size=10, regul_param=1.0, early_stop=1),
    ]
    """
    - the best results achieved with:
      inercia=0.01, subset=5000, learning_rate=3.0, hidden_layers=100
      inercia=0.01, subset=5000, learning_rate=1.0, hidden_layers=30
    - however learning_rate=1.0, hidden_layers=30 is faster
    - inercia doesn't affect accuracy, only the runtime
    """
    def make_nn(training_data, **hparams):
        hparams = hparams.copy()
        print(f'Training the NN with hyperparams {hparams}')

        training_x, training_y = training_data
        pixels_in_picture = training_x.get_value().shape[1]
        output_classes = training_y.eval().shape[1]
        return NeuralNetwork([
            FullyConnectedLayer(n_in=pixels_in_picture, n_out=100),
            FullyConnectedLayer(n_in=100, n_out=output_classes)
        ], hparams)

    run_benchmarks(make_nn, all_hparams)


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
        y_test_flat = binary_arrays_to_digits(y)
        return T.mean(T.eq(y_pred_flat, y_test_flat))


class NeuralNetwork:
    def __init__(self, layers: List[Layer], hparams: Dict):
        """
        Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.
        """
        self.epochs = hparams.get('epochs', 100)
        self.learning_rate = hparams.get('learning_rate', 3.0)
        self.batch_size = hparams.get('batch_size', 10)
        self.regul_param = hparams.get('regul_param', 0.01)
        self.early_stop = hparams.get('early_stop', 10)
        self.inercia = hparams.get('inercia', 1.0)

        self.layers = layers
        self.params = []
        for layer in layers:
            self.params.extend(layer.params)
        self.x = T.matrix('x')
        self.y = T.matrix('y')
        init_layer = layers[0]
        init_layer.set_input(self.x, self.batch_size)
        for prev_layer, layer in zip(layers[:-1], layers[1:]):
            layer.set_input(prev_layer.output, self.batch_size)
        self.output = self.layers[-1].output

    def predict(self, x_data):
        self.make_predict_fn(x_data)
        return self.layers[-1].predict()

    def accuracy(self, test_x, test_y, batch_size=None):
        batch_size = batch_size or self.batch_size
        num_batches = math.ceil(test_x.get_value().shape[0] / batch_size)
        accuracy_fn = self.make_accuracy_fn(test_x, test_y)
        return np.mean([accuracy_fn(j) for j in range(num_batches)])

    def make_predict_fn(self, data_x, batch_size=None):
        batch_size = batch_size or self.batch_size
        i = T.lscalar()  # batch index
        return theano.function(
            [i], self.layers[-1].predict(),
            givens={
                self.x: data_x[i * batch_size: (i+1) * batch_size]
            })

    def make_accuracy_fn(self, data_x, data_y, batch_size=None):
        batch_size = batch_size or self.batch_size
        i = T.lscalar()  # batch index
        return theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: data_x[i * self.batch_size: (i+1) * batch_size],
                self.y: data_y[i * self.batch_size: (i+1) * batch_size]
            })

    def make_train_fn(self, data_x, data_y,
                      learning_rate, regularisation, batch_size, num_batches):
        regul_param = 0.5 * regularisation * \
                      sum((l.w ** 2).sum() for l in self.layers) / \
                      num_batches
        cost = cross_entropy_cost(self.layers[-1].output, self.y) + regul_param
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

    def learn(self, training_data, validation_data, test_data,
              epochs=None,
              batch_size=None,
              learning_rate=None,
              regularisation=None,
              inercia=None,
              early_stop=None):
        """ Train the network using mini-batch stochastic gradient descent
        """
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size
        learning_rate = learning_rate or self.learning_rate
        regularisation = regularisation or self.regul_param
        inercia = inercia or self.inercia
        early_stop = early_stop or self.early_stop

        if validation_data is None and self.early_stop:
            sys.stderr.write('When early_stop is set, '
                             'validation data must be provided\n')
            sys.exit(1)

        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = training_data

        # Partition into smaller batches
        num_training_batches = math.ceil(training_x.get_value().shape[0] / batch_size)
        num_validation_batches = math.ceil(validation_x.get_value().shape[0] / batch_size)
        num_test_batches = math.ceil(test_x.get_value().shape[0] / batch_size)

        train_fn = self.make_train_fn(training_x, training_y,
            learning_rate, regularisation, batch_size, num_training_batches)

        accuracy_validation_fn = self.make_accuracy_fn(validation_x, validation_y, batch_size)

        # Do the actual training
        best_validation_accuracy = 0.0
        corresponding_test_accuracy = None
        best_iteration = 0
        accuracies = []
        costs = []
        ini_learning_rate = learning_rate
        for epoch_i in range(epochs):
            for batch_i in range(num_training_batches):
                iteration = num_training_batches * epoch_i + batch_i
                if iteration % 1000 == 0:
                    print(f'Epoch {epoch_i} Training mini-batch number {iteration}')
                current_cost = train_fn(batch_i)
                if iteration % 1000 == 0:
                    costs.append(current_cost)
                    val_acc = np.mean([accuracy_validation_fn(j) for j in
                                       range(num_validation_batches)])
                    accuracies.append(val_acc)
                    print(f'  current validation accuracy is {val_acc:.2%}')
                    if val_acc >= best_validation_accuracy:
                        print('  this is the best validation accuracy to date.')
                        best_validation_accuracy = val_acc
                        best_iteration = iteration
                        if test_data:
                            corresponding_test_accuracy = self.accuracy(test_x, test_y)
                            print(f'The corresponding test accuracy is '
                                  f'{corresponding_test_accuracy:.2%}')

                    if early_stop and \
                            len(costs) > early_stop and \
                            all(val_acc < a for a in accuracies[-early_stop - 1: -1]):
                        learning_rate /= 2
                        print(f'Decreasing learning rate to {learning_rate}')
                        if learning_rate < ini_learning_rate / 128:
                            print(f'Learning rate dropped down to '
                                  f'{learning_rate}<{ini_learning_rate}/128,'
                                  f'stopping at iteration {iteration}')
                            break

        print('Finished training network.')
        print(f'Best validation accuracy of {best_validation_accuracy:.2%} '
              f'obtained at iteration {best_iteration}')
        print(f'Corresponding test accuracy of {corresponding_test_accuracy:.2%}')
        return accuracies, costs


def cross_entropy_cost(a, y):
    """ Return the cost associated with an output ``a`` and desired output ``y``
    """
    return -T.mean(y * T.log(a) + (1 - y) * T.log(1 - a))


def data_size(data):
    """ Return the size of the dataset `data`
    """
    return data[0].get_value(borrow=True).shape[0]


def binary_arrays_to_digits(y_pred):
    # convert 10-long arrays of 0/1 into a flat shape (arrays of digits 0-9)
    return T.argmax(y_pred, axis=1)


if __name__ == '__main__':
    main()
