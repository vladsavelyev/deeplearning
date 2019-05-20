import itertools
import math
import random
import numpy as np
import datetime

from digits.load_data import get_X, get_Y, evaluate, collapse_digits, expand_digits


class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(k, j)
                        for k, j in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.biases = [np.random.randn(l, 1)
                       for l in layer_sizes[1:]]

    def feedforward(self, X):
        A = X  # activation of the zero'th layer = input layer, shape = (i'th, m)
        assert A.shape[0] == self.layer_sizes[0], (A.shape, self.layer_sizes)
        zs = []
        activations = [X]
        for W, B in zip(self.weights, self.biases):
            # print(W.shape, A.shape, B.shape)
            # assert A.shape == (m, self.layer_sizes[i-1]), (A.shape[0], self.layer_sizes[i-1])
            # assert W.shape == (self.layer_sizes[i-1], self.layer_sizes[i])
            # assert B.shape == (1, self.layer_sizes[i])
            Z = np.dot(W, A) + B
            # assert Z.shape == (m, self.layer_sizes[i])
            A = sigmoid(Z)
            zs.append(Z)
            activations.append(A)
        return activations, zs

    @staticmethod
    def _predict(A):
        return collapse_digits(A)

    def predict(self, X):
        activations, _ = self.feedforward(X)
        return self._predict(activations[-1])

    @staticmethod
    def shuffle(X, Y):
        # Shuffle training data
        state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(state)
        np.random.shuffle(Y)

    def learn(self, train_data, epochs, learning_rate, batch_max_size,
              test_data=None, print_cost_every=None):

        # Partition into smaller batches
        m = train_data.shape[0]
        n_batches = math.ceil(m / batch_max_size)
        batch_size = math.ceil(m / n_batches)

        test_X = get_X(test_data)
        test_Y = get_Y(test_data)
        Y_classes = get_Y(train_data).shape[0]

        for epoch in range(epochs):
            np.random.shuffle(train_data)
            batches = [train_data[k*batch_size:(k+1)*batch_size] for k in range(n_batches)]

            output_activations = []
            for batch in batches:
                batch_X = get_X(batch)
                batch_Y = get_Y(batch)
                activations, zs = self.feedforward(batch_X)
                self.backprop(activations, zs, batch_Y, learning_rate)
                output_activations.append(activations[-1])

            # if epoch % (epochs//10) == 0 or epoch == epochs-1:
            if print_cost_every and (epoch % print_cost_every == 0 or epoch == epochs-1):
                print(f"Epoch {epoch+1}", end='')

                output_activation = np.concatenate(output_activations, axis=1)
                cost = calc_cost(output_activation, get_Y(train_data))
                print(f", cost: {cost}", end='')

                if test_data is not None:
                    predictions = self.predict(test_X)
                    acc = evaluate(predictions, test_Y, Y_classes)
                    print(f", accuracy: {acc}", end='')
                print()

    def backprop(self, activations, zs, train_Y, learning_rate):
        """
        cost = MSR(Ak)  Ak = s(Zk)          Zk = Wkj x Aj + Bk           ->
        dAk  = Y - Ak   dZk = dAk * s'(Zk)  dWkj = dZk.T x Aj, dBk = dZk, dAj = dZk x Wkj
        """
        m = train_Y.shape[1]
        dAk = activations[-1] - train_Y
        for k, Zk, Ak, Aj in reversed(list(zip(itertools.count(), zs, activations[1:], activations[:-1]))):
            dZk = dAk * sigmoid_prime(Zk)
            dWkj = np.dot(dZk, Aj.T) / m   # (k_layer_size, m) x (j_layer_size, m).T -> (k_ls, j_ls)
            dBk = np.sum(dZk, axis=1, keepdims=1) / m  # (k_layer_size, 1)
            dAj = np.dot(self.weights[k].T, dZk)  # (j_ls, k_ls) x (k_ls, m) -> (j_ls, m)
            # print(dWkj.shape, self.weights[k].shape)
            self.weights[k] -= learning_rate * dWkj
            # print(dBk.shape, self.biases[k].shape)
            self.biases[k] -= learning_rate * dBk
            dAk = dAj


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def calc_cost(A, train_Y):
    """ assume test_Y and pred_Y are 1-dim arrays of 9 integers 0 or 1
    """
    assert A.shape == train_Y.shape, (A.shape, train_Y.shape)  # (m, 1)
    m = A.shape[0]
    return np.sum(np.abs(train_Y - A)) / (2 * m)

