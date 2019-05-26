import itertools
import json
import math
import random
import numpy as np
import datetime

from digits.load_data import get_X, get_Y, evaluate, collapse_digits


class NeuralNetwork:

    @staticmethod
    def run(train_data, valid_data, **hparams):
        hparams = hparams.copy()
        print(f'Training the NN with hyperparams {hparams}')
        train_data = train_data[:hparams.pop('subset')]
        layers = [get_X(train_data).shape[0]]
        if 'hidden_layer' in hparams:
            layers.append(hparams.pop('hidden_layer'))
        layers.append(get_Y(train_data).shape[0])
        nn = NeuralNetwork(layers)
        return nn, nn.learn(np.copy(train_data), valid_data=valid_data, **hparams)

    def __init__(self, layer_sizes):
        np.random.seed(2)
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(k, j) / np.sqrt(j)
                        for j, k in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.momentums = [np.ones(w.shape) for w in self.weights]
        self.biases = [np.random.randn(k, 1)
                       for k in layer_sizes[1:]]

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

    def learn(self, train_data, valid_data=None, epochs=100, learning_rate=0.01, batch_max_size=10,
              regul_param=0.01, early_stop=10, inercia=1.0):

        # Partition into smaller batches
        n = train_data.shape[0]
        n_batches = math.ceil(n / batch_max_size)
        batch_size = math.ceil(n / n_batches)

        accs = []
        costs = []
        ini_learning_rate = learning_rate
        np.random.seed(3)
        for epoch in range(epochs):
            np.random.shuffle(train_data)
            batches = [train_data[bn*batch_size:(bn+1)*batch_size] for bn in range(n_batches)]

            output_activations = []
            for batch in batches:
                batch_X = get_X(batch)
                batch_Y = get_Y(batch)
                activations, zs = self.feedforward(batch_X)
                self.backprop(n, activations, zs, batch_Y, learning_rate, regul_param=regul_param, inercia=inercia)
                output_activations.append(activations[-1])

            acc, cost = self.monitor(epoch + 1, train_data, regul_param, valid_data=valid_data)
            costs.append(cost)
            accs.append(acc)
            if early_stop and \
                    len(accs) > early_stop and \
                    all(acc < a for a in accs[-early_stop-1:-1]):
                learning_rate /= 2
                print(f'Decreasing learning rate to {learning_rate}')
                if learning_rate < ini_learning_rate/128:
                    print(f'Learning rate dropped down to {learning_rate}<{ini_learning_rate}/128,'
                          f'stopping at epoch {epoch}')
                    break

        return accs, costs

    def monitor(self, epoch_n, train_data, regul_param, valid_data=None):
        print(f"Epoch {epoch_n}", end='')
        cost = None
        acc = None

        activations, zs = self.feedforward(get_X(train_data))
        y = get_Y(train_data)
        cost = regularized_cost(activations[-1], y, self.weights, regul_param=regul_param)
        print(f", cost: {cost}", end='')

        if valid_data is not None:
            val_X = get_X(valid_data)
            val_Y = get_Y(valid_data)
            Y_classes = get_Y(train_data).shape[0]
            predictions = self.predict(val_X)
            acc = evaluate(predictions, val_Y, Y_classes)
            print(f", accuracy: {acc}", end='')
        print()
        return acc, cost

    def backprop(self, n, activations, zs, train_Y, learning_rate, regul_param, inercia):
        """
        n is the size of full training set

        cost = MSR(Ak)  Ak = s(Zk)          Zk = Wkj x Aj + Bk           ->
        dAk  = Y - Ak   dZk = dAk * s'(Zk)  dWkj = dZk.T x Aj, dBk = dZk, dAj = dZk x Wkj
        """
        m = train_Y.shape[1]
        # m is the size of mini-batch training set

        # collecting dW and dB
        dW = [0] * (len(self.layer_sizes) - 1)
        dB = [0] * (len(self.layer_sizes) - 1)

        dZk = activations[-1] - train_Y  # cross-entropy cost
        dAk = None
        for k, Zk, Ak, Aj in reversed(list(zip(itertools.count(), zs, activations[1:], activations[:-1]))):
            if dZk is None:
                dZk = dAk * sigmoid_prime(Zk)

            dW[k] = np.dot(dZk, Aj.T) / m   # (k_layer_size, m) x (j_layer_size, m).T -> (k_ls, j_ls)
            dB[k] = np.sum(dZk, axis=1, keepdims=1) / m  # (k_layer_size, 1)

            dAj = np.dot(self.weights[k].T, dZk)  # (j_ls, k_ls) x (k_ls, m) -> (j_ls, m)
            # moving to the next layer
            dAk = dAj
            dZk = None

        # updating weights and b iases
        for k, Mk, Wk, Bk, dWk, dBk, in reversed(list(zip(
                itertools.count(), self.momentums, self.weights, self.biases, dW, dB))):
            # for Vk, Wk, Bk, dWk, dBk in zip(self.velocities, self.weights, self.biases, dW, dB):
            #     self.velocities[k] = friction * self.velocities[k] - learning_rate * dW[k]
            #     self.weights[k] += self.velocities[k]
            #     self.weights[k] -= learning_rate * (regul_param / n * self.weights[k])  # regularization
            #     Vk = friction * Vk - learning_rate * dWk
            Mk = Mk * inercia - learning_rate * dWk  # calculating new momentum at the moment
            Wk += Mk  # updating the weight with the momentum
            Wk -= learning_rate * (regul_param / n * self.weights[k])  # applying regularization
            Bk -= learning_rate * dBk  # updating the biases - simply, without momentum or regularization
            self.momentums[k] = Mk
            self.weights[k] = Wk
            self.biases[k] = Bk

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def cross_entropy_cost(a, y):
    """
    Return the cost associated with an output ``a`` and desired output
    ``y``.  Note that np.nan_to_num is used to ensure numerical
    stability.  In particular, if both ``a`` and ``y`` have a 1.0
    in the same slot, then the expression (1-y)*np.log(1-a)
    returns nan.  The np.nan_to_num ensures that that is converted
    to the correct value (0.0).
    """
    return -np.sum(np.nan_to_num(y * np.log(a) + (1 - y) * np.log(1 - a)))


def regularized_cost(a, y, weights, regul_param):
    # a.shape = (m, 1)
    cost = cross_entropy_cost(a, y)
    n = a.shape[0]
    regularization = 0.5 * (regul_param / n) * sum(np.linalg.norm(w) ** 2 for w in weights)
    return cost + regularization


def save_nn(nn: NeuralNetwork, fpath):
    """ Save the neural network to the file `fpath`
    """
    data = {"sizes": nn.layer_sizes,
            "weights": [w.tolist() for w in nn.weights],
            "biases": [b.tolist() for b in nn.biases],
            }
    with open(fpath, 'w') as f:
        json.dump(data, f)


def load_nn(fpath):
    """ Load a neural network from the file `fpath`.
        Returns an instance of NeuralNetwork.
    """
    with open(fpath) as f:
        data = json.load(f)

    net = NeuralNetwork(data['sizes'])
    net.weights = [np.array(w) for w in data['weights']]
    net.biases = [np.array(b) for b in data['biases']]
    return net

