import itertools
import math
import random
import numpy as np


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
        return np.argmax(A, axis=0)  # convert to the flat shape (array of digits 0-9)

    def predict(self, X):
        activations, _ = self.feedforward(X)
        return self._predict(activations[-1])

    def learn(self, train_X, train_Y, epochs, learning_rate, batch_max_size,
              test_X=None, test_Y=None, print_cost=False):

        # Shuffle training data
        np.random.seed(1)
        state = np.random.get_state()
        np.random.shuffle(train_X)
        np.random.set_state(state)
        np.random.shuffle(train_Y)

        # Partition into smaller batches
        m = train_X.shape[1]
        n_batches = math.ceil(m / batch_max_size)
        batch_size = math.ceil(m / n_batches)
        X_batches = [train_X[:, k*batch_size:(k+1)*batch_size] for k in range(n_batches)]
        Y_batches = [train_Y[:, k*batch_size:(k+1)*batch_size] for k in range(n_batches)]

        for epoch in range(epochs):

            output_activations = []
            for X, Y in zip(X_batches, Y_batches):
                activations, zs = self.feedforward(X)
                self.backprop(activations, zs, Y, learning_rate)
                output_activations.append(activations[-1])

            if epoch % (epochs//10) == 0 or epoch == epochs-1:
                print(f"Epoch {epoch+1}", end='')

                if print_cost:
                    output_activation = np.concatenate(output_activations, axis=1)
                    cost = calc_cost(output_activation, train_Y)
                    print(f", cost: {cost}", end='')

                if test_X is not None:
                    pred_Y = self.predict(test_X)
                    acc = evaluate(pred_Y, test_Y)
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


def evaluate(pred_Y, test_Y):
    """ assume pred_Y and test_Y are 1-dim arrays of integers 0 through 9
    """
    assert pred_Y.shape == test_Y.shape, (pred_Y.shape, test_Y.shape)
    return sum(int(p == t) for p, t in zip(pred_Y, test_Y)) / len(pred_Y)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def calc_cost(A, train_Y):
    """ assume test_Y and pred_Y are 1-dim arrays of 9 integers 0 or 1
    """
    assert A.shape == train_Y.shape, (A.shape, train_Y.shape)  # (m, 1)
    return np.mean(np.mean((train_Y - A) ** 2, axis=0))

