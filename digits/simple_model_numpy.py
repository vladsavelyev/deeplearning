import itertools
import json
import math
import sys
import numpy as np

from digits import NeuralNetwork
from digits.mnist_loader import binary_arrays_to_digits, digits_to_binary_arrays, load_data_wrapper, get_x, get_y
from digits.run_benchmarks import run_benchmarks


def main():
    all_hparams = [
        # dict(inercia=0.01, subset=5000, hidden_layers=(30,),  epochs=100, learning_rate=3.0,
        #      batch_max_size=10, regul_param=1.0, early_stop=1),
        dict(inercia=0.01, subset=5000, hidden_layers=(100,), epochs=100, learning_rate=3.0,
             batch_max_size=10, regul_param=1.0, early_stop=1),
        dict(inercia=0.01, subset=5000, hidden_layers=(30,),  epochs=100, learning_rate=1.0,
             batch_max_size=10, regul_param=1.0, early_stop=1),
        # dict(inercia=0.01, subset=1000, hidden_layers=(30,),  epochs=100, learning_rate=3.0,
        #      batch_max_size=10, regul_param=1.0, early_stop=1),
        # dict(inercia=0,    subset=5000, hidden_layers=(30,),  epochs=100, learning_rate=3.0,
        #      batch_max_size=10, regul_param=1.0, early_stop=1),
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
        if 'subset' in hparams:
            subset_to = hparams.pop('subset')
            tr_d = tr_d[:subset_to]

        layers = [get_x(tr_d).shape[0]]
        if 'hidden_layers' in hparams:
            layers.extend(hparams.pop('hidden_layers'))
        layers.append(get_y(tr_d).shape[0])

        return SimpleNeuralNetwork(layers, hparams)

    def train_nn(nn, tr_d, va_d):
        return nn.learn(tr_d, va_d)

    def eval_nn(nn, te_d):
        return nn.evaluate(te_d)

    run_benchmarks(make_nn, train_nn, eval_nn,
                   load_data_wrapper, all_hparams)


class SimpleNeuralNetwork(NeuralNetwork):
    def __init__(self, layer_sizes, hparams):
        super().__init__(hparams)
        self.hparams = dict(
            epochs=hparams.get('epochs', 100),
            learning_rate=hparams.get('learning_rate', 3.0),
            batch_size=hparams.get('batch_size', 50),
            regul_param=hparams.get('regul_param', 0.01),
            early_stop=hparams.get('early_stop', 10),
            inercia=hparams.get('inercia', 1.0),
        )
        self.layer_sizes = layer_sizes
        np.random.seed(2)
        self.biases = [np.random.randn(k, 1)
                       for k in layer_sizes[1:]]
        self.weights = [np.random.randn(k, j) / np.sqrt(j)
                        for j, k in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.momentums = [np.ones(w.shape) for w in self.weights]

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
    def _activations_to_y(A):
        return binary_arrays_to_digits(A)

    def predict(self, X):
        activations, _ = self.feedforward(X)
        return self._activations_to_y(activations[-1])

    def learn(self, training_data, validation_data=None,
              epochs=None,
              batch_size=None,
              learning_rate=None,
              regularisation=None,
              inercia=None,
              early_stop=None,
              monitor_frequency=None):
        epochs = epochs or self.hparams.get('epochs')
        batch_size = batch_size or self.hparams.get('batch_size')
        learning_rate = learning_rate or self.hparams.get('learning_rate')
        regularisation = regularisation or self.hparams.get('regul_param')
        inercia = inercia or self.hparams.get('inercia')
        early_stop = early_stop or self.hparams.get('early_stop')

        if validation_data is None and early_stop:
            sys.stderr.write('When early_stop is set, validation data must be provided\n')
            sys.exit(1)

        # Partition into smaller batches
        n = len(training_data)
        n_batches = math.ceil(n / batch_size)
        batch_size = math.ceil(n / n_batches)

        # best_validation_accuracy = 0.0
        # best_iteration = 0
        accuracies = []
        costs = []
        ini_learning_rate = learning_rate
        early_stopped = False
        np.random.seed(3)
        for epoch_i in range(epochs):
            # np.random.shuffle(training_data)
            for batch_i in range(n_batches):
                if early_stopped:
                    break
                n_batches_trained = n_batches * epoch_i + batch_i
                monitor_frequency = monitor_frequency or n_batches
                if n_batches_trained % monitor_frequency == 0:
                    print(f'Epoch {epoch_i + 1}. '
                          f'Training mini-batch number {n_batches_trained}', end='')

                batch = training_data[batch_i * batch_size: (batch_i + 1) * batch_size]
                batch_x = get_x(batch)
                batch_y = get_y(batch)

                activations, zs = self.feedforward(batch_x)
                # calculating the weights and biases changes by doing the backprop
                self.backprop(n, activations, zs, batch_y,
                    learning_rate, regul_param=regularisation, inercia=inercia)

                if n_batches_trained % monitor_frequency == 0:
                    if n_batches_trained == 0:
                        print()
                    else:
                        activations, zs = self.feedforward(batch_x)
                        current_cost = regularized_cost(
                            activations[-1], batch_y,
                            self.weights, regularisation=regularisation)
                        validation_accuracy = self.evaluate(validation_data)
                        print(f", current cost: {current_cost}", end='')
                        print(f", validation accuracy: {validation_accuracy}", end='')
                        print()

                        costs.append(current_cost)
                        accuracies.append(validation_accuracy)
                        if early_stop and \
                                len(costs) > early_stop and \
                                all(validation_accuracy < a for a in accuracies[-early_stop-1:-1]):
                            learning_rate /= 2
                            print(f'Decreasing learning rate to {learning_rate}')
                            if learning_rate < ini_learning_rate/128:
                                print(f'Learning rate dropped down to '
                                      f'{learning_rate}<{ini_learning_rate}/128,'
                                      f'stopping at epoch {epoch_i + 1}')
                                early_stopped = True


        print('Finished training network.')
        # print(f'Best validation accuracy of {best_validation_accuracy:.2%} '
        #       f'obtained at iteration {best_iteration}')
        return accuracies, costs

    def evaluate(self, test_data):
        x = get_x(test_data)
        y_flat = get_y(test_data)
        predictions = self.predict(x)
        return self.accuracy(predictions, y_flat)

    def accuracy(self, pred_y_flat, test_y_flat):
        """ pred_y      is a [n x 10] array of 0 and 1
            test_y_flat is a [n] array of digits 0 through 9
        """
        if len(test_y_flat.shape) > 1:
            test_y_flat = binary_arrays_to_digits(test_y_flat)
        assert pred_y_flat.shape == test_y_flat.shape, (
            pred_y_flat.shape, test_y_flat.shape)
        return sum(int(p == t) for p, t in zip(pred_y_flat, test_y_flat)) / len(pred_y_flat)

    def backprop(self, n, activations, zs, train_y, learning_rate,
                 regul_param, inercia):
        """
        n is the size of full training set

        cost = MSR(Ak)
        Ak = s(Zk)
        Zk = Wkj x Aj + Bk
        dAk = Y - Ak  # cross-entropy cost derivative
        dZk = dAk * s'(Zk)
        dWkj = dZk.T x Aj
        dBk = dZk
        dAj = dZk x Wkj
        """
        m = train_y.shape[1]
        # m is the size of mini-batch training set

        # Collecting dW and dB
        dW = [0] * (len(self.layer_sizes) - 1)
        dB = [0] * (len(self.layer_sizes) - 1)
        # iterating over the layers, where the layers are indexed by j and k.
        # the derivative is a bit differrent for the last layer, so calculating it outside of the loop
        dAk = None
        dZk = activations[-1] - train_y  # cross-entropy cost derivative
        for k, Zk, Ak, Aj in reversed(list(zip(
                itertools.count(), zs, activations[1:], activations[:-1]
        ))):
            if dAk is not None:  # not the first loop
                dZk = dAk * sigmoid_prime(Zk)

            dW[k] = np.dot(dZk, Aj.T) / m   # (k_layer_size, m) x (j_layer_size, m).T -> (k_ls, j_ls)
            dB[k] = np.sum(dZk, axis=1, keepdims=1) / m  # (k_layer_size, 1)

            dAj = np.dot(self.weights[k].T, dZk)  # (j_ls, k_ls) x (k_ls, m) -> (j_ls, m)
            # moving to the next layer
            dAk = dAj

        # Updating weights and biases
        for k, Mk, Wk, Bk, dWk, dBk, in reversed(list(zip(
                itertools.count(), self.momentums, self.weights, self.biases, dW, dB
        ))):
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

    def save_nn(self, fpath):
        """ Save the neural network to the file `fpath`
        """
        data = {
            "sizes": self.layer_sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "hparams": self.hparams,
        }
        with open(fpath, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def load_nn(self, fpath):
        """ Load a neural network from the file `fpath`.
            Returns an instance of NeuralNetwork.
        """
        with open(fpath) as f:
            data = json.load(f)

        net = SimpleNeuralNetwork(data['sizes'], data['hparams'])
        net.weights = [np.array(w) for w in data['weights']]
        net.biases = [np.array(b) for b in data['biases']]
        return net


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


def regularized_cost(a, y, weights, regularisation):
    # a.shape = (m, 1)
    cost = cross_entropy_cost(a, y)
    n = a.shape[0]
    regularization = 0.5 * (regularisation / n) * \
                     sum(np.linalg.norm(w) ** 2 for w in weights)
    return cost + regularization


if __name__ == '__main__':
    main()