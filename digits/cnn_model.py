import itertools
import json
import math
import random
import numpy as np
import datetime

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

from digits.load_data import get_X, get_Y, evaluate, collapse_digits, load_mnist
np.set_string_function(lambda a: str(a.shape), repr=False)


def main():
    y_classes, train, test, valid = load_mnist()
    nn = CNN.run(train, valid, subset=1000)
    nn.predict(test)


class ConvPoolLayer:
    def __init__(self, m, inp_ftr_maps_n, image_h, image_w, out_ftr_maps_n, field_h, field_w, poolsize=(2, 2)):
        self.input_shape = m, inp_ftr_maps_n, image_h, image_w
        self.poolsize = poolsize
        self.filter_shape = out_ftr_maps_n, inp_ftr_maps_n, field_h, field_h

        # there are "num-input-feature-maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = inp_ftr_maps_n * image_h * image_w
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" / pooling size
        fan_out = out_ftr_maps_n * field_h * field_w // np.prod(poolsize)

        # initialize weights with random weights.
        # adjusting the normal distribution to avoid flat tails that will oversaturate weights:
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                np.random.uniform(low=-W_bound, high=W_bound, size=self.filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((out_ftr_maps_n,), dtype=theano.config.floatX)
        self.b = theano.shared(b_values, borrow=True)

        conv_out = conv2d(input, self.W, self.input_shape, self.filter_shape)

        pooled_out = pool.pool_2d(conv_out, ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

    def set_input(self, inpt, mini_batch_size):
        self.input = inpt.reshape(self.input_shape)

        conv_out = conv2d(
            input=self.input, filters=self.W, filter_shape=self.filter_shape,
            image_shape=self.input_shape)

        pooled_out = pool.pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

    class FullyConnectedLayer(object):
        def __init__(self, n_in, n_out):
            self.n_in = n_in
            self.n_out = n_out

            # Initialize weights and biases
            self.w = theano.shared(
                np.asarray(np.random.normal(loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)),
                           dtype=theano.config.floatX),
                name='w', borrow=True)

            self.b = theano.shared(
                np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                           dtype=theano.config.floatX),
                name='b', borrow=True)

        def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
            self.input = inpt.reshape((mini_batch_size, self.n_in))
            self.output = T.tanh(T.dot(self.input, self.w) + self.b)
            self.y_out = T.argmax(self.output, axis=1)

        def accuracy(self, y):
            "Return the accuracy for the mini-batch."
            return T.mean(T.eq(y, self.y_out))


class CNN:
    @staticmethod
    def run(train_data, valid_data, **hparams):
        hparams = hparams.copy()
        print(f'Training the NN with hyperparams {hparams}')
        if 'subset' in hparams:
            train_data = train_data[:hparams.pop('subset')]

        y_classes = get_Y(train_data).shape[0]

        nn = CNN(train_data, y_classes=y_classes)

        return nn, nn.learn(np.copy(train_data), valid_data=valid_data, **hparams)


    def __init__(self, train_data, y_classes, receptive_field_size=5, feature_maps_num=6, polling_region_size=2):
        np.random.seed(2)

        x = get_X(train_data)
        pixels, m = x.shape
        img_colors = 1
        img_h = int(round(math.sqrt(pixels)))
        img_w = pixels // img_h

        layer0_input = x.reshape((m, img_colors, img_h, img_w))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, 6, 12, 12)
        layer0 = LeNetConvPoolLayer(layer0_input, feature_maps_num, receptive_field_size, receptive_field_size,
                                    (polling_region_size, polling_region_size))

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, 6 * 12 * 12)
        layer_hidden_input = layer0.output.flatten(2)
        layer_hidden = HiddenLayer(layer_hidden_input, n_in = )

    #
    #     self.image_shape = image_shape
    #     self.y_classes = y_classes
    #     self.receptive_field_size = receptive_field_size
    #     self.feature_maps_num = feature_maps_num
    #     self.polling_region_size = polling_region_size
    #     self.feature_map_h = image_shape[0] - receptive_field_size + 1
    #     self.feature_map_w = image_shape[1] - receptive_field_size + 1
    #     assert self.feature_map_h > 0, (image_shape[0], receptive_field_size, self.feature_map_h)
    #     assert self.feature_map_w > 0, (image_shape[1], receptive_field_size, self.feature_map_w)
    #
    #     self.a0_size = receptive_field_size ** 2
    #     self.a1_size = feature_maps_num
    #     w1_shape = self.a1_size, self.a0_size  # total number of such networks is "feature_map_shape"
    #     Ws1_shape = self.feature_map_h, self.feature_map_w, self.a1_size, self.a0_size
    #     self.Ws1 = np.random.randn(*Ws1_shape) / 100
    #     Bs1_shape = self.feature_map_h, self.feature_map_w, self.a1_size, 1
    #     self.Bs1 = np.random.randn(*Bs1_shape)
    #
    #     self.a1_polled_h = self.feature_map_h // self.polling_region_size
    #     self.a1_polled_w = self.feature_map_w // self.polling_region_size
    #     self.a1_polled_size = self.feature_maps_num * self.a1_polled_h * self.a1_polled_w
    #     W2_shape = y_classes, self.a1_polled_size
    #     B2_shape = y_classes, 1
    #     self.W2 = np.random.randn(*W2_shape ) / np.sqrt(W2_shape [1])
    #     self.B2 = np.random.randn(*B2_shape)


    def feedforward(self, X):
        # shape of X is (p, n)
        n = X.shape[-1]


        X = X.copy()
        X = X.reshape(*self.image_shape, -1)  # (p, m) -> (x, y, m) - converting from flat to spacial

        As1_shape = self.feature_map_h, self.feature_map_w, self.a1_size, n
        Zs1 = np.zeros(As1_shape)
        As1 = np.zeros(As1_shape)
        for window_i in range(self.feature_map_h):
            for window_j in range(self.feature_map_w):
                A0 = X[window_i:window_i + self.receptive_field_size, window_j:window_j + self.receptive_field_size, :]
                A0 = A0.reshape(self.a0_size, -1)  # converting from spacial to flat
                W1 = self.Ws1[window_i, window_j, :, :]
                B1 = self.Bs1[window_i, window_j, :, :]
                Z1 = np.dot(W1, A0) + B1
                Zs1[window_i, window_j, :, :] = Z1
                As1[window_i, window_j, :, :] = sigmoid(Z1)

        # Max-polling
        A1_polled = np.zeros((self.a1_polled_size, n))
        A1_polled_max_index = np.zeros((self.a1_polled_size, n))
        for feature_map_k in range(self.feature_maps_num):
            for polled_i in range(self.a1_polled_h):
                for polled_j in range(self.a1_polled_w):
                    ap1 = As1[polled_i*self.polling_region_size : (polled_i+1)*self.polling_region_size,
                              polled_j*self.polling_region_size : (polled_j+1)*self.polling_region_size,
                              feature_map_k,
                              :]
                    max_index = np.argmax(ap1, axis=(0, 1))
                    A1_polled[feature_map_k * polled_i * polled_j] = np.max(ap1, axis=(0, 1))

        Z2 = np.dot(self.W2, A1_polled) + self.B2
        A2 = sigmoid(Z2)
        return Zs1, As1, A1_polled, Z2, A2

    def backprop(self, n, cache, train_Y, learning_rate, regul_param, inercia):
        """
        n is the size of full training set

        forward:
        1. for each window, Zij = Wij x Aij + B1, Aij = s(Zij)
        2. A1_polled = max()
        3. Z2 = W2 x A1_polled + B2, A2 = s(Z2), cost = c(A2, Y)

        back:
        1. dA2 = Y - A2, dZ2 = dA
        cost = MSR(Ak)  Ak = s(Zk)          Zk = Wkj x Aj + Bk           ->
        dAk  = Y - Ak   dZk = dAk * s'(Zk)  dWkj = dZk.T x Aj, dBk = dZk, dAj = dZk x Wkj


        """
        m = train_Y.shape[1]  # m is the size of mini-batch training set

        Zs1, As1, A1_polled, Z2, A2 = cache

        dZ2 = A2 - train_Y
        dW2 = np.dot(dZ2, A1_polled) / m
        dB2 = np.sum(dZ2, axis=1, keepdims=1) / m
        dA1_polled = np.dot(self.W2.T, dZ2)

        dAs1 = np.zeros(As1.shape)
        dZs1 = np.zeros(As1.shape)  # self.feature_map_h, self.feature_map_w, self.a1_size, n
        for window_i in range(self.feature_map_h):
            for window_j in range(self.feature_map_w):


        # ap1 = As1[polled_i*self.polling_region_size : (polled_i+1)*self.polling_region_size,
        #           polled_j*self.polling_region_size : (polled_j+1)*self.polling_region_size,
        #           feature_map_k,
        #           :]
        # A1_polled[feature_map_k * polled_i * polled_j] = np.max(ap1, axis=(0, 1))

        # # collecting dW and dB
        # dW = [0] * (len(self.layer_sizes) - 1)
        # dB = [0] * (len(self.layer_sizes) - 1)
        #
        # dZk = activations[-1] - train_Y  # cross-entropy cost
        # dAk = None
        # for k, Zk, Ak, Aj in reversed(list(zip(itertools.count(), zs, activations[1:], activations[:-1]))):
        #     if dZk is None:
        #         dZk = dAk * sigmoid_prime(Zk)
        #
        #     dW[k] = np.dot(dZk, Aj.T) / m   # (k_layer_size, m) x (j_layer_size, m).T -> (k_ls, j_ls)
        #     dB[k] = np.sum(dZk, axis=1, keepdims=1) / m  # (k_layer_size, 1)
        #
        #     dAj = np.dot(self.weights[k].T, dZk)  # (j_ls, k_ls) x (k_ls, m) -> (j_ls, m)
        #     # moving to the next layer
        #     dAk = dAj
        #     dZk = None
        #
        # # updating weights and b iases
        # for k, Mk, Wk, Bk, dWk, dBk, in reversed(list(zip(
        #         itertools.count(), self.momentums, self.weights, self.biases, dW, dB))):
        #     Mk = Mk * inercia - learning_rate * dWk  # calculating new momentum at the moment
        #     Wk += Mk  # updating the weight with the momentum
        #     Wk -= learning_rate * (regul_param / n * self.weights[k])  # applying regularization
        #     Bk -= learning_rate * dBk  # updating the biases - simply, without momentum or regularization
        #     self.momentums[k] = Mk
        #     self.weights[k] = Wk
        #     self.biases[k] = Bk

    @staticmethod
    def _predict(A):
        return collapse_digits(A)

    def predict(self, X):
        Zs1, As1, A1polled, Z2, A2 = self.feedforward(X)
        return self._predict(A2)

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
                Zs1, As1, A1_polled, Z2, A2 = self.feedforward(batch_X)
                self.backprop(n, (Zs1, As1, A1_polled, Z2, A2), batch_Y, learning_rate, regul_param=regul_param, inercia=inercia)
                output_activations.append(A2)

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

        Zs1, As1, A1_polled, Z2, A2 = self.feedforward(get_X(train_data))
        y = get_Y(train_data)
        weights = np.stack(self.Ws1.flatten(), self.W2.flatten())
        cost = regularized_cost(A2, y, weights, regul_param=regul_param)
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
    regularization = 0
    if weights:
        regularization = 0.5 * (regul_param / n) * sum(np.linalg.norm(w) ** 2 for w in weights)
    return cost + regularization


if __name__ == '__main__':
    main()