import gzip
import pickle
import numpy as np


def load_mnist():
    """ train_X.shape = (50000, 784)
        train_Y.shape = (50000, 1)
        valid_X.shape = (10000, 784)
        valid_Y.shape = (10000, 1)
        test_X.shape = (10000, 784)
        test_Y.shape = (10000, 1)
    """
    path = '/Users/vsaveliev/git/vladsaveliev/neural-networks-and-deep-learning/DeepLearningPython35/mnist.pkl.gz'
    with gzip.open(path, 'rb') as f:
        (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = pickle.load(f, encoding='bytes')

    Y_classes = 10
    train_Y = expand_digits(train_Y, Y_classes)
    test_Y = expand_digits(test_Y, Y_classes)
    valid_Y = expand_digits(valid_Y, Y_classes)
    return Y_classes, make_data(train_X, train_Y), make_data(test_X, test_Y), make_data(valid_X, valid_Y)


def load_toy():
    Y_classes = 2
    train_X = np.array([
        [0,1,0,
         0,1,0,
         0,1,0],

        [1,1,0,
         0,1,0,
         0,1,0],

        [0,1,0,
         1,1,0,
         0,1,0],

        [1,1,1,
         1,0,1,
         1,1,1],

        [0,1,0,
         1,0,1,
         1,1,1],

        [0,1,0,
         1,0,1,
         0,1,0],
    ])
    train_Y = expand_digits([1,1,1,0,0,0], Y_classes)

    test_X = np.array([
        [0,1,1,
         0,0,1,
         0,0,1],

        [1,1,0,
         1,0,0,
         1,0,0],

        [1,0,0,
         1,0,0,
         1,0,0],

        [1,0,0,
         0,1,0,
         0,0,1],

        [0,1,1,
         1,0,1,
         1,1,1],

        [0,1,1,
         1,0,1,
         0,1,1],

        [0,1,1,
         1,1,1,
         0,1,1],
    ])
    test_Y = expand_digits([1,1,1,1,0,0,0], Y_classes)
    return Y_classes, make_data(train_X, train_Y), make_data(test_X, test_Y), None


def load_toy2():
    Y_classes = 2
    train_X = np.array([
        [0,0,1,
         0,0,0,
         0,0,0],

        [0,1,1,
         0,0,0,
         0,0,0],

        [0,0,1,
         0,0,1,
         0,0,0],

        [0,0,0,
         1,0,0,
         1,0,0],

        [0,0,0,
         1,0,0,
         1,1,0],

        [0,0,0,
         0,0,0,
         1,1,0],
    ])
    train_Y = expand_digits([1,1,1,0,0,0], Y_classes)

    test_X = np.array([
        [0,1,1,
         0,0,1,
         0,0,1],

        [0,0,1,
         0,0,1,
         0,0,0],

        [0,0,0,
         0,0,0,
         1,0,0]
    ])
    test_Y = expand_digits(np.array([1,1,0]), Y_classes)
    return Y_classes, make_data(train_X, train_Y), make_data(test_X, test_Y), None


def make_data(X, Y):
    # assuming X is an array of m training examples of k dimentions (shape = (m, k))
    # and Y is an array of m training examples of j dimentions (shape = (m, j))
    # making an array of tuples of shape (m, 2)
    return np.array(list(zip(X, Y)))


def expand_digits(Y, Y_classes):
    # convert Y digits to 10-long arrays of 0 or 1
    return np.array([[int(i == y) for i in range(Y_classes)] for y in Y])


def collapse_digits(Y):
    # converts to the flat shape (array of digits 0-9)
    return np.argmax(Y, axis=0)


def evaluate(pred_Y, val_Y, Y_classes):
    """ assume pred_Y and test_Y are 1-dim arrays of integers 0 through 9
    """
    if len(pred_Y.shape) == 1:
        pred_Y = expand_digits(pred_Y, Y_classes).T
    if val_Y.shape[0] > 1: val_Y = collapse_digits(val_Y)
    if pred_Y.shape[0] > 1: pred_Y = collapse_digits(pred_Y)
    assert pred_Y.shape == val_Y.shape, (pred_Y.shape, val_Y.shape)
    return sum(int(p == t) for p, t in zip(pred_Y, val_Y)) / len(pred_Y)


def get_X(data):
    # assuming data is an array of tuples of shape (m, 2).
    # extracing array of Xs of size (m, 1), and recombining elements into
    # an array of size (k, m) again using np.stack
    return np.stack(data[:, 0]).T


def get_Y(data):
    # assuming data is an array of tuples of shape (m, 2).
    # extracing array of Xs of size (m, 1), and recombining elements into
    # an array of size (j, m) again using np.stack
    return np.stack(data[:, 1]).T


def get_Y_flat(data):
    # assuming data is an array of tuples of shape (m, 2).
    # extracing array of Xs of size (m, 1), and recombining elements into
    # an array of size (1, m) again using np.stack
    # the difference with get_Y is that it returns 1-dimentional Ys containing 1 digit 0-9
    return np.stack(data[:, 1]).T



