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

    train_X = train_X.T
    test_X = test_X.T
    valid_X = valid_X.T

    Y_classes = 10
    return Y_classes, (train_X, train_Y), (test_X, test_Y), (valid_X, valid_Y)


def load_toy():
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
    ]).T
    train_Y = np.array([1,1,1,0,0,0]).T

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
    ]).T
    test_Y = np.array([1,1,1,1,0,0,0]).T

    Y_classes = 2
    return Y_classes, (train_X, train_Y), (test_X, test_Y), (None, None)


def load_toy2():
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
    ]).T
    train_Y = np.array([1,1,1,0,0,0]).T

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
    ]).T
    test_Y = np.array([1,1,0]).T

    Y_classes = 2
    return Y_classes, (train_X, train_Y), (test_X, test_Y), (None, None)

