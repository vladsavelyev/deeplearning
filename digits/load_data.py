import gzip
import pickle
from os.path import dirname, join
import numpy as np
import theano
import theano.tensor as T


def shared_dataset(data_x, data_y):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, shared_y


def load_mnist():
    """ train_X.shape = (50000, 784)
        train_Y.shape = (50000, 1)
        valid_X.shape = (10000, 784)
        valid_Y.shape = (10000, 1)
        test_X.shape = (10000, 784)
        test_Y.shape = (10000, 1)
    """
    path = join(dirname(__file__), 'mnist.pkl.gz')
    with gzip.open(path, 'rb') as f:
        (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = \
            pickle.load(f, encoding='bytes')

    Y_classes = 10
    train_Y = digits_to_binary_arrays(train_Y, Y_classes)
    test_Y  = digits_to_binary_arrays(test_Y, Y_classes)
    valid_Y = digits_to_binary_arrays(valid_Y, Y_classes)
    return Y_classes, \
           shared_dataset(train_X, train_Y), \
           shared_dataset(test_X, test_Y), \
           shared_dataset(valid_X, valid_Y)


def load_toy():
    num_y_classes = 2
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
    train_Y = [1, 1, 1, 0, 0, 0]

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
    test_Y = [1, 1, 1, 1, 0, 0, 0]

    train_Y = digits_to_binary_arrays(train_Y, num_y_classes)
    test_Y = digits_to_binary_arrays(test_Y, num_y_classes)
    return num_y_classes, \
           shared_dataset(train_X, train_Y), \
           shared_dataset(test_X, test_Y), \
           shared_dataset(test_X, test_Y)


def load_toy2():
    num_y_classes = 2
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
    train_Y = digits_to_binary_arrays([1, 1, 1, 0, 0, 0], num_y_classes)

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
    test_Y = digits_to_binary_arrays(np.array([1, 1, 0]), num_y_classes)
    return num_y_classes, \
           shared_dataset(train_X, train_Y), \
           shared_dataset(test_X, test_Y), \
           shared_dataset(test_X, test_Y)


def zip_x_y(X, Y):
    # assuming X is an array of m training examples of k dimentions (shape = (m, k))
    # and Y is an array of m training examples of j dimentions (shape = (m, j))
    # making an array of tuples of shape (m, 2)
    return np.array(list(zip(X, Y)))


def digits_to_binary_arrays(Y, num_y_classes):
    # convert 0-9 digits into 10-long arrays of 0 or 1
    return np.array([[int(i == y) for i in range(num_y_classes)] for y in Y])


def binary_arrays_to_digits(Y):
    # convert 10-long arrays of 0/1 into a flat shape (arrays of digits 0-9)
    return np.argmax(Y, axis=0)


def evaluate(pred_Y, val_Y, Y_classes):
    """ assume pred_Y and test_Y are 1-dim arrays of integers 0 through 9
    """
    if len(pred_Y.shape) == 1:
        pred_Y = digits_to_binary_arrays(pred_Y, Y_classes).T
    if val_Y.shape[0] > 1: val_Y = binary_arrays_to_digits(val_Y)
    if pred_Y.shape[0] > 1: pred_Y = binary_arrays_to_digits(pred_Y)
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
    # the difference with get_Y is that it returns 1-dimentional
    # Ys containing 1 digit 0-9
    return np.stack(data[:, 1]).T



