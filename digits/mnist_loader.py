import gzip
import pickle
from os.path import dirname, join
import numpy as np


def load_data():
    """
    Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images. This is a
    numpy ndarray with 50,000 entries. Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries. Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    path = join(dirname(__file__), 'mnist.pkl.gz')
    with gzip.open(path, 'rb') as f:
        (train_x, train_Y), (valid_x, valid_Y), (test_X, test_y) = \
            pickle.load(f, encoding='bytes')
    return (train_x, train_Y), \
           (test_X, test_y), \
           (valid_x, valid_Y)


def load_data_wrapper():
    """
    Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code.
    """
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = load_data()
    num_x_classes = tr_x.shape[1]
    num_y_classes = max(tr_y) + 1
    tr_y = digits_to_binary_arrays(tr_y, num_y_classes)
    return num_x_classes, \
           num_y_classes, \
           zip_x_y(tr_x, tr_y), \
           zip_x_y(va_x, va_y), \
           zip_x_y(te_x, te_y)


def zip_x_y(x, y):
    """
    Assuming X is an array of m training examples of k dimentions (shape = (m, k))
    and Y is an array of m training examples of j dimentions (shape = (m, j))
    making an array of tuples of shape (m, 2)
    """
    return np.array(list(zip(x, y)))


def get_x(data):
    # assuming data is an array of tuples of shape (m, 2).
    # extracing array of Xs of size (m, 1), and recombining elements
    # into an array of size (k, m) again using np.stack
    return np.stack(data[:, 0]).T


def get_y(data):
    """ assuming data is an array of tuples of shape (m, 2).
    extracing array of Xs of size (m, 1), and recombining elements
    into an array of size (j, m) again using np.stack
    """
    return np.stack(data[:, 1]).T


def digits_to_binary_arrays(y_data, num_y_classes):
    """ convert 0-9 digits into 10-long arrays of 0 or 1
    """
    return np.array([[int(i == y) for i in range(num_y_classes)]
                     for y in y_data])


def binary_arrays_to_digits(Y):
    # convert 10-long arrays of 0/1 into a flat shape (arrays of digits 0-9)
    return np.argmax(Y, axis=0)


# def load_mnist_old():
#     """ train_X.shape = (50000, 784)
#         train_Y.shape = (50000, 1)
#         valid_X.shape = (10000, 784)
#         valid_Y.shape = (10000, 1)
#         test_X.shape = (10000, 784)
#         test_Y.shape = (10000, 1)
#     """
#     path = join(dirname(__file__), 'mnist.pkl.gz')
#     with gzip.open(path, 'rb') as f:
#         (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = \
#             pickle.load(f, encoding='bytes')
#
#     Y_classes = 10
#     train_Y = digits_to_binary_arrays(train_Y, Y_classes)
#     test_Y  = digits_to_binary_arrays(test_Y, Y_classes)
#     valid_Y = digits_to_binary_arrays(valid_Y, Y_classes)
#     return Y_classes, \
#            (train_X, train_Y), \
#            (test_X, test_Y), \
#            (valid_X, valid_Y)


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

    return num_y_classes, \
           (train_X, train_Y), \
           (test_X, test_Y), \
           (test_X, test_Y)


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
    train_Y = np.array([1, 1, 1, 0, 0, 0])

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
    test_Y = np.array([1, 1, 0])
    return num_y_classes, \
           (train_X, train_Y), \
           (test_X, test_Y), \
           (test_X, test_Y)


# def get_X(data):
#     # assuming data is an array of tuples of shape (m, 2).
#     # extracing array of Xs of size (m, 1), and recombining elements into
#     # an array of size (k, m) again using np.stack
#     return np.stack(data[:, 0]).T



