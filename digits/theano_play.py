import math

from theano import tensor as T
import numpy as np
import theano
from theano.tensor.nnet import sigmoid

# state = shared(0)
# inc = T.iscalar('inc')
# accumulator = function([inc], state, updates=[(state, state+inc)])
# print(state.get_value())
# accumulator(1)
# print(state.get_value())
#
# state = shared(0)
# fn_of_state = state * 2 + inc
# # The type of foo must match the shared variable we are replacing
# # with the ``givens``:
# foo = T.scalar(dtype=state.dtype)
# skip_shared = function([inc, foo], fn_of_state, givens={state: foo})
# print(state.get_value())
# skip_shared(1, 3)
# print(state.get_value())

from digits.load_data import load_mnist, load_toy
y_classes, training_data, test_data, validation_data = load_mnist()
training_x, training_y = training_data
test_x, test_y = test_data

def binary_arrays_to_digits(y_pred):
    # convert 10-long arrays of 0/1 into a flat shape (arrays of digits 0-9)
    return T.argmax(y_pred, axis=1)

def accuracy(y_pred, y_test):
    # assert pred.eval().shape == y.eval.shape
    # return T.mean(T.eq(pred, y))
    y_pred_flat = binary_arrays_to_digits(y_pred)
    y_test_flat = binary_arrays_to_digits(y_test)
    return T.mean(T.eq(y_pred_flat, y_test_flat))

def cross_entropy_cost(a, y):
    return -T.sum(y * T.log(a) + (1 - y) * T.log(1 - a))

n_in = training_x.get_value(borrow=True).shape[1]  # num of picture pixels
n_out = y_classes
w = theano.shared(
    np.asarray(np.random.normal(
        loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)),
        dtype=theano.config.floatX
    ), name='w', borrow=True)
b = theano.shared(
    np.asarray(np.random.normal(
        loc=0.0, scale=1.0, size=(n_out,)),
        dtype=theano.config.floatX
    ), name='b', borrow=True)
params = [w, b]

x = T.matrix('x')
y = T.matrix('y')

batch_size = 100
layer_input = x.reshape((batch_size, n_in))
layer_output = sigmoid(T.dot(layer_input, w) + b)

cost = cross_entropy_cost(layer_output, y)
# cost = -T.sum(y * T.log(a) + (1 - y) * T.log(1 - a))
[gw, gb] = T.grad(cost, [w, b])
updates = {w: w - gw, b: b - gb}

pred = binary_arrays_to_digits(layer_output)

num_training_batches = math.ceil(training_x.get_value().shape[0] / batch_size)
num_test_batches = math.ceil(test_x.get_value().shape[0] / batch_size)
i = T.lscalar()  # batch index
train = theano.function(
    inputs=[i], outputs=[pred, cost], updates=updates,
    name='train', givens={
        x: training_x[i * batch_size: (i+1) * batch_size],
        y: training_y[i * batch_size: (i+1) * batch_size]
    })
# predict = theano.function(
#     inputs=[i], outputs=[pred],
#     name='predict', givens={
#         x: training_x[i * batch_size: (i+1) * batch_size],
#         y: training_y[i * batch_size: (i+1) * batch_size]
#     }
# )
test_data_accuracy = theano.function(
    inputs=[i], outputs=accuracy(layer_output, y),
    name='test_data_accuracy', givens={
        x: test_x[i * batch_size: (i+1) * batch_size],
        y: test_y[i * batch_size: (i+1) * batch_size]
    })

epochs = 30
for epoch in range(epochs):
    for batch_i in range(num_training_batches):
        pred_ij, cost_ij = train(batch_i)
        print(f'cost_ij: {cost_ij}')
        test_acc = np.mean([test_data_accuracy(j) for j in range(num_test_batches)])
        print(f'The corresponding test accuracy is {test_acc}')

# train_batch = function(
#     [i], cost, updates=updates,
#     givens={
#         x: training_x,
#         y: training_y
#     })

# i = T.lscalar()  # batch index
# train_batch = function(
#     [i], cost, updates=updates,
#     givens={
#         x: training_x[i * batch_size: (i+1) * batch_size],
#         y: training_y[i * batch_size: (i+1) * batch_size]
#     })


""" play in numpy:
w = np.asarray(np.random.normal(
       loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)),
       dtype=theano.config.floatX
)
b = np.asarray(np.random.normal(
      loc=0.0, scale=1.0, size=(n_out,)),
      dtype=theano.config.floatX
)
x = training_x.get_value()
y = training_y.eval()
a = 1 / (1 + np.exp(-np.dot(x, w)-b))
pred = T.where(a > 0.5

cost = -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))

"""