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

class Layer:
    def __init__(self):
        self.n_in = None
        self.n_out = None
        self.activation_fn = None

        self.params = []
        self.w = None
        self.b = None

        self.input = None
        self.output = None

    def set_input(self, inpt, batch_size):
        pass

    def cost(self):
        "Return the log-likelihood cost."

    def predict(self):
        pass

    def accuracy(self, y):
        pass

class FullyConnectedLayer(Layer):
    def __init__(self, n_in, n_out, activation_fn=sigmoid):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(np.random.normal(
                loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX
            ), name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(
                loc=0.0, scale=1.0, size=(n_out,)),
                dtype=theano.config.floatX
            ), name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_input(self, inpt, batch_size):
        self.input = inpt.reshape((batch_size, self.n_in))
        self.output = self.activation_fn(T.dot(self.input, self.w) + self.b)

    def predict(self):
        return binary_arrays_to_digits(self.output)

    def accuracy(self, y):
        y_pred_flat = binary_arrays_to_digits(self.output)
        y_test_flat = binary_arrays_to_digits(y)
        return T.mean(T.eq(y_pred_flat, y_test_flat))

pixels_in_picture = training_x.get_value().shape[1]
output_classes = training_y.eval().shape[1]
layers = [
    FullyConnectedLayer(n_in=pixels_in_picture, n_out=100),
    FullyConnectedLayer(n_in=100, n_out=output_classes)
]
params = []
for layer in layers:
    params.extend(layer.params)

x = T.matrix('x')
y = T.matrix('y')

epochs = 30
learning_rate = 1.0
batch_size = 1000
num_training_batches = math.ceil(training_x.get_value().shape[0] / batch_size)
num_test_batches = math.ceil(test_x.get_value().shape[0] / batch_size)

layers[0].set_input(x, batch_size)
for prev_layer, layer in zip(layers[:-1], layers[1:]):
    layer.set_input(prev_layer.output, batch_size)

# cost = cross_entropy_cost(layers[-1].output, y)
cost = -T.mean(y * T.log(layers[-1].output) + (1 - y) * T.log(1 - layers[-1].output))
grads = T.grad(cost, params)
updates = [(p, p - g) for p, g in zip(params, grads)]
pred = binary_arrays_to_digits(layers[-1].output)

i = T.lscalar()  # batch index
train = theano.function(
    inputs=[i], outputs=cost, updates=updates,
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
    inputs=[i], outputs=accuracy(layers[-1].output, y),
    name='test_data_accuracy', givens={
        x: test_x[i * batch_size: (i+1) * batch_size],
        y: test_y[i * batch_size: (i+1) * batch_size]
    })

for epoch in range(epochs):
    print(f'Epoch {epoch}')
    for batch_i in range(num_training_batches):
        cost = train(batch_i)
        print(f'  Current cost: {cost}')
        test_acc = np.mean([test_data_accuracy(j) for j in range(num_test_batches)])
        print(f'  The corresponding test accuracy is {test_acc}')

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