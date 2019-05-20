import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly_express as px
import plotly
import sklearn
import sklearn.datasets
import sklearn.linear_model

from week3.planar_utils import plot_decision_boundary, load_planar_dataset, load_extra_datasets
from week3.libs import load_planar_dataset, nn_model_test_case


class Model:
    def __init__(self):
        self.n_0 = None
        self.n_1 = None
        self.n_2 = None
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.last_A2 = None

    def learn(self, X, Y, hidden_layer_size=4, learning_rate=1, num_passes=20000, print_costs=True):
        self._calc_sizes(X, Y, hidden_layer_size)
        self._initialize_parameters()
        m = X.shape[1]
        # print(f"W1 = {self.W1}")
        # print(f"b1 = {self.b1}")
        # print(f"W2 = {self.W2}")
        # print(f"b2 = {self.b2}")
        A2 = None
        for i in range(num_passes):
            Z1, A1, Z2, A2 = self._forward(X)
            cost = calc_cost(A2, Y)
            if print_costs and (i % (num_passes // 10) == 0 or i == num_passes - 1):
                print(f"Cost at iteration {i}: {cost}")  #, W1={self.W1}, b1={self.b1}, W2={self.W2}, b2={self.b2}")
            dW1, dW2, db1, db2 = self._calc_grads(X, Z1, A1, Z2, A2, Y, m)
            self._update_parameters(dW1, dW2, db1, db2, learning_rate)
        return A2

    def _calc_sizes(self, X, Y, hidden_layer_size):
        """
        :param X: input, shape should be (number of input features, number of examples) = (2 (point cooridantes), 400)
        :param Y: output, shape should be (number of ouptut features, number of examples) = (1 (color), 400)
        """
        self.n_0 = X.shape[0]         # size of layer 0 (input layer) = number of input features
        self.n_1 = hidden_layer_size  # size of layer 1 (the hidden layer)
        self.n_2 = Y.shape[0]         # size of output layer, which is 1 in our case
        print(f"n0={self.n_0}, n1={self.n_1}, n2={self.n_2}")

    def _initialize_parameters(self):
        np.random.seed(2)
        self.W1 = np.random.randn(self.n_1, self.n_0) * 0.01  # weights of the connections between layers 0 and 1
        self.b1 = np.zeros((self.n_1, 1))  # biases of each node in layer 1
        self.W2 = np.random.randn(self.n_2, self.n_1) * 0.01  # weights of the connections between layers 1 and 2
        self.b2 = np.zeros((self.n_2, 1))  # biases of each output node

    def _forward(self, X):
        m = X.shape[1]
        assert X.shape == (self.n_0, m), (X.shape, self.n_0, m)

        assert self.W1.shape == (self.n_1, self.n_0)
        assert self.b1.shape == (self.n_1, 1)
        Z1 = np.dot(self.W1, X) + self.b1
        assert Z1.shape == (self.n_1, m)
        A1 = np.tanh(Z1)             # tanh
        # A1 = np.where(Z1 > 0, Z1, 0)   # relu
        assert A1.shape == (self.n_1, m)
        Z2 = np.dot(self.W2, A1) + self.b2
        assert Z2.shape == (self.n_2, m)
        A2 = sigmoid(Z2)
        assert A2.shape == (self.n_2, m)  # same size as outout
        return Z1, A1, Z2, A2

    def _calc_grads(self, X, Z1, A1, Z2, A2, Y, m):
        """ Forward was: A0
                         +(W1, b1) -> Z1=W1*A0+b1
                         -> A1=tanh(Z1)
                         +(W2, b2) -> Z2=W2*A1+b1
                         -> A2=s(Z2)
                         -> cost(A2, Y) = -1/m * sum(Y*log(A2) + (1-Y)*log(1-A2))

            Back is: dCost/dA2 = -sum( Y/A2 + (1-Y)/-((1-A2)) )/m
                     dCost/dZ2 = dCost/dA2 * ( dA2/dZ2 = ds(Z2) = ds(Z2)(1-ds(Z2)) = A2(1-A2) ) =
                               = -sum( Y/A2 + (1-Y)/-(1-A2) )/m * A2(1-A2)
                               = -sum( Y*(1-A2) + -(1-Y)*A2 )/m
                               = -sum( Y - Y*A2 - A2 + Y*A2 )/m
                               = -sum( Y - A2 ) / m
                     dCost/dW2 = dZ2 * ( dZ2/dW2 = A1 ) = dZ2 * A1
                     dCost/db2 = dZ2 * ( dZ2/db2 = 1 )  = dZ2
                     dCost/dA1 = dZ2 * ( dZ2/dA2 = W2 ) = dZ2 * W2
                     # dCost/dZ1 = dA1 * ( dA1/dZ1 = relu'(Z1) = np.where(Z1>0, 1, 0) )
                     dCost/dZ1 = dA1 * ( dA1/dZ1 = tanh'(Z1) = 1 − tanh(Z1)**2 = 1 - A1**2 )
                     dCost/dW1 = dZ1 * ( dZ1/dW1 = A0 ) = dZ1 * A0
                     dCost/db1 = dZ1 * ( dZ1/db1 = 1 )  = dZ1
        """
        A0 = X
        dZ2 = A2 - Y
        assert dZ2.shape == Z2.shape == (self.n_2, m), (dZ2.shape, Z2.shape, (self.n_2, m))
        assert A1.shape == (self.n_1, m)
        assert A1.T.shape == (m, self.n_1)
        dW2 = np.dot(dZ2, A1.T) / m
        assert dW2.shape == self.W2.shape == (self.n_2, self.n_1)
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        assert db2.shape == self.b2.shape == (self.n_2, 1)
        dA1 = np.dot(self.W2.T, dZ2)  # -> n1:n2 x n2:m = n1:m
        assert dA1.shape == A1.shape == (self.n_1, m)
        dZ1 = dA1 * (1 - A1**2)                # tanh
        # dZ1 = dA1 * np.where(Z1 > 0, 1, 0)     # relu
        assert dZ1.shape == Z1.shape == (self.n_1, m)
        assert A0.shape == (self.n_0, m)
        dW1 = np.dot(dZ1, A0.T) / m
        assert dW1.shape == self.W1.shape == (self.n_1, self.n_0)
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        assert db1.shape == self.b1.shape == (self.n_1, 1)
        return dW1, dW2, db1, db2

    def _update_parameters(self, dW1, dW2, db1, db2, learning_rate):
        """ We calculated cost for current parameters.
            Now we calculate derivatives of the cost function by each parameter W and b.
            Then we can shift parameters in the direction of lesser cost by learning_rate.
            Which would be W2-=learning_rate*dW2, etc
        """
        self.W2 -= learning_rate * dW2
        # print(f"b2 updated: {self.b2}->",)
        self.b2 -= learning_rate * db2
        # print(f"{self.b2}")
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def predict(self, X):
        Z1, A1, Z2, A2 = self._forward(X)
        self.last_A1 = A1
        self.last_A2 = A2
        self.last_Z1 = Z1
        self.last_Z2 = Z2
        return np.where(A2 > 0.5, 1, 0)

def calc_cost(A2, Y):
    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    assert(isinstance(cost, float))
    return cost

def calc_cost_diff(A2, Y):
    m = Y.shape[1]
    return 1/m * np.sum(np.abs(Y - A2))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return np.where(x > 0, x, 0)



# xx = np.arange(-2, 2, 1)
# yy = np.arange(-2, 2, 1)
# xx, yy = np.meshgrid(xx, yy)
# xx = xx.ravel()
# yy = yy.ravel()
# print(xx)  # (m, 1)
# print(yy)  # (m, 1)
# X = np.array([xx, yy]).T  # (m, 2)


def main():
    np.random.seed(1)  # set a seed so that the results are consistent

    X_train, Y_train = load_planar_dataset()

    d = pd.DataFrame(zip(X_train[0, :], X_train[1, :], Y_train[0, :]), columns=["x", "y", "color"])
    # p = px.scatter(d, x="x", y="y", color="color")
    # plotly.offline.plot(p)
    # plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
    # plt.show()

    m = Model()

    # hidden_layer_sizes = range(1, 20)
    # accs = []
    # costs = []
    hls = 6

    # for hls in hidden_layer_sizes:
    m.learn(X_train, Y_train, hidden_layer_size=hls, learning_rate=1.2, num_passes=2000, print_costs=False)

    # Generate a grid of points with distance h between them
    x_min, x_max = X_train[0, :].min() - 1, X_train[0, :].max() + 1
    y_min, y_max = X_train[1, :].min() - 1, X_train[1, :].max() + 1
    h = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh_X = np.array([xx.ravel(), yy.ravel()])
    assert mesh_X.shape == (2, xx.ravel().shape[0])
    mesh_preds = m.predict(mesh_X)
    print("Mesh: xx.shape, preds.shape:", xx.shape, mesh_preds.shape)

    print(m.last_A1.shape)   # (hls=colors in hidden layers, m)
    print(m.last_A2.shape)   # (1=color, m))

    xx_flat = xx.ravel()
    yy_flat = yy.ravel()
    # Plot the contour and training examples
    # print(xx.shape, m.last_A2.shape, m.last_A2.reshape(xx.shape).shape)
    rgba_colors = np.zeros((xx_flat.shape[0], 4))
    # mesh_preds = mesh_preds.reshape(xx_flat.shape)
    mesh_alphas = m.last_A2.reshape(xx_flat.shape)
    print("mesh_alphas:", mesh_alphas.shape)
    print("rgba_colors:", rgba_colors.shape)
    # for red the first column needs to be one
    print("rgba_colors[:, 0]", rgba_colors[:, 0].shape)
    print("np.where(mesh_alphas > 0.5, 1, 0)", np.where(mesh_alphas > 0.5, 1, 0).shape)
    rgba_colors[:, 0] = np.where(mesh_alphas < 0.5, 1, 0)
    # for blue the first column needs to be one
    rgba_colors[:, 2] = np.where(mesh_alphas >= 0.5, 1, 0)
    # the fourth column needs to be alphas
    rgba_colors[:, 3] = 1
    print(rgba_colors)

    plt.scatter(xx_flat, yy_flat, color=rgba_colors)
    # plt.scatter(xx, yy, mesh_preds.reshape(xx.shape), cmap=plt.cm.Spectral)
    plt.scatter(X_train[0, :], X_train[1, :], c=Y_train[0, :], s=40, cmap=plt.cm.Spectral)
    plt.show()

    pred_hidden_layers = np.where(m.last_A1 > 0.5, 1, 0)
    print("pred_hidden_layers:", pred_hidden_layers.shape)
    tmp = m.W2.T * m.last_A1 + m.b2  # (hls, 1) * (hls, m) + (hls, 1) = (hls, m)
    weights_hidden_layers = np.sum(tmp, axis=1, keepdims=True) / tmp.shape[1]  # (hls, m) -> (hls, )
    print("weights_hidden_layers:", weights_hidden_layers)

    plt.figure(figsize=(16, 32))
    for i in range(hls):
        plt.subplot(5, 2, i+1)
        pred_hl = pred_hidden_layers[i, :]
        print(xx.shape, yy.shape, pred_hl.shape)
        # alpha = m.last_A2
        plt.contourf(xx, yy, pred_hl.reshape(xx.shape), cmap=plt.cm.Spectral, alpha=weights_hidden_layers[i])
        plt.scatter(X_train[0, :], X_train[1, :], c=Y_train[0, :], s=40, cmap=plt.cm.Spectral)
    plt.show()

    #   acc = float((np.dot(Y_train, pred_train.T) + np.dot(1 - Y_train,1 - pred_train.T))/float(Y_train.size)*100)
    #   accs.append(acc)
    #
    #   cost = calc_cost(m.last_A2, Y_train)
    #   costs.append(cost)
    #
    #   print(f" {hls} hidden units. Accuracy: {acc}, cost: {cost}")
    # plt.plot(hidden_layer_sizes, costs)
    # plt.show()
    #
    # plt.plot(hidden_layer_sizes, accs)
    # plt.show()


main()






















