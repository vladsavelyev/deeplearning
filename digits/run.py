import itertools
import math
import random
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly_express as pe
from digits.load_data import load_mnist, load_toy2, get_Y, get_X, load_toy
from digits.model import NeuralNetwork, evaluate

np.set_string_function(lambda a: str(a.shape), repr=False)

np.random.seed(1)


def show_images(X, Y=None):
    # X.shape == (pixels, images)
    datas = X.T
    cmaps = itertools.repeat('Reds')
    if Y is not None:
        if len(Y.shape) > 1 and Y.shape[0] > 1:
            Y = np.argmax(Y, axis=0)  # convert to the flat shape
        cmaps = np.where(Y == 1, 'Greens', 'Reds')
        cmaps = cmaps.reshape((X.shape[1]))

    cols = 3
    rows = math.ceil(len(datas) / cols)
    fig, axes = plt.subplots(rows, cols)
    axes = axes.reshape((rows, cols))
    for i, (x, cm) in enumerate(zip(datas, cmaps)):
        i, j = i//cols, i%cols
        width = math.ceil(math.sqrt(x.shape[0]))
        plt.axis('off')
        axes[i, j].get_xaxis().set_visible(False)
        axes[i, j].get_yaxis().set_visible(False)
        axes[i, j].imshow(x.reshape((width, width)), cmap=cm)
    plt.show()


def show_weights(nn, layer_n=1):
    second_layer = nn.weights[layer_n-1]  # (k, j)
    show_images(second_layer.T)


#%%
print('Loading data...')

Y_classes, train, test, valid = load_mnist()
hidden_layers = [100]
epochs = 30
learning_rate = 0.5
batch_size = 10
lamb = 5.0

# Y_classes, train, test, valid = load_toy()
# hidden_layers = [8]
# epochs = 30
# learning_rate = 3.0
# batch_size = 10
# show_images(get_X(train), get_Y(train))

#%%
nn = NeuralNetwork([get_X(train).shape[0]] + hidden_layers + [get_Y(train).shape[0]])
print('Training the NN...')
nn.learn(train, epochs, learning_rate, batch_size, test_data=test, print_cost_every=10, regul_param=5.0)
# show_images(train_X, nn.predict(train_X))

pred_Y = nn.predict(get_X(test))
# show_images(get_X(test), pred_Y)
# show_weights(nn)

acc = evaluate(pred_Y, get_Y(test), Y_classes)
print(f'Accuracy: {acc}')

# show_images(nn.weights[0][0:9,:].T)
#
# # show average of all weights of first layer
# w_784_10 = np.dot(nn.weights[0].T + nn.biases[0].T, nn.weights[1].T) + nn.biases[1].T
# show_images(w_784_10)
#
# # train on digit 3 and show weights that are activated highly:
# As, Zs = nn.feedforward(get_X(train)[:,0:1])
# ws = nn.weights[0][np.where(np.round(As[-2]*1000) >= 1000, 1, 0)[:,0],:]
# # we.shape == (784, number-of-images)
# show_images(ws.T)















