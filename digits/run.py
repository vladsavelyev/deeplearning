import itertools
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly_express as px
import plotly
show = plotly.offline.plot
from digits.load_data import load_mnist, load_toy2, get_Y, get_X, load_toy, collapse_digits, make_data
from digits.simple_model import NeuralNetwork, evaluate
import seaborn as sns

np.set_string_function(lambda a: str(a.shape), repr=False)

np.random.seed(1)


def show_images(X, Y=None, cols=3):
    # X.shape == (pixels, images)
    datas = X.T
    cmaps = itertools.repeat('Reds')
    if Y is not None:
        if len(Y.shape) > 1 and Y.shape[0] > 1:
            Y = np.argmax(Y, axis=0)  # convert to the flat shape
        cmaps = np.where(Y == 1, 'Greens', 'Reds')
        cmaps = cmaps.reshape((X.shape[1]))

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


print('Loading data...')
Y_classes, train, test, valid = load_mnist()


df = pd.DataFrame(dict(epoch=[], cost=[], acc=[], key=[]))

all_hparams = [
    # dict(inercia=0.04, subset=5000, hidden_layers=(100,), epochs=100, learning_rate=0.5, batch_max_size=10, regul_param=0.01, early_stop=1),
    # dict(inercia=0.04, subset=1000, hidden_layers=(100,), epochs=100, learning_rate=0.5, batch_max_size=10, regul_param=0.01, early_stop=1),
    dict(inercia=0.04, subset=1000, hidden_layers=(100,), epochs=100, learning_rate=0.5, batch_max_size=10, regul_param=1.0, early_stop=1),
    # dict(inercia=0.04, subset=1000, hidden_layers=(100,), epochs=100, learning_rate=0.5, batch_max_size=10, regul_param=5.0, early_stop=1),
]
for hparams in all_hparams:
    key = {k: v for k, v in hparams.items() if len(list(set(d[k] for d in all_hparams))) > 1}
    key = ', '.join(f'{k}: {v}' for k, v in key.items())
    nn, (accs, costs) = NeuralNetwork.run(train, valid_data=valid, **hparams)
    df = df.append(pd.DataFrame(dict(epoch=range(len(costs)), cost=costs, acc=accs, key=key)))

    # Evaluating on test data:
    pred_Y = nn.predict(get_X(test))
    acc = evaluate(pred_Y, get_Y(test), Y_classes)
    print(f'Accuracy on test data: {acc}')
    print()

cmn_params = {k: v for k, v in all_hparams[0].items() if len(list(set(d[k] for d in all_hparams))) == 1}
cmn_params = [f'{k}: {v}' for k, v in cmn_params.items()]
show(px.line(df, x='epoch', y='acc', color='key', title=', '.join(cmn_params)))


# Y_classes, train, test, valid = load_toy()
# hidden_layers = [8]
# epochs = 30
# learning_rate = 3.0
# batch_size = 10
# show_images(get_X(train), get_Y(train))



# show_images(nn.weights[0][0:9,:].T)
#
# # show average of all weights of first layer
# w_784_10 = np.dot(nn.weights[0].T + nn.biases[0].T, nn.weights[1].T) + nn.biases[1].T
# show_images(w_784_10)
#
# # train on digit 1 and show weights that are activated highly:
# As, Zs = nn.feedforward(get_X(train)[:,0:1])
# ws = nn.weights[0][np.where(np.round(As[-2]*1000) >= 1000, 1, 0)[:,0],:]
# # we.shape == (784, number-of-images)
# show_images(ws.T)
# # show 4th image (a 3)
# show_images(get_X(train)[:,3:4])

def explore_0layer_nn():
    # accuracy for each digit type
    plots = []
    for i in range(10):
        y = collapse_digits(get_Y(test))
        yi = y[y == i]
        x = get_X(test).T
        xi = x[y == i].T
        predictions = nn.predict(xi)
        acc = sum(int(p == t) for p, t in zip(predictions, yi)) / len(yi)
        print(f"Accuracy for digit {i}: {acc}")

        # plottoing for 1 example
        test_digit = xi[:,i:i+1]
        As, Zs = nn.feedforward(test_digit)
        weight = np.mean((As[-1] * nn.weights[0]).T, axis=1).reshape((xi.shape[0],1))
        plots.extend([test_digit, weight])

    show_images(np.concatenate(plots, axis=1), cols=4)






