import itertools
import math
import random
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly_express as pe
from digits.load_data import load_mnist
from digits.model import NeuralNetwork, evaluate

np.set_string_function(lambda a: str(a.shape), repr=False)


def main():
    print('Loading data...')

    #%%
    Y_classes, (train_X, train_Y), (test_X, test_Y), (valid_X, valid_Y) = load_mnist()
    # Y_classes, (train_X, train_Y), (test_X, test_Y), (valid_X, valid_Y) = load_toy2()
    # show_images(train_X, train_Y)

    # # convert digits to 10-long arrays of 0 or 1
    train_Y = np.array([[int(i == y) for i in range(Y_classes)] for y in train_Y]).T
    print(f"X: {train_X.shape}, Y: {train_Y.shape}")

    #%%
    nn = NeuralNetwork([train_X.shape[0], 30, train_Y.shape[0]])
    print('Training the NN...')
    nn.learn(train_X, train_Y, epochs=1000, learning_rate=3, batch_max_size=10000,
             test_X=test_X, test_Y=test_Y, print_cost=True)
    # show_images(train_X, nn.predict(train_X))

    pred_Y = nn.predict(test_X)
    # show_images(test_X, pred_Y)
    # show_weights(nn)

    acc = evaluate(pred_Y, test_Y)
    print(f'Accuracy: {acc}')


def show_images(X, Y=None):
    datas = X.T
    cmaps = itertools.repeat('Reds')
    if Y is not None:
        if len(Y.shape) > 1:
            Y = np.argmax(Y, axis=0)  # convert to the flat shape
        cmaps = np.where(Y == 1, 'Greens', 'Reds')

    cols = 3
    rows = math.ceil(len(datas) / cols)
    fig, axes = plt.subplots(rows, cols)
    axes = axes.reshape((rows, cols))
    for i, (x, cm) in enumerate(zip(datas, cmaps)):
        i, j = i//cols, i%cols
        width = int(math.sqrt(x.shape[0]))
        axes[i, j].imshow(x.reshape((width, width)), cmap=cm)
    plt.show()


def show_weights(nn, layer_n=1):
    second_layer = nn.weights[layer_n-1]  # (k, j)
    show_images(second_layer.T)


if __name__ == '__main__':
    main()
