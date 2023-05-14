import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 
from tensorflow import keras
from sklearn import preprocessing as pre

def task1():
    df = pd.read_csv('nn_0.csv')
    plt.figure(figsize=(5, 5))
    plt.scatter(df['X1'], df['X2'], c=df['class'])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True)
    plt.show()
    X1 = np.array(df['X1']).reshape(-1, 1).astype('float32')
    X2 = np.array(df['X2']).reshape(-1, 1).astype('float32')
    Y = np.array(df['class']).reshape(-1, 1).astype('int32')
    Y[Y == -1] = 0
    X1 = pre.MinMaxScaler().fit_transform(X1)
    X2 = pre.MinMaxScaler().fit_transform(X2)
    X = np.concatenate((X1, X2), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
    accuracies = {}
    epoches = [i for i in range(5, 101, 5)]
    for epoch in epoches:
        model = keras.Sequential([ keras.layers.Flatten(input_shape=(2,)), keras.layers.Dense(1, activation="sigmoid")])
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=epoch, verbose=0, batch_size=64)
        test_loss, test_acc = model.evaluate(x_test, y_test)
        accuracies[epoch] = test_acc
    draw_plot(epoches, list(accuracies.values()), 'Epochs', 'Accuracy', 'Accuracy(Epochs)', 'AE1.png')
    activations = ["relu", "sigmoid", "softmax", "softplus", "softsign", "exponential", "elu", "selu", "tanh"]
    optimizers = ["Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "RMSprop", "SGD"]
    for activation in activations:
        accuracies = {}
        for optimizer in optimizers:
            model = keras.Sequential([ keras.layers.Flatten(input_shape=(2,)), keras.layers.Dense(1, activation=activation)])
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
            model.fit(x_train, y_train, verbose=0, epochs=64, batch_size=64)
            test_loss, test_acc = model.evaluate(x_test, y_test)
            accuracies[optimizer] = test_acc
        plt.bar(optimizers, list(accuracies.values()), color="blue")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy") 
        plt.title(activation) 
        plt.show()

def task2():
    df = pd.read_csv('nn_0.csv')
    X1 = np.array(df['X1']).reshape(-1, 1).astype('float32')
    X2 = np.array(df['X2']).reshape(-1, 1).astype('float32')
    Y = np.array(df['class']).reshape(-1, 1).astype('int32')
    Y[Y == -1] = 0
    X1 = pre.MinMaxScaler().fit_transform(X1)
    X2 = pre.MinMaxScaler().fit_transform(X2)
    X = np.concatenate((X1, X2), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
    accuracies = {}
    epoches = [i for i in range(5, 101, 5)]
    for epoch in epoches:
        model = keras.Sequential([ keras.layers.Flatten(input_shape=(2,)), keras.layers.Dense(64, activation="relu"), keras.layers.Dense(64, activation="relu"), keras.layers.Dense(1, activation="sigmoid")])
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=["accuracy"]) 
        model.fit(x_train, y_train, epochs=100, verbose=0, batch_size=64)
        test_loss, test_acc = model.evaluate(x_test, y_test) 
        accuracies[epoch] = test_acc
    plt.plot(epoches, list(accuracies.values()), color="blue", scaley=True)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy(Epochs)')
    plt.grid(True)
    plt.show()

def task3():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train/255.0
    x_test = x_test/255.0
    model = keras.Sequential([ keras.layers.Flatten(input_shape=(28, 28, 1)), keras.layers.Dense(128, activation="relu"), keras.layers.Dense(10, activation="softmax")])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Accuracy: {}\nLoss: {}".format(test_acc, test_loss))
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
    plt.imshow(x_train[i], cmap=pyplot.get_cmap('gray')) 
    plt.show()




