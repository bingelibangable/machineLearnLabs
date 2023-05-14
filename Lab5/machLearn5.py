import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def task1():
    data = pd.read_csv('reglab1.txt', sep='\t')
    data.head()
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes = np.ravel(axes)
    for i, ax in enumerate(axes):
        x1 = data.iloc[:, i-1]
        x2 = data.iloc[:, i]
        ax.scatter(x1, x2)
        ax.set_xlabel(x1.name)
        ax.set_ylabel(x2.name)
    plt.show()
    for i in range(3):
        k = i % 3
        features = data.iloc[:, [i-1, i]]
        target = data.loc[:, data.columns.difference(features.columns)]
        model = LinearRegression().fit(features, target)
    
        print('Features:', *features.columns.values, '; target:', *target.columns.values)
        print('Score:', model.score(features, target))
        print()

def get_RSS(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.4, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    error = y_test.values - model.predict(X_test)
    return sum(error ** 2)

def task2():
    data = pd.read_csv('reglab.txt', sep='\t')
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    results = dict()
    rss = get_RSS(X, y)
    print('Features:', X.columns.values)
    print('RSS:', rss)
    print()
    results[' '.join(X.columns.values)] = rss
    columns_1 = X.columns.values
    used_1 = []
    for i, col_1 in enumerate(columns_1):
        features = X.loc[:, [col_1]]
        rss = get_RSS(features, y)
        print('Features:', features.columns.values)
        print('RSS:', rss)
        print()
        results[' '.join(features.columns.values)] = rss
        columns_2 = columns_1.copy()[i+1:]
        for j, col_2 in enumerate(columns_2):
            features = X.loc[:, [col_1, col_2]]
            rss = get_RSS(features, y)
            print('Features:', features.columns.values)
            print('RSS:', rss)
            print()
            results[' '.join(features.columns.values)] = rss
            columns_3 = columns_2.copy()[j+1:]
        
            for k, col_3 in enumerate(columns_3):
                features = X.loc[:, [col_1, col_2, col_3]]
                rss = get_RSS(features, y)
                print('Features:', features.columns.values)
                print('RSS:', rss)
                print()
                results[' '.join(features.columns.values)] = rss

def task3():
    data = pd.read_csv('cygage.txt', sep='\t')
    X = data['Depth']
    y = data['calAge']
    weights = data['Weight']
    X = X.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    model = LinearRegression()
    model.fit(X, y, sample_weight=weights)
    y_pred1 = model.predict(X)
    plt.figure(figsize=(7, 5))
    plt.plot(X, y, X, y_pred, X, y_pred1)
    plt.ylabel('calAge')
    plt.xlabel('Depth')
    plt.show()

def taks4():
    data = pd.read_csv('longley.csv')
    data = data.drop('Population', axis=1)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=56)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print('Train score:', model.score(X_train, y_train))
    print('Test score:', model.score(X_test, y_test))
    train_res = []
    test_res = []
    alpha = list()
    for i in range(26):
        model = Ridge(alpha=10**(-3+0.2*i))
        alpha.append(10**(-3+0.2*i))
        model.fit(X_train, y_train)
        train_res.append(model.score(X_train, y_train))
        test_res.append(model.score(X_test, y_test))
    plt.plot(alpha, train_res, label='train')
    plt.plot(alpha, test_res, label='test')
    plt.xlabel('alpha')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    model = Ridge()
    model.fit(X_train, y_train)
    print('Ridge Regression')
    print('Train score:', model.score(X_train, y_train))
    print('Test score:', model.score(X_test, y_test))

def task5():
    data = pd.read_csv('eustock.csv')
    data.head()
    coefs = []
    intercepts = []
    X = np.arange(data.shape[0]).reshape(-1, 1)
    plt.figure(figsize=(12, 9))
    for col in data.columns.values:
        y = data[col]
        plt.plot(X, y, label=col)
        model = LinearRegression()
        model.fit(X, y)
        coefs.append(model.coef_)
        intercepts.append(model.intercept_)
        print(col)
        print('Coefficient of determination:', model.score(X, y))
        print('Slope coef:', model.coef_)
        print()
    plt.xticks()
    plt.yticks()
    plt.legend()
    plt.show()
    model = LinearRegression()
    model.fit(X, data)
    print('All')
    print('Coefficient of determination:', model.score(X, data))
    print('Slope coef:', model.coef_)
    plt.figure(figsize=(12, 9))
    for i, col in enumerate(data.columns.values):
        vals = coefs[i] * X + intercepts[i]
        plt.plot(X, vals, label=col)
    plt.xticks()
    plt.yticks()
    plt.legend()
    plt.legend()
    plt.show()

def task6():
    data = pd.read_csv('JohnsonJohnson.csv')
    data.head()
    data[['Year', 'Quater']] = data['index'].str.split(' ', expand=True)
    data.head()
    data = data.drop('index', axis=1)
    data = data.pivot(index='Year', columns='Quater', values='value')
    data = data.rename_axis([None], axis=1)
    data.head()
    coefs = []
    intercepts = []
    X = np.arange(1960, 1981).reshape(-1, 1)
    target_year = np.array([[2016]])
    plt.figure(figsize=(12, 9))
    for col in data.columns.values:
        y = data.loc[:, col]
        plt.plot(X, y, label=col)
        model = LinearRegression()
        model.fit(X, y)
        coefs.append(model.coef_)
        intercepts.append(model.intercept_)
        print(col)
        print('Coefficient of determination:', model.score(X, y))
        print('Slope coef:', model.coef_)
        print('Prediction:', model.predict(target_year))
    plt.xticks(range(1960, 1981, 1))
    plt.yticks()
    plt.legend()
    plt.legend()
    plt.title("Изменение прибыли")
    plt.show()
    model = LinearRegression()
    model.fit(X, data)
    coefs.append(model.coef_)
    intercepts.append(model.intercept_)
    print('All')
    print('Coefficient of determination:', model.score(X, data))
    print('Slope coef:', model.coef_)
    print('Prediction:', model.predict(target_year).mean())
    plt.figure(figsize=(12, 9))
    for i, col in enumerate(data.columns.values):
        vals = coefs[i] * X + intercepts[i]
        plt.plot(X, vals, label=col)
    plt.xticks(range(1960, 1981, 1))
    plt.yticks()
    plt.legend()
    plt.legend()
    plt.title("Линейная регрессия")
    plt.show()

def task7():
    data = pd.read_csv('cars.csv')
    data.head()
    X = data['speed'].values.reshape(-1, 1)
    y = data['dist']
    model = LinearRegression()
    model.fit(X, y)
    print('Coefficient of determination:', model.score(X, y))
    print('Prediction:', model.predict([[40]]))
    vals = model.coef_[0] * X + model.intercept_
    plt.figure(figsize=(8, 5))
    plt.plot(X, vals, c='r')
    plt.xlabel('Скорость')
    plt.ylabel('Дистанция')
    plt.scatter(X, y)
    plt.show()

def task8():
    data = pd.read_csv('svmdata6.txt', sep='\t')
    data.head()
    X = data['X'].values.reshape(-1, 1)
    y = data['Y']
    mse = []
    for eps in np.arange(0, 1, 0.05):
        model = SVR(epsilon=eps)
        model.fit(X, y)
        mse.append(mean_squared_error(y, model.predict(X)))
    plt.plot(np.arange(0, 1, 0.05), mse)
    plt.xlabel('Эпсилон')
    plt.ylabel('Ошибка')
    plt.show()

def task9():
    data = pd.read_csv('nsw74psid1.csv')
    data.head()
    X = data.loc[:, data.columns != 're78']
    y = data.loc[:, 're78']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=47)
    models = (
        LinearRegression(),
        SVR(),
        SVR(kernel='poly', degree=2),
        DecisionTreeRegressor(),
        DecisionTreeRegressor(max_depth=3)
    )

    titles = (
        'Linear regression',
        'Default SVR',
        'SVR with poly kernel and degree=3',
        'Default Decision tree',
        'Decision tree with max_depth=3'
    )

    for model, title in zip(models, titles):
        model.fit(X_train, y_train)
        print(title)
        print('Test score:', model.score(X_test, y_test))
        print()