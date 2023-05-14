import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def task1():
    dataset = pd.read_csv('glass.csv', header = None)
    df = pd.DataFrame(dataset)
    print(df)
    df=df.drop(df.columns[0], axis=1)
    print(df)
    X = (df[df.columns[[0,1,2,3,4,5,6,7,8]]]).to_numpy()
    X = np.delete(X, (0), axis=0)
    y = df[10]
    y = np.delete(y, (0), axis=0)
    xVal = list()
    yVal = list()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    print(clf.score(X_test,y_test))
    for i in range(50):
        clf = BaggingClassifier(estimator=GaussianNB(), n_estimators=i + 1, random_state=6).fit(X_train, y_train)
        xVal.append(i)
        yVal.append(clf.score(X_test, y_test))
    plt.plot(xVal,yVal)
    plt.xlabel("кол-во классификаторов")
    plt.ylabel("точность")
    plt.title("наивный байесовский")
    ax = plt.gca()
    plt.show()
    xVal.clear()
    yVal.clear()
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print(clf.score(X_test,y_test))
    for i in range(50):
        clf = BaggingClassifier(estimator=tree.DecisionTreeClassifier(), n_estimators=i + 1, random_state=6).fit(X_train, y_train)
        xVal.append(i)
        yVal.append(clf.score(X_test, y_test))
    plt.plot(xVal,yVal)
    plt.xlabel("кол-во классификаторов")
    plt.ylabel("точность")
    plt.title("Дерево решений")
    ax = plt.gca()
    plt.show()
    xVal.clear()
    yVal.clear()
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)
    print(clf.score(X_test,y_test))
    for i in range(50):
        clf = BaggingClassifier(estimator=KNeighborsClassifier(n_neighbors=1), n_estimators=i + 1, random_state=6).fit(X_train, y_train)
        xVal.append(i)
        yVal.append(clf.score(X_test, y_test))
    plt.plot(xVal,yVal)
    plt.xlabel("кол-во классификаторов")
    plt.ylabel("точность")
    plt.title("KNN")
    ax = plt.gca()
    plt.show()

def task2():
    dataset = pd.read_csv('vehicle.csv', header = None)
    df = pd.DataFrame(dataset)
    print(df)
    X = (df[df.columns[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]]).to_numpy()
    X = np.delete(X, (0), axis=0)
    y = df[18]
    y = np.delete(y, (0), axis=0)
    print(X)
    print(y)
    xVal = list()
    yVal = list()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)
    for i in range(50):
        clf = AdaBoostClassifier(estimator=GaussianNB(),n_estimators=i + 1, random_state=0)
        clf.fit(X_train,y_train)
        xVal.append(i)
        yVal.append(clf.score(X_test, y_test))
    plt.plot(xVal,yVal)
    plt.xlabel("кол-во классификаторов")
    plt.ylabel("точность")
    plt.title("наивный байесовский")
    ax = plt.gca()
    plt.show()
    xVal.clear()
    yVal.clear()
    for i in range(50):
        clf = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(max_depth=2),n_estimators=i + 1, random_state=6)
        clf.fit(X_train,y_train)
        xVal.append(i)
        yVal.append(clf.score(X_test, y_test))
    plt.plot(xVal,yVal)
    plt.xlabel("кол-во классификаторов")
    plt.ylabel("точность")
    plt.title("Дерево решений(макс. глубина = 2)")
    ax = plt.gca()
    plt.show()
    xVal.clear()
    yVal.clear()
    for i in range(50):
        clf = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(),n_estimators=i + 1, random_state=6)
        clf.fit(X_train,y_train)
        xVal.append(i)
        yVal.append(clf.score(X_test, y_test))
    plt.plot(xVal,yVal)
    plt.xlabel("кол-во классификаторов")
    plt.ylabel("точность")
    plt.title("Дерево решений")
    ax = plt.gca()
    plt.show()

def task3():
    data = pd.read_csv('titanic_train.csv')
    data.head()
    dataset = pd.DataFrame(data)
    print(dataset)
    dataset = dataset.drop(dataset.columns[0], axis=1)
    dataset = dataset.drop(dataset.columns[2], axis=1)
    dataset = dataset.drop(dataset.columns[6], axis=1)
    dataset = dataset.drop(dataset.columns[7], axis=1)
    dataset = dataset.drop(index=0)
    dataset.isnull().sum()
    print(dataset)
    dataset.astype('object').describe(include='all').loc['unique', :]
    dataset = dataset.fillna({'Age': dataset['Age'].median(), 'Embarked': dataset['Embarked'].mode()})
    dataset = pd.get_dummies(dataset, columns=['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'])
    X = dataset.loc[:, dataset.columns != 'Survived']
    y = dataset.loc[:, dataset.columns == 'Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    estimators = [
        ('bais', GaussianNB()),
        ('knn', KNeighborsClassifier(n_neighbors=1)),
        ('dt', tree.DecisionTreeClassifier(max_depth = 5)),
        ('rf', RandomForestClassifier(n_estimators=10, random_state=0))
    ]
    clf = StackingClassifier(estimators=estimators, final_estimator=tree.DecisionTreeClassifier())
    clf.fit(X,y)
    print(clf.score(X_test,y_test))
