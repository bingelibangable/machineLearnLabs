import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn import tree
import graphviz

def task1():
    dataset = pd.read_csv('tic_tac_toe.txt', header = None)
    df = pd.DataFrame(dataset)
    dataset = pd.read_csv('spam.csv', header = None)
    spamDf = pd.DataFrame(dataset)
    print(spamDf)
    X = (df[df.columns[[0,1,2,3,4,5,6,7,8]]]).to_numpy()
    y = df[9]
    print(X)
    for i in range(len(X)):
        for k in range(len(X[i])):
            if (X[i][k] == 'o'):
                X[i][k] = 0
            if (X[i][k] == 'x'):
                X[i][k] = 1
            if (X[i][k] == 'b'):
                X[i][k] = 2
    print(X)
    print(y)
    yVal = list()
    xVal = list()
    i = 1
    while (i < 10):
        xVal.append(i * 0.1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i * 0.1, random_state=0)
        print(X_train)
        y_test = y_test.to_numpy()
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)
        yVal.append(gnb.score(X_test, y_test))
        i = i + 1
    plt.plot(xVal,yVal)
    plt.xlabel("относительный объем тестовой выборки")
    plt.ylabel("точность")
    ax = plt.gca()
    ax.set_ylim([0.6, 0.75])
    plt.show()

    col = list()
    for i in range(57):
        col.append(i + 1)
    X = (spamDf[spamDf.columns[col]]).to_numpy()
    X = np.delete(X, (0), axis=0)
    y = spamDf[58]
    y = np.delete(y, (0), axis=0)
    print(X)
    print(y)
    print(len(X))
    print(len(y))
    i = 1
    xVal.clear()
    yVal.clear()
    while (i < 10):
        xVal.append(i * 0.1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i * 0.1, random_state=0)
        gnb2 = GaussianNB()
        gnb2.fit(X_train, y_train)
        y_pred = gnb2.predict(X_test)
        print(gnb2.score(X_test, y_test))
        yVal.append(gnb2.score(X_test, y_test))
        i = i + 1
    plt.plot(xVal,yVal)
    plt.xlabel("относительный объем тестовой выборки")
    plt.ylabel("точность")
    ax = plt.gca()
    ax.set_ylim([0.75, 0.85])
    plt.show()

#2
def task2():
    listX1 = list()
    listX2 = list()
    listClass1 = list()
    listClass2 = list()
    Mx1_1 = 17
    Mx1_2 = 10
    S1 = 3
    Mx2_1 = 15
    Mx2_2 = 8
    S2 = 5
    class1 = (0, 1, 0)
    class2 = (1, 0, 0)
    for i in range(60):
        listX1.append([np.random.normal(Mx1_1, S1), np.random.normal(Mx1_2, S1)])
        listClass1.append(-1)
    for i in range(40):
        listX2.append([np.random.normal(Mx2_1, S2), np.random.normal(Mx2_2, S2)])
        listClass2.append(1)
    listX1 = np.array(listX1)
    listX2 = np.array(listX2)
    listClass1 = np.array(listClass1)
    listClass2 = np.array(listClass2)
    print(listX1)
    print(listX2)
    print(listClass1)
    print(listClass2)
    col = list()
    x = list()
    y = list()
    for i in range(60):
        x.append(listX1[i][0])
    for i in range(60):
        y.append(listX1[i][1])
        col.append(class1)
    for i in range(40):
        x.append(listX2[i][0])
    for i in range(40):
        y.append(listX2[i][1])
        col.append(class2)
    plt.scatter(x,y,c=col,s=6)
    plt.xlabel('Mx1')
    plt.ylabel('Mx2')
    plt.show()
    X = np.concatenate((listX1, listX2))
    y = np.concatenate((listClass1, listClass2))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    yVal = gnb.score(X_test, y_test)
    pred = gnb.predict_proba(X_test)
    y_pred = gnb.predict(X_test)
    print(pred)
    print(yVal)

    print(confusion_matrix(y_test,y_pred))#true = rows, pred = columns
    RocCurveDisplay.from_predictions(
        y_test,
        pred[:,1],
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()
    precision, recall, thresholds = precision_recall_curve(
        y_test, pred[:,1])
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
#3
def task3():
    dataset = pd.read_csv('glass.csv', header = None)
    df = pd.DataFrame(dataset)
    print(df)
    df=df.drop(df.columns[0], axis=1)
    print(df)
    X = (df[df.columns[[0,1,2,3,4,5,6,7,8]]]).to_numpy()
    X = np.delete(X, (0), axis=0)
    y = df[10]
    y = np.delete(y, (0), axis=0)
    print(X)
    print(y)
    y_val = list()
    x_val = list()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    for i in range(14):
        neigh = KNeighborsClassifier(n_neighbors=i + 1)
        neigh.fit(X_train,y_train)
        predScore = neigh.score(X_test, y_test)
        y_val.append(predScore)
        x_val.append(i + 1)
    plt.plot(x_val,y_val)
    plt.xlabel("количество соседей")
    plt.ylabel("точность")
    plt.show()
    for i in range(7):
        print('количество соседей: ' + (str)(i + 1))
        neigh = KNeighborsClassifier(n_neighbors=i + 1, metric = 'cityblock')
        neigh.fit(X_train,y_train)
        print('точность с расстоянием городских кварталов = ' + (str)(neigh.score(X_test, y_test)))
        neigh = KNeighborsClassifier(n_neighbors=i + 1, metric = 'cosine')
        neigh.fit(X_train,y_train)
        print('точность с косинусновым расстояни = ' + (str)(neigh.score(X_test, y_test)))
        neigh = KNeighborsClassifier(n_neighbors=i + 1, metric = 'euclidean')
        neigh.fit(X_train,y_train)
        print('точность с евклидовой метрикой = ' + (str)(neigh.score(X_test, y_test)))
        neigh = KNeighborsClassifier(n_neighbors=i + 1, metric = 'manhattan')
        neigh.fit(X_train,y_train)
        print('точность с манхэттенским расстоянием = ' + (str)(neigh.score(X_test, y_test)))
    neigh = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean')
    neigh.fit(X,y)
    result = neigh.predict([[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]])
    print(result)
#4
def task4_1():
    dataset = pd.read_csv('svmdata_a.txt', header = None, sep = '\t|\n')
    dfTest = pd.DataFrame(pd.read_csv('svmdata_a_test.txt', header = None, sep = '\t|\n'))
    df = pd.DataFrame(dataset)
    print(dataset)
    print(dfTest)
    df = df.drop(df.columns[0], axis=1)
    print(df)
    X = (df[df.columns[[0,1]]]).to_numpy()
    X = np.delete(X, (0), axis=0)
    y = df[3]
    y = np.delete(y, (0), axis=0)
    dfTest = dfTest.drop(dfTest.columns[0], axis=1)
    X_fileTest = (dfTest[dfTest.columns[[0,1]]]).to_numpy()
    X_fileTest = np.delete(X_fileTest, (0), axis=0)
    y_fileTest = dfTest[3]
    y_fileTest = np.delete(y_fileTest, (0), axis=0)
    print(X)
    print(y)
    X = X.astype(float)
    print(X_fileTest)
    print(y_fileTest)
    clf = SVC(kernel = 'linear')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    clf.fit(X_train,y_train)
    print('Number of vectors = ' + (str)(len(clf.support_vectors_)))
    y_pred = clf.predict(X_test)

    print(confusion_matrix(y_test,y_pred))#true = rows, pred = columns
    y_pred = clf.predict(X_fileTest)
    print(confusion_matrix(y_fileTest,y_pred))#true = rows, pred = columns
    disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            response_method="predict",
            cmap=ListedColormap(["Green","Red"]),
            alpha=0.8,
        )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    plt.show()

def task4_2():
    dataset = pd.read_csv('svmdata_b.txt', header = None, sep = '\t|\n')
    dfTest = pd.DataFrame(pd.read_csv('svmdata_b_test.txt', header = None, sep = '\t|\n'))
    df = pd.DataFrame(dataset)
    #print(dataset)
    #print(dfTest)
    df = df.drop(df.columns[0], axis=1)
    #print(df)
    X = (df[df.columns[[0,1]]]).to_numpy()
    X = np.delete(X, (0), axis=0)
    y = df[3]
    y = np.delete(y, (0), axis=0)
    dfTest = dfTest.drop(dfTest.columns[0], axis=1)
    X_fileTest = (dfTest[dfTest.columns[[0,1]]]).to_numpy()
    X_fileTest = np.delete(X_fileTest, (0), axis=0)
    y_fileTest = dfTest[3]
    y_fileTest = np.delete(y_fileTest, (0), axis=0)
    print(X)
    print(y)
    X = X.astype(float)
    X_fileTest = X_fileTest.astype(float)
    #print(X_fileTest)
    #print(y_fileTest)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = SVC(kernel = 'linear')
    clf.fit(X,y)
    y_pred = clf.predict(X)
    print(confusion_matrix(y,y_pred)[0,1])#true = rows, pred = columns
    #y_pred = clf.predict(X_fileTest)
    #print(confusion_matrix(y_fileTest,y_pred))#true = rows, pred = columns
    disp = DecisionBoundaryDisplay.from_estimator(
                clf,
                X,
                response_method="predict",
                cmap=ListedColormap(["Green","Red"]),
                alpha=0.8,
            )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    plt.show()
    #i = 0
    print(confusion_matrix(y,y_pred))
    #while (not (confusion_matrix(y,y_pred)[0,1] == 0 and confusion_matrix(y,y_pred)[1,0] == 0)):
       # i = i + 1
        #clf = SVC(kernel = 'linear', C = i)
       # clf.fit(X,y)
       # y_pred = clf.predict(X)
        #print(confusion_matrix(y,y_pred)[0,1])#true = rows, pred = columns
        #y_pred = clf.predict(X_fileTest)
        #print(confusion_matrix(y_fileTest,y_pred))#true = rows, pred = columns
    #print(i)
    clf = SVC(kernel = 'linear', C = 483)
    clf.fit(X,y)
    y_pred = clf.predict(X)
    print(confusion_matrix(y,y_pred))
    disp = DecisionBoundaryDisplay.from_estimator(
                clf,
                X,
                response_method="predict",
                cmap=ListedColormap(["Green","Red"]),
                alpha=0.8,
            )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    plt.show()
    #коэффициент для тестовой выборки был найден грубым способ, потому что одна из красных точек находится рядом с зелеными, поэтому он такой большой, однако это не будет значить, что класификатор будет работать точно, так как судя по расположению зеленых точек, они могут нахожится за пределами зеленой области
    clf = SVC(kernel = 'linear')
    clf.fit(X,y)
    y_pred = clf.predict(X_fileTest)
    print(confusion_matrix(y_fileTest,y_pred))#true = rows, pred = columns
    disp = DecisionBoundaryDisplay.from_estimator(
                clf,
                X_fileTest,
                response_method="predict",
                cmap=ListedColormap(["Green","Red"]),
                alpha=0.8,
            )
    disp.ax_.scatter(X_fileTest[:, 0], X_fileTest[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    plt.show()

def gaussianKernelGramMatrixFull(X1, X2, sigma=0.1):
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.flatten()
            x2 = x2.flatten()
            gram_matrix[i, j] = np.exp(- np.sum( np.power((x1 - x2),2) ) / float( 2*(sigma**2) ) )
    return gram_matrix

def gaussian_kernel(x, y, sigma=1):
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(- (np.linalg.norm(x - y, 2)) ** 2 / (2 * sigma ** 2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(- (np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(- (np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))
    return result

def task4_3():
    dataset = pd.read_csv('svmdata_c.txt', header = None, sep = '\t|\n')
    dfTest = pd.DataFrame(pd.read_csv('svmdata_c_test.txt', header = None, sep = '\t|\n'))
    df = pd.DataFrame(dataset)
    #print(dataset)
    #print(dfTest)
    df = df.drop(df.columns[0], axis=1)
    #print(df)
    X = (df[df.columns[[0,1]]]).to_numpy()
    X = np.delete(X, (0), axis=0)
    y = df[3]
    y = np.delete(y, (0), axis=0)
    dfTest = dfTest.drop(dfTest.columns[0], axis=1)
    X_fileTest = (dfTest[dfTest.columns[[0,1]]]).to_numpy()
    X_fileTest = np.delete(X_fileTest, (0), axis=0)
    y_fileTest = dfTest[3]
    y_fileTest = np.delete(y_fileTest, (0), axis=0)
    print(X)
    print(y)
    X = X.astype(float)
    X_fileTest = X_fileTest.astype(float)
    clf = SVC(kernel = 'linear')
    clf.fit(X,y)
    disp = DecisionBoundaryDisplay.from_estimator(
                clf,
                X_fileTest,
                response_method="predict",
                cmap=ListedColormap(["Green","Red"]),
                alpha=0.8,
            )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    plt.title("Линейное")
    plt.show()
    for i in range(5):
        clf = SVC(kernel = 'poly', degree = i + 1)
        clf.fit(X,y)
        disp = DecisionBoundaryDisplay.from_estimator(
                    clf,
                    X_fileTest,
                    response_method="predict",
                    cmap=ListedColormap(["Green","Red"]),
                    alpha=0.8,
                )
        disp.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        plt.title("Полиноминальное " + (str)(i + 1) + " степени")
        plt.show()
    clf = SVC(kernel = 'sigmoid')
    clf.fit(X,y)
    disp = DecisionBoundaryDisplay.from_estimator(
                clf,
                X_fileTest,
                response_method="predict",
                cmap=ListedColormap(["Green","Red"]),
                alpha=0.8,
            )
    disp.ax_.scatter(X[:, 0],X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    plt.title("Сигмоидальное")
    plt.show()
    clf = svm.SVC(C = 1, kernel='rbf')
    model = clf.fit( X, y )
    disp = DecisionBoundaryDisplay.from_estimator(
                model,
                X_fileTest,
                response_method="predict",
                cmap=ListedColormap(["Green","Red"]),
                alpha=0.8,
            )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    plt.title("Гауссово")
    plt.show()
#task4_3()
def task4_4():
    dataset = pd.read_csv('svmdata_d.txt', header = None, sep = '\t|\n')
    dfTest = pd.DataFrame(pd.read_csv('svmdata_d_test.txt', header = None, sep = '\t|\n'))
    df = pd.DataFrame(dataset)
    #print(dataset)
    #print(dfTest)
    df = df.drop(df.columns[0], axis=1)
    #print(df)
    X = (df[df.columns[[0,1]]]).to_numpy()
    X = np.delete(X, (0), axis=0)
    y = df[3]
    y = np.delete(y, (0), axis=0)
    dfTest = dfTest.drop(dfTest.columns[0], axis=1)
    X_fileTest = (dfTest[dfTest.columns[[0,1]]]).to_numpy()
    X_fileTest = np.delete(X_fileTest, (0), axis=0)
    y_fileTest = dfTest[3]
    y_fileTest = np.delete(y_fileTest, (0), axis=0)
    print(X)
    print(y)
    X = X.astype(float)
    X_fileTest = X_fileTest.astype(float)
    for i in range(5):
        clf = SVC(kernel = 'poly', degree = i + 1)
        clf.fit(X,y)
        disp = DecisionBoundaryDisplay.from_estimator(
                    clf,
                    X_fileTest,
                    response_method="predict",
                    cmap=ListedColormap(["Green","Red"]),
                    alpha=0.8,
                )
        disp.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        plt.title("Полиноминальное " + (str)(i + 1) + " степени")
        plt.show()
    clf = SVC(kernel = 'sigmoid')
    clf.fit(X,y)
    disp = DecisionBoundaryDisplay.from_estimator(
                clf,
                X_fileTest,
                response_method="predict",
                cmap=ListedColormap(["Green","Red"]),
                alpha=0.8,
            )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    plt.title("Сигмоидальное")
    plt.show()
    clf = svm.SVC(C = 1, kernel='rbf')
    model = clf.fit( X, y )
    disp = DecisionBoundaryDisplay.from_estimator(
                model,
                X_fileTest,
                response_method="predict",
                cmap=ListedColormap(["Green","Red"]),
                alpha=0.8,
            )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    plt.title("Гауссово")
    plt.show()
def task4_5():
    dataset = pd.read_csv('svmdata_e.txt', header = None, sep = '\t|\n')
    dfTest = pd.DataFrame(pd.read_csv('svmdata_e_test.txt', header = None, sep = '\t|\n'))
    df = pd.DataFrame(dataset)
    #print(dataset)
    #print(dfTest)
    df = df.drop(df.columns[0], axis=1)
    #print(df)
    X = (df[df.columns[[0,1]]]).to_numpy()
    X = np.delete(X, (0), axis=0)
    y = df[3]
    y = np.delete(y, (0), axis=0)
    dfTest = dfTest.drop(dfTest.columns[0], axis=1)
    X_fileTest = (dfTest[dfTest.columns[[0,1]]]).to_numpy()
    X_fileTest = np.delete(X_fileTest, (0), axis=0)
    y_fileTest = dfTest[3]
    y_fileTest = np.delete(y_fileTest, (0), axis=0)
    print(X)
    print(y)
    X = X.astype(float)
    X_fileTest = X_fileTest.astype(float)
    gam = 10
    gamm = 100
    for i in range(5):
        clf = SVC(kernel = 'poly', degree = i + 1, gamma = gam)
        clf.fit(X,y)
        disp = DecisionBoundaryDisplay.from_estimator(
                    clf,
                    X_fileTest,
                    response_method="predict",
                    cmap=ListedColormap(["Green","Red"]),
                    alpha=0.8,
                )
        disp.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        plt.title("Полиноминальное " + (str)(i + 1) + " степени. Гамма = " + (str)(gam))
        plt.show()
    for i in range(5):
        clf = SVC(kernel = 'poly', degree = i + 1, gamma=20)
        clf.fit(X,y)
        disp = DecisionBoundaryDisplay.from_estimator(
                    clf,
                    X_fileTest,
                    response_method="predict",
                    cmap=ListedColormap(["Green","Red"]),
                    alpha=0.8,
                )
        disp.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        plt.title("Полиноминальное " + (str)(i + 1) + " степени. Гамма = " + (str)(gamm))
        plt.show()
    clf = SVC(kernel = 'sigmoid', gamma=gam)
    clf.fit(X,y)
    disp = DecisionBoundaryDisplay.from_estimator(
                clf,
                X_fileTest,
                response_method="predict",
                cmap=ListedColormap(["Green","Red"]),
                alpha=0.8,
            )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    plt.title("Сигмоидальное. Гамма = " + (str)(gam))
    plt.show()
    clf = SVC(kernel = 'sigmoid', gamma=gamm)
    clf.fit(X,y)
    disp = DecisionBoundaryDisplay.from_estimator(
                clf,
                X_fileTest,
                response_method="predict",
                cmap=ListedColormap(["Green","Red"]),
                alpha=0.8,
            )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    plt.title("Сигмоидальное. Гамма = " + (str)(gamm))
    plt.show()
    clf = svm.SVC(C = 1, kernel='rbf', gamma=gam)
    model = clf.fit( X, y )
    disp = DecisionBoundaryDisplay.from_estimator(
                model,
                X_fileTest,
                response_method="predict",
                cmap=ListedColormap(["Green","Red"]),
                alpha=0.8,
            )
    disp.ax_.scatter(X_fileTest[:, 0], X_fileTest[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    plt.title("Гауссово. Гамма = " + (str)(gam))
    plt.show()
    clf = svm.SVC(C = 1, kernel='rbf', gamma=gamm)
    model = clf.fit( X, y )
    disp = DecisionBoundaryDisplay.from_estimator(
                model,
                X_fileTest,
                response_method="predict",
                cmap=ListedColormap(["Green","Red"]),
                alpha=0.8,
            )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    plt.title("Гауссово. Гамма = " + (str)(gamm))
    plt.show()
def task5_1():
    dataset = pd.read_csv('glass.csv', header = None)
    df = pd.DataFrame(dataset)
    df=df.drop(df.columns[0], axis=1)
    X = (df[df.columns[[0,1,2,3,4,5,6,7,8]]]).to_numpy()
    X = np.delete(X, (0), axis=0)
    y = df[10]
    y = np.delete(y, (0), axis=0)
    print(y)
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"],  
                         class_names=["1","2","3","5","6","7"],  
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = graphviz.Source(dot_data)
    graph.render("glass", view=True)
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf = clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"],  
                         class_names=["1","2","3","5","6","7"],  
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = graphviz.Source(dot_data)
    graph.render("glassdepth", view=True)
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"],  
                         class_names=["1","2","3","5","6","7"],  
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = graphviz.Source(dot_data)
    graph.render("glassEntropy", view=True)
    clf = tree.DecisionTreeClassifier(splitter='random')
    clf = clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"],  
                         class_names=["1","2","3","5","6","7"],  
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = graphviz.Source(dot_data)
    graph.render("glassrandom", view=True)
    clf = tree.DecisionTreeClassifier(max_depth=9, criterion='entropy', splitter='random')
    clf = clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"],  
                         class_names=["1","2","3","5","6","7"],  
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = graphviz.Source(dot_data)
    graph.render("glassOptimal", view=True)
#gini - chance of error, value - пометка, сколько объектов попало в условие на данном узле

def task5_2():
    dataset = pd.read_csv('spam7.csv', header = None)
    df = pd.DataFrame(dataset)
    print(df)
    #df=df.drop(df.columns[0], axis=1)
    X = (df[df.columns[[0,1,2,3,4,5]]]).to_numpy()
    X = np.delete(X, (0), axis=0)
    y = df[6]
    y = np.delete(y, (0), axis=0)
    print(y)
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = tree.DecisionTreeClassifier(min_impurity_decrease=0.0015)#
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    print(confusion_matrix(y_test,y_pred))
    dot_data = tree.export_graphviz(clf, out_file=None, 
                            feature_names=["crl.tot","dollar","bang","money","n000","make"],  
                            class_names=["yes","no"],  
                            filled=True, rounded=True,  
                            special_characters=True)  
    graph = graphviz.Source(dot_data)
    graph.render("spam7", view=True)

def task6():
    dataset = pd.read_csv('bank_scoring_train.csv', header = None, sep = '\t|\n')
    df = pd.DataFrame(dataset)
    print(len(df.index))
    #df=df.drop(df.columns[0], axis=1)
    X = (df[df.columns[[1,2,3,4,5,6,7,8,9,10]]]).to_numpy()
    X = np.delete(X, (0), axis=0)
    y = df[0]
    y = np.delete(y, (0), axis=0)
    dataset = pd.read_csv('bank_scoring_test.csv', header = None, sep = '\t|\n')
    df = pd.DataFrame(dataset)
    print(len(df.index))
    X_test = (df[df.columns[[1,2,3,4,5,6,7,8,9,10]]]).to_numpy()
    X_test = np.delete(X_test, (0), axis=0)
    y_test = df[0]
    y_test = np.delete(y_test, (0), axis=0)
    print(y)
    print(len(y_test))
    print(X)
    y = y.astype(float)
    y_test = y_test.astype(float)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    score = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    pred = clf.predict_proba(X_test)
    print(score)
    print(confusion_matrix(y_test,y_pred))
    RocCurveDisplay.from_predictions(
            y_test,
            pred[:,1],
            color="darkorange",
        )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()
    gnb = GaussianNB()
    gnb.fit(X, y)
    y_pred = gnb.predict(X_test)
    score = gnb.score(X_test, y_test)
    pred = gnb.predict_proba(X_test)
    print(score)
    print(confusion_matrix(y_test,y_pred))
    RocCurveDisplay.from_predictions(
            y_test,
            pred[:,1],
            color="darkorange",
        )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()
task6()

"""
Even though the ratio behind the two criteria is very similar as well as their results, there are some elements to note, especially when we are dealing with large dataset
(different from the Iris one, often used for demo purposes).
Namely, the Gini Index is less computationally intensive compared to the Information Gain criterion, since it has no logaritmic functions to compute.
Furthermore, the Information Gain criterion tends to privilege a high number of small nodes, with the risk of growing huge trees which lead to overfitting 
(however, you can bypass this problem by relying on a modification of the Information Gain, called Gain Index, which penalizes trees with large number of partitions).

Let’s say you have hundreds of features, then “best” splitter would be ideal because it will calculate the best features to split based on the impurity measure and use that to split the nodes, 
whereas if you choose “random” you have a high chance of ending up with features that don’t really give you that much information, which would lead to a more deeper less precise tree.
On the other hand, the “random” splitter has some advantages, specifically, since it selects a set of features randomly and splits, it doesn’t have the computational overhead of computing the optimal split.
Next, it is also less prone to overfitting because you are not essentially calculating the best split before each split and the additional randomness will help you here, so if your model is overfitting, 
then you can change the splitter to “random” and retrain.
"""