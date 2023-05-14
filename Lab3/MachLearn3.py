import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import math
from PIL import Image

def task1():
    df = pd.read_csv('pluton.csv')
    ans = {}
    for i in range(1, 101):
        kmeans = KMeans(n_clusters=3, max_iter=i).fit(df[['Pu238', 'Pu239', 'Pu240', 'Pu241']])
        labels = kmeans.labels_
        score = silhouette_score(df[['Pu238', 'Pu239', 'Pu240', 'Pu241']], labels) 
        ans[i] = score
    plt.plot(ans.keys(), ans.values(), color="blue", scaley=True)
    plt.xlabel("iter")
    plt.ylabel("score")
    plt.title("K-MEANS CLUSTERING Без стандартизации")
    plt.grid(True)
    plt.show()
    ans_std = {}
    for i in range(1, 101): scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[['Pu238', 'Pu239', 'Pu240', 'Pu241']])
    kmeans_scaled = KMeans(n_clusters=3, max_iter=100).fit(X_scaled)
    labels_scaled = kmeans_scaled.labels_
    score = silhouette_score(X_scaled, labels_scaled)
    ans_std[i] = score
    plt.plot(ans_std.keys(), ans_std.values(), color="blue", scaley=True)
    plt.xlabel("iter")
    plt.ylabel("score")
    plt.title("K-MEANS CLUSTERING Со стандартизацией")
    plt.grid(True)
    plt.show()

def KMEANCE(name, n_clusters):
    df = pd.read_csv(name)
    x = []
    y = []
    for e in df.values:
       x.append(e[0].split('\t')[0])
       y.append(e[0].split('\t')[1])
    X = np.array(x).reshape(-1, 1).astype('float32') 
    Y = np.array(y).reshape(-1, 1).astype('float32')
    points = np.concatenate((X, Y), axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(points)
    kmeans_scaled = KMeans(n_clusters=n_clusters, max_iter=100).fit(X_scaled)
    labels_scaled = kmeans_scaled.labels_
    plt.scatter(points[:, 0], points[:, 1], Y, c=labels_scaled) 
    plt.title("K-MEANS CLUSTERING")
    plt.grid(True)
    plt.show()

def DBSCAN(path):
    df = pd.read_csv(name) 
    x = []
    y = []
    for e in df.values:
       x.append(e[0].split('\t')[0])
       y.append(e[0].split('\t')[1])
    X = np.array(x).reshape(-1, 1).astype('float32') 
    Y = np.array(y).reshape(-1, 1).astype('float32') 
    points = np.concatenate((X, Y), axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(points)
    dbscan_scaled = DBSCAN(eps=0.2, min_samples=5).fit(X_scaled) 
    labels_scaled = dbscan_scaled.labels_
    plt.scatter(points[:, 0], points[:, 1], Y, c=labels_scaled) 
    plt.title("DBSCAN CLUSTERING")
    plt.grid(True)
    plt.show()

def AgglomerativeClustering(name, n_clusters):
    df = pd.read_csv(name)
    x = []
    y = []
    for e in df.values:
       x.append(e[0].split('\t')[0])
       y.append(e[0].split('\t')[1])
    X = np.array(x).reshape(-1, 1).astype('float32') 
    Y = np.array(y).reshape(-1, 1).astype('float32') 
    points = np.concatenate((X, Y), axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(points)
    dbscan_scaled = AgglomerativeClustering(n_clusters=n_clusters).fit(X_scaled)
    labels_scaled = dbscan_scaled.labels_
    plt.scatter(points[:, 0], points[:, 1], Y, c=labels_scaled)
    plt.title("Иерархическая CLUSTERING")
    plt.grid(True)
    plt.show()

def task2():
    KMEANCE("clustering_1.csv", 2)
    KMEANCE("clustering_2.csv", 3)
    KMEANCE("clustering_3.csv", 5)
    DBSCAN("clustering_1.csv") 
    DBSCAN("clustering_2.csv")
    DBSCAN("clustering_3.csv")
    AgglomerativeClustering("clustering_1.csv", 2)
    AgglomerativeClustering("clustering_2.csv", 3)
    AgglomerativeClustering("clustering_3.csv", 5)

def task3():
    img = Image.open('C:/Users/SHADOW/source/repos/machChusg/machChusg/test.jpg')
    r = []
    g = []
    b = []
    print()
    for x in range(0, img.size[0]):
        for y in range(0, img.size[1]):
           pixel = img.getpixel((x, y)) 
           r.append(pixel[0]) 
           g.append(pixel[1]) 
           b.append(pixel[2])
    R = np.array(r).reshape(-1, 1).astype('int32')
    G = np.array(g).reshape(-1, 1).astype('int32')
    B = np.array(b).reshape(-1, 1).astype('int32') 
    PIXELS = np.concatenate((R, G, B), axis=1)
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=5, max_iter=100).fit(PIXELS) 
    cluster_centers = kmeans.cluster_centers_ 
    cluster_centers = np.around(cluster_centers).astype(int)
    for x in range(0, img.size[0]):
        for y in range(0, img.size[1]):
            pixel = img.getpixel((x, y))
            d_min = math.sqrt((cluster_centers[0][0] - pixel[0]) ** 2 + (cluster_centers[0][1] - pixel[1]) ** 2 + (cluster_centers[0][2] - pixel[2])** 2)
            c_min = cluster_centers[0]
            for c in cluster_centers:
                d = math.sqrt((c[0] - pixel[0]) ** 2 + (c[1] - pixel[1]) ** 2 + (c[2] - pixel[2]) ** 2)
                if d < d_min:
                    d_min = d 
                    c_min = c
            img.putpixel((x, y), tuple(c_min))
    img.save('new5.jpeg')

def plot_dendrogram(model, **kwargs):
    np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
       current_count = 0
    for child_idx in merge:
        if child_idx < n_samples:
            current_count += 1
        else:
            current_count += counts[child_idx - n_samples] 
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

def task4():
    df = pd.read_csv('votes.csv')
    for column in df.columns:
        df[column] = df[column].fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(X_scaled) 
    plt.title('Dendrogram')
    plot_dendrogram(model, truncate_mode='level', p=3) 
    plt.show()








