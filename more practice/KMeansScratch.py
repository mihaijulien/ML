import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

#find the closest centroid for each instance in the data
def find_closest_centroids(X, centroids):
    X_rows = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(X_rows)
    
    for i in range(X_rows):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i,:] - centroids[j,:]) ** 2) # euclidian distance between two points
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    
    return idx

data = loadmat('data/ex7data2.mat')
X = data['X']

print('Points: ', X)

#three random centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

idx = find_closest_centroids(X, initial_centroids)
print('Assign each point to the closest centroid: ')
print(idx[0:3])

# Compute the centroid of a cluster. The centroid is simply the
# mean of all of the examples currently assigned to the cluster.
def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    
    for i in range(k):
        indices = np.where(idx == i)
        centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()
    
    return centroids

print('Compute new centroids for each cluster: ')
print(compute_centroids(X, idx, 3))

def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids
    
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
    
    return idx, centroids

idx, centroids = run_k_means(X, initial_centroids, 10)

cluster1 = X[np.where(idx == 0)[0],:]
cluster2 = X[np.where(idx == 1)[0],:]
cluster3 = X[np.where(idx == 2)[0],:]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')
ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
ax.legend()
plt.show()

