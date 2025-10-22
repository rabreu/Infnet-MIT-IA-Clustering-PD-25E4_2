import math
import random

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


#metrica de similaridade: distancia euclidiana
def euclidean_distance(point1, point2):
  dist = math.sqrt(
      sum(
         (p1 - p2) ** 2  for p1, p2 in zip(point1, point2)
      )
  )

  return dist

def initialize_centroids(data:list, k):
  data_list = data.tolist()

  centroids = random.sample(data_list, k)

  return centroids

def assign_clusters(data, centroids):

  clusters = [[] for _ in range(len(centroids))]

  for point in data:
    distances = [euclidean_distance(point, centroid) for centroid in centroids]
    cluster_index = distances.index(min(distances))
    clusters[cluster_index].append(point)

  return clusters

def update_centroids(clusters, old_clusters):

  new_centroids = list()

  for i, cluster in enumerate(clusters):
    if cluster:
      new_centroid = [sum(dim)/len(cluster) for dim in zip(*cluster)]
    else:
      new_centroid = old_clusters[i]

    new_centroids.append(new_centroid)

  return new_centroids

def k_means_from_scracth(data, k, max_iterations=100):

  centroids = initialize_centroids(data, k)

  for _ in range(max_iterations):

    clusters = assign_clusters(data, centroids)
    new_centroids = update_centroids(clusters, centroids)

    centroids = new_centroids

  return centroids, clusters

X, labels = make_blobs(n_samples=500, n_features=2, centers=3, random_state=12)


X_scaled = StandardScaler().fit_transform(X)

sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=labels, palette='viridis')
plt.show()

centroids, clusters = k_means_from_scracth(X_scaled, 3)
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=labels, palette='viridis')

for centroid in centroids:
  plt.scatter(centroid[0], centroid[1], marker='x', color='red')

plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

print(kmeans.cluster_centers_)

f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4))

# kmeans from scratch
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=labels, palette='viridis', ax=ax[0])
ax[0].scatter(
    [x for x, _ in centroids],
    [y for _, y in centroids],
    marker='X',
    color='red'
)
ax[0].set_title('K-Means from scratch', fontsize=16)


# sklearn
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=labels, palette='viridis', ax=ax[1])
ax[1].scatter(
    [x for x, _ in kmeans.cluster_centers_],
    [y for _, y in kmeans.cluster_centers_],
    marker='X',
    color='red'
)
ax[1].set_title('K-Means Scikit-Learn', fontsize=16)

plt.show()