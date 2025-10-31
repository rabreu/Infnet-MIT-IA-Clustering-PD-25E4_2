from idlelib.macosx import overrideRootMenu
from math import dist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from math import dist
from sklearn.metrics import pairwise_distances
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris

dados = {
  'estudante' : ['Gabriela', 'Luiz Felipe', 'Patricia', 'Ovidio', 'Leonor'],
  'matematica' : [3.7, 7.8, 8.9, 7.0, 3.4],
  'fisica': [2.7, 8.0, 1.0, 1.0, 2.0],
  'quimica': [9.1, 1.5, 2.7, 9.0, 5.0]
}

df = pd.DataFrame(dados)
print(df)

px.scatter_3d(df, x='matematica', y='fisica', z='quimica', text='estudante').show()

notas_matematica = df.iloc[:, 1].values #df['matematica'].values
notas_fisica = df.iloc[:, 2].values
notas_quimica = df.iloc[:,3].values

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(notas_matematica, notas_fisica, notas_quimica)
ax.set_xlabel("matematica")
ax.set_ylabel("fisica")
ax.set_zlabel("quimica")

plt.show()

distance_matrix_0 = []

for i in range(df.shape[0]): # get rows
    distances = []
    for j in range(df.shape[0]): # get rows again
        if i > j: # compare with themselves
            distance = dist(df.iloc[i,1:], df.iloc[j,1:])
        else:
            distance = 0
        distances.append(round(distance,3))
    distance_matrix_0.append(distances)

df_matrix_distance_0 = pd.DataFrame(distance_matrix_0)

print(df_matrix_distance_0)

pw_distance = pairwise_distances(df.iloc[:,1:], metric='euclidean')
print(pw_distance)

single_linkage = hierarchy.linkage(
    df.iloc[:, 1:],
    method='single',
    metric='euclidean'
)
print(single_linkage)

f, ax = plt.subplots()
ax.autoscale(tight=False)

hierarchy.dendrogram(
    single_linkage,
    labels=df['estudante'].values,
    ax=ax,
    orientation='right',
    above_threshold_color='red',
    distance_sort=True,
    color_threshold=0.5
)

sns.despine(f, ax)
plt.show()

cluster_single_linkage = AgglomerativeClustering(n_clusters=3, linkage='single', metric='euclidean')
cluster_single_linkage.fit(df.iloc[:, 1:])
df['cluster_single_linkage'] = cluster_single_linkage.labels_

px.scatter_3d(df, x='matematica', y='fisica', z='quimica', color='cluster_single_linkage', text='estudante').show()


f, ax = plt.subplots(1, 2, figsize=(12, 4))

hierarchy.dendrogram(
    single_linkage,
    labels=df['estudante'].values,
    distance_sort=True,
    above_threshold_color='red',
    ax=ax[0],

)

complete_linkage = hierarchy.linkage(
    df.iloc[:, 1:],
    method='complete',
    metric='euclidean'
)

hierarchy.dendrogram(
    complete_linkage,
    labels=df['estudante'].values,
    distance_sort=True,
    above_threshold_color='red',
    ax=ax[1],

)

ax[0].set_title('Single Linkage')
ax[1].set_title('Complete Linkage')

sns.despine()
plt.show()

load_iris = load_iris()

iris_df = pd.DataFrame(load_iris.data, columns=load_iris.feature_names)
iris_df['target'] = load_iris.target
iris_df['target'] = iris_df['target'].apply(lambda x: load_iris.target_names[x])

iris_single_linkage = hierarchy.linkage(
    iris_df.iloc[:, :-1],
    method='single',
    metric='euclidean'
)

f, ax = plt.subplots(figsize=(12, 4))

hierarchy.dendrogram(
    iris_single_linkage,
    above_threshold_color='red',
    labels=iris_df['target'].values,
    distance_sort=True,
    color_threshold=0.5
)

sns.despine()
plt.show()