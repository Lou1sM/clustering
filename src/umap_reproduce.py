from pdb import set_trace
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils

# Dimension reduction and clustering libraries
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

sns.set(style='white', rc={'figure.figsize':(10,8)})
#mnist = fetch_mldata('MNIST Original')
mnist_data, mnist_labels = fetch_openml('mnist_784', version=1, return_X_y=True)
mnist_labels = mnist_labels.astype(np.float)

standard_embedding = umap.UMAP(random_state=42).fit_transform(mnist_data)
#standard_embedding = np.random.normal(size=(70000,2))
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=mnist_labels, s=0.1, cmap='Spectral')
plt.show()

kmeans_labels = cluster.KMeans(n_clusters=10).fit_predict(mnist_data)
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=kmeans_labels, s=0.1, cmap='Spectral')
plt.show()

print(adjusted_rand_score(mnist_labels, kmeans_labels),adjusted_mutual_info_score(mnist_labels, kmeans_labels))

lowd_mnist = PCA(n_components=50).fit_transform(mnist_data)
hdbscan_labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500).fit_predict(lowd_mnist)

#clusterable_embedding = np.random.normal(size=(70000,2))
clusterable_embedding = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42).fit_transform(mnist_data)
plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], c=mnist_labels, s=0.1, cmap='Spectral');
plt.show()
hdbscan_labels = hdbscan.HDBSCAN(min_samples=10,min_cluster_size=500,).fit_predict(clusterable_embedding)

print(adjusted_rand_score(mnist_labels, hdbscan_labels), adjusted_mutual_info_score(mnist_labels, hdbscan_labels))

"""

clustered = (hdbscan_labels >= 0)
plt.scatter(standard_embedding[~clustered, 0],
            standard_embedding[~clustered, 1],
            c=(0.5, 0.5, 0.5),
            s=0.1,
            alpha=0.5)
plt.scatter(standard_embedding[clustered, 0],
            standard_embedding[clustered, 1],
            c=hdbscan_labels[clustered],
            s=0.1,
            cmap='Spectral')


plt.scatter(standard_embedding[~clustered, 0],
            standard_embedding[~clustered, 1],
            c=(0.5, 0.5, 0.5),
            s=0.1,
            alpha=0.5)
plt.scatter(standard_embedding[clustered, 0],
            standard_embedding[clustered, 1],
            c=labels[clustered],
            s=0.1,
            cmap='Spectral')
plt.show()

(
    adjusted_rand_score(mnist.target, hdbscan_labels),
    adjusted_mutual_info_score(mnist.target, hdbscan_labels)
)

np.sum(clustered) / mnist.data.shape[0]
"""


