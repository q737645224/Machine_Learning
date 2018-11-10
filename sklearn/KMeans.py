import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
sns.set()



def find_clusters(X, n_clusters, rseed=2):
    # 1. 随机选择中心点
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # 2a. 基于最近的中心指定标签
        labels = pairwise_distances_argmin(X, centers)

        # 2b. 根据点的平均值找到新的中心
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])

        # 2c. 确认收敛
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
# plt.scatter(X[:, 0], X[:, 1], s=50)

centers, labels = find_clusters(X, 4)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()