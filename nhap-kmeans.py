from __future__ import print_function 
import numpy as np
from sklearn.cluster import KMeans
N=20
X = np.array([[6.7, 5.7, 9.7],
[9.7, 9.7, 8.7],
[8.7, 3.7, 7.7],
[6.7, 4.7, 3.7],
[8.7, 5.7, 4.7],
[6.7, 5.7, 9.7],
[7.7, 8.7, 8.7],
[5.7, 4.7, 6.7],
[7.7, 3.7, 4.7],
[9.7, 9.7, 7.7],
[2.7, 5.7, 3.7],
[8.7, 8.7, 6.7],
[3.7, 9.7, 4.7],
[3.7, 9.7, 9.7],
[7.7, 6.7, 9.7],
[9.7, 7.7, 6.7],
[7.7, 4.7, 4.7],
[7.7, 9.7, 5.7],
[4.7, 9.7, 7.7],
[1.7, 3.7, 4.7]
])

K = 3
original_label = np.asarray([0]*N + [1]*N + [2]*N).T
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(X)
