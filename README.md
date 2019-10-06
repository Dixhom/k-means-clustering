# k-means clustering
A simple implementation of k-means clustering.

# Install
`pip install git+https://github.com/Dixhom/k-means-clustering`

# Usage
```python
from k_means_clustering.k_means_clustering import KMeansClustering
from sklearn.datasets import make_classification

dat = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, 
                          n_classes=3, class_sep=2, n_clusters_per_class=1, random_state=0)
X = dat[0]
y = dat[1]

# n_classes is the number of centroids
# max_iter is the maximum number of iteration to update centroids
kmeans = KMeansClustering(n_classes=3, max_iter=10000)
kmeans.fit(X,y)
print(kmeans.centroids)
```

# Licence
MIT

# Appendix
An experimental code is in `prototype` directory.
