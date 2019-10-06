import numpy as np

class KMeansClustering:
    def __init__(self, n_classes, max_iter=10000):
        self.n_classes = n_classes
        self.max_iter = max_iter
    
    def __get_random_centroids(self, X):
        # centroids shouldn't be too far from observations
        mx = np.apply_along_axis(lambda x: max(x), 0, X)
        mn = np.apply_along_axis(lambda x: min(x), 0, X)
        rnd = np.random.rand(self.n_classes)
        return (mx - mn) * rnd.reshape(-1,1) + mn        

    def __distances_from_point(self, X, p):
        # square of euclidian distance
        return np.apply_along_axis(lambda x: sum((x-p)*(x-p)), 1, X)    
    
    def fit(self, X, y):
        centroids = self.__get_random_centroids(X)
        self.centroid_history = []
        next_centroids = np.zeros_like(centroids)
        self.class_history = []
        n_iter = 0

        # if the iteration reaches max iter, stop
        while True and n_iter < self.max_iter:
            # save centroid coordinates
            self.centroid_history.append(centroids)
            # distances of each data point from each centroid
            dists = np.apply_along_axis(lambda p: self.__distances_from_point(X, p), 1, centroids)
            # the class each data point belongs to
            class_ = np.apply_along_axis(lambda x: np.argmin(x), 0, dists)
            # save class 
            self.class_history.append(class_)
            # update centroids
            next_centroids = np.array([np.average(X[class_==i], axis=0) for i in range(self.n_classes)])
            # if the centroids don't change, break
            if np.all(next_centroids == centroids): break
            centroids = next_centroids
            n_iter += 1
            
        self.centroids = centroids
        self.class_ = self.class_history[-1]