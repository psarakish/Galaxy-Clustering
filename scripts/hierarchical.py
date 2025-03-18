from sklearn.cluster import AgglomerativeClustering
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore")

# Custom wrapper for AgglomerativeClustering
class WrappedAgglomerativeClustering(BaseEstimator, ClusterMixin):
    def __init__(self, full_distance_matrix, n_clusters=None, distance_threshold=None, affinity='precomputed', linkage='single'):
        self.full_distance_matrix = full_distance_matrix
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.affinity = affinity
        self.linkage = linkage

    def fit(self, X, y=None):
    # Check if X is the full distance matrix
        if X.shape == self.full_distance_matrix.shape and np.all(X == self.full_distance_matrix):
            subset_distance_matrix = X
        else:
            # X is expected to be indices in this case
            subset_distance_matrix = self.full_distance_matrix[X][:, X]

        self.model_ = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            distance_threshold=self.distance_threshold,
            affinity=self.affinity,
            linkage=self.linkage
    )
        self.labels_ = self.model_.fit_predict(subset_distance_matrix)
        return self

    def predict(self, X):
        # This is not a proper predict method but we need it for RandomizedSearchCV compatibility
        subset_distance_matrix = self.full_distance_matrix[X][:, X]
        return self.model_.fit_predict(subset_distance_matrix)


param_grid = {
    'distance_threshold': np.linspace(6.25,6.35 , num=5),
    'affinity': ['precomputed'],
    'linkage': ['single']
}

# Using train_dist_matrix in the estimator
search = RandomizedSearchCV(
    WrappedAgglomerativeClustering(full_distance_matrix=train_dist_matrix, n_clusters=None), 
    param_distributions=param_grid, 
    refit=True, 
    n_iter=2, 
    cv=3, 
    verbose=3, 
    scoring=score, 
    error_score='raise', 
    n_jobs=1, 
    return_train_score=True
)

search.fit(np.arange(train_dist_matrix.shape[0]), y_train)

best_threshold = search.best_params_['distance_threshold']
final_model = WrappedAgglomerativeClustering(full_distance_matrix=all_data_dmatrix, distance_threshold=best_threshold)

labels_pred = final_model.fit_predict(all_data_dmatrix)

num_clusters = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
print(f"labels: {labels_pred} ")
print(f"Number of clusters found: {num_clusters}")
print(f"Best threshold: {best_threshold}")

test_indices = X_test.index #Obtain the indices of test sample
labels_pred_test = labels_pred[test_indices] #Obtain the corresponding predicted for test
labels_pred_test

#Compute the scoring ! 
avg_scoring = scoring_metric(y_test, labels_pred_test)
print(f"Average Scoring: {avg_scoring}")
