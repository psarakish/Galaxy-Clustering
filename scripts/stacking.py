import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from astropy.coordinates import Distance
from sklearn.neighbors import DistanceMetric
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('Clusteringdata.csv')
features = data[['RA', 'DEC', 'Redshift']]
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data['TrueLabels'])

X1_train, X2_test, y1_train, y2_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)
X1_ttrain, X2_ttest, y1_ttrain, y2_ttest = train_test_split(X1_train, y1_train, test_size = 0.5, random_state=42)

def spherical_distance(coord1, coord2):
    ra1, dec1, z1 = coord1
    ra2, dec2, z2 = coord2
    c =  299792 #in km/sec
    ra1, ra2, dec1, dec2 = np.radians([ra1, ra2, dec1, dec2])
    r1 = (z1*c)/67.0
    r2 = (z2*c)/67.0
    distance = sqrt( r1**2 + r2**2 - 2*r1*r2*(sin(dec1)*sin(dec2)*cos(ra1-ra2) + (cos(dec1)*cos(dec2)) ))      
    return distance

# Extract the coordinates from the training sample
#coordinates = X1_ttrain.values
coordinates = X2_ttest.values
# Create a distance metric object using the custom distance function
distance_metric = DistanceMetric.get_metric(spherical_distance)
# Compute the pairwise distances using the custom distance metric
# dbscan_dist_matrix = distance_metric.pairwise(coordinates)
agglo_dist_matrix = distance_metric.pairwise(coordinates)


class DBSCANWrapper(DBSCAN):
    def predict(self,X):
        return self.labels_.astype(int)
    
param_grid = {'eps': np.linspace(1.98 ,2.05,num=10),
             'min_samples': [4] 
             ,'metric': ['precomputed']} 

search_cv = RandomizedSearchCV(estimator = DBSCANWrapper(),
                               param_distributions=param_grid,
                               scoring=score, 
                               cv=3,
                               verbose=3,
                               n_iter=3,
                               refit=True,
                               n_jobs=1,
                               return_train_score=True,
                               error_score='raise'
                              )
search_cv.fit(dbscan_dist_matrix, y1_ttrain)
best_eps = search_cv.best_params_['eps']
best_min_samples = search_cv.best_params_['min_samples']
final_model = search_cv.best_estimator_
print(f"best epsilon: {best_eps}")
print(f"best min_samples: {best_min_samples}")
labels_pred1 = final_model.fit_predict(agglo_dist_matrix)
num_clusters = len(set(labels_pred1)) - (1 if -1 in labels_pred1 else 0)
print(f"labels: {labels_pred1} ")
print(f"Number of clusters found: {num_clusters}")
#Compute the scoring ! 
avg_scoring = scoring_metric(y2_ttest, labels_pred1)
print(f"Average Scoring: {avg_scoring}")

#Store outliers and remove them

outlier_indices = np.where(labels_pred1 == -1)[0]
outliers_X2_ttest = X2_ttest.iloc[outlier_indices]
#print("Outliers detected:\n", outliers_X2_test)
X2_ttest_cleaned = X2_ttest.drop(X2_ttest.index[outlier_indices])

non_outlier_indices = np.where(labels_pred1 != -1)[0]
y2_ttest_cleaned = y2_ttest[non_outlier_indices]

def spherical_distance(coord1, coord2):
    ra1, dec1, z1 = coord1
    ra2, dec2, z2 = coord2
    c =  299792 #in km/sec
    ra1, ra2, dec1, dec2 = np.radians([ra1, ra2, dec1, dec2])
    r1 = (z1*c)/67.0
    r2 = (z2*c)/67.0
    distance = sqrt( r1**2 + r2**2 - 2*r1*r2*(sin(dec1)*sin(dec2)*cos(ra1-ra2) + (cos(dec1)*cos(dec2)) ))      
    return distance

# Extract the coordinates from the training sample
coordinates = X2_ttest_cleaned.values
coordinates 

# Create a distance metric object using the custom distance function
distance_metric = DistanceMetric.get_metric(spherical_distance)

# Compute the pairwise distances using the custom distance metric
#dist_matrix = distance_metric.pairwise(coordinates)
cleanedagglo_dist_matrix = distance_metric.pairwise(coordinates)

# Custom wrapper for AgglomerativeClustering
class WrappedAgglomerativeClustering(BaseEstimator, ClusterMixin):
    
    def __init__(self, full_distance_matrix, n_clusters=None, distance_threshold=None, 
                 affinity='precomputed', linkage='single'):
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

param_grid = {'distance_threshold': np.linspace(4.9,5.1, num=10),
              'affinity' : ['precomputed'], 
              'linkage': ['single']}  # Assuming you are using a precomputed distance matrix

search = RandomizedSearchCV(
    WrappedAgglomerativeClustering(full_distance_matrix=cleanedagglo_dist_matrix, n_clusters=None), 
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
search.fit(np.arange(cleanedagglo_dist_matrix.shape[0]), y2_ttest_cleaned)
best_threshold = search.best_params_['distance_threshold']
final_model = WrappedAgglomerativeClustering(full_distance_matrix=all_data_dmatrix, distance_threshold=best_threshold)
print(f"Best threshold: {best_threshold}")

labels_pred = final_model.fit_predict(all_data_dmatrix)
num_clusters = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
print(f"labels: {labels_pred} ")
print(f"Number of clusters found: {num_clusters}")
test_indices = X2_test.index #Obtain the indices of test sample
labels_pred_test1 = labels_pred[test_indices] #Obtain the corresponding predicted for test
labels_pred_test1.shape

#Compute the scoring ! 
avg_scoring = scoring_metric(y2_test, labels_pred_test1)
print(f"Average Scoring: {avg_scoring}")
