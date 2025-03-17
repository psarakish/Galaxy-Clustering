import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import pairwise_distances 
from astropy.coordinates import Distance
from sklearn.neighbors import DistanceMetric
from numpy import sqrt 
from math import cos
from math import sin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV #use gridsearch instead
from sklearn.metrics import make_scorer
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('Clusteringdata.csv')
features = data[['RA', 'DEC', 'Redshift']]
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data['TrueLabels'])

X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
labels_vc = data['TrueLabels'].value_counts()
count_once = (labels_vc > 3 ).sum()

print(labels_vc.shape[0])
print(labels_vc)
print(count_once)

## Distance metric for all data
def spherical_distance(coord1, coord2):
    ra1, dec1, z1 = coord1
    ra2, dec2, z2 = coord2
    c =  299792 #in km/sec
    ra1, ra2, dec1, dec2 = np.radians([ra1, ra2, dec1, dec2])
    r1 = (z1*c)/67.0
    r2 = (z2*c)/67.0
    distance = sqrt(r1**2 + r2**2 - 2*r1*r2*(sin(dec1)*sin(dec2)*cos(ra1-ra2) + (cos(dec1)*cos(dec2))))      
    return distance

# Extract the coordinates
coordinates = data[['RA', 'DEC', 'Redshift']].values
# Create a distance metric object
distance_metric = DistanceMetric.get_metric(spherical_distance)
# Compute the pairwise distances
dist_matrix = distance_metric.pairwise(coordinates)

## Distance metric for train sample
  def spherical_distance(coord1, coord2):
      ra1, dec1, z1 = coord1
      ra2, dec2, z2 = coord2
      c =  299792 #in km/sec
      ra1, ra2, dec1, dec2 = np.radians([ra1, ra2, dec1, dec2])
      r1 = (z1*c)/67.0
      r2 = (z2*c)/67.0
      distance = sqrt( r1**2 + r2**2 - 2*r1*r2*(sin(dec1)*sin(dec2)*cos(ra1-ra2) + (cos(dec1)*cos(dec2)) ))      
      return distance
  # Extract the coordinates
  coordinates = X_train.values
  coordinates 
  # Create a distance metric object
  distance_metric = DistanceMetric.get_metric(spherical_distance)
  # Compute the pairwise distances
  train_distance_matrix = distance_metric.pairwise(coordinates)
