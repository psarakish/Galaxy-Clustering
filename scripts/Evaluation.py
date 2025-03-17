import numpy as np
from sklearn.metrics import make_scorer

def scoring_metric(y, yhat, return_match=False):
    unique_y = np.unique(y)  # Unique real clusters
    n_clusters = len(unique_y)  # Number of real clusters
    performances = np.zeros(n_clusters)
    mapping_dict = {}  

    for j, unique_y_j in enumerate(unique_y):
        idxs_real_galaxies_in_j = np.where(y == unique_y_j)[0]  # Indices of real clusters
        N_j = len(idxs_real_galaxies_in_j)
        max_performance = 0.0
        corresponding_i = None
       
        for unique_yhat_i in np.unique(yhat):
            yhat_galaxies_in_j = yhat[idxs_real_galaxies_in_j] # Predicted labels for all instances where the true label is unique_y_j
            yhat_galaxies_not_in_j = yhat[~idxs_real_galaxies_in_j]
            N_j_i = len(np.where(yhat_galaxies_in_j == unique_yhat_i)[0])
            FP = len(np.where(yhat_galaxies_not_in_j == unique_yhat_i)[0])
            FN = len(np.where(yhat_galaxies_in_j != unique_yhat_i)[0])
            performance_i = (2*N_j_i) / (2*N_j_i + FP + FN) if (2*N_j_i + FP + FN)>0 else 0                  
            if performance_i > max_performance:
                max_performance = performance_i
                corresponding_i = unique_yhat_i

        performances[j] = max_performance
        mapping_dict[unique_y_j] = corresponding_i

    avg_performance = np.mean(performances)

    if not return_match:
        return avg_performance
    else:
        return avg_performance, mapping_dict

score = make_scorer(scoring_metric)
