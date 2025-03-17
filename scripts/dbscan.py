from sklearn.cluster import DBSCAN
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV #use gridsearch instead
import warnings
warnings.filterwarnings("ignore")

param_grid = {'eps': np.linspace(1.9, 2.1, num=10), #Set wider eps range if needed
             'min_samples': [4]
             ,'metric': ['precomputed']} 

class DBSCANWrapper(DBSCAN):
    def predict(self,X):
        return self.labels_.astype(int)
    
search_cv = RandomizedSearchCV(estimator = DBSCANWrapper(),
                               param_distributions=param_grid,
                               scoring=score, 
                               cv=3, #Set wider
                               verbose=3,
                               n_iter=1, #Set wider
                               refit=True, #False = no best_estimator_
                               n_jobs=1,
                               return_train_score=True,
                               error_score='raise'
                              )

search_cv.fit(train_distance_matrix, y_train)

best_eps = search_cv.best_params_['eps']
best_min_samples = search_cv.best_params_['min_samples']
final_model = search_cv.best_estimator_
print(f"best epsilon: {best_eps}")
print(f"best min_samples: {best_min_samples}")

labels_pred = final_model.fit_predict(all_data_dmatrix) ##ReFit on whole distance matrix -> Overfitting!!

num_clusters = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
print(f"labels: {labels_pred} ")
print(f"Number of clusters found: {num_clusters}")

data['PredLabels'] = labels_pred
test_indices = X_test.index #Obtain the indices of test sample
labels_pred_test1 = labels_pred[test_indices] #Obtain the corresponding predicted for test
labels_pred_test1.shape

#Compute the performance on the TEST SET!! 
avg_scoring = scoring_metric(labels_pred_test1, y_test)
print(f"Average Scoring: {avg_scoring}")
