import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
import sklearn.model_selection as ms
from sklearn.grid_search import GridSearchCV
import math

os.chdir('C:/Users/moyke/Desktop/NYCDSA/Housing_Working')


##############################################################################

# Loading in Training and Validation sets
house_train_x = pd.read_csv('house_train_x_v2.csv').drop('Unnamed: 0', axis = 1)
house_train_y = pd.read_csv('house_train_y_v2.csv').drop('Unnamed: 0', axis = 1).values

# Loading in Actual Test Set
house_test_x = pd.read_csv('house_test_x_v2.csv').drop('Unnamed: 0', axis = 1)



###############################################################################

# Create sperate out original training set to the new training and test set
x_train, x_test, y_train, y_test = ms.train_test_split(house_train_x, \
                                                       house_train_y, \
                                                       test_size = 1/8, \
                                                       random_state = 0)

############################################################################

from sklearn.ensemble import RandomForestRegressor


# GRID SEARCH CV RANDOM FOREST

rf_tree_cv = RandomForestRegressor(
                                bootstrap = True,
                                oob_score = True,
                                random_state = 0
                                )

rf_param_grid = [{
        'n_estimators' : [1000], # [900,1000,1100]
        'max_depth' : [15], # [14,15,16,17]
        #'min_samples_split' : np.arange(2,6)
        #'min_samples_leaf' : np.arange(1,5), # np.arange(5,10) 6 is best
        'max_features' : [36] # np.arange(35,38)
}]

rf_grid_search = GridSearchCV(rf_tree_cv,
                              param_grid = rf_param_grid,
                              cv = 5,
                              verbose = 2,
                              scoring= 'neg_mean_squared_error',
                              n_jobs = 1
                              )

rf_grid_search.fit(x_train, y_train)

#  1000 trees, 15 depth and 36 max features
rf_grid_search.best_params_

# RMSE .13706
math.sqrt(-rf_grid_search.best_score_)


# Refit randorm forest 
rf_tree = RandomForestRegressor(n_estimators = 1000,
                                max_features = 36,
                                max_depth = 15,
                                #min_samples_split = 6,
                                bootstrap = True,
                                oob_score = True,
                                random_state = 0)


rf_tree.fit(x_train, y_train)



# Training: 98.32%
print('Training Score:', rf_tree.score(x_train, y_train))
# OOB: 88.47%
print('OOB Score:', rf_tree.oob_score_)
# RMSE: 0.0514
print('Training RMSE:' , math.sqrt(mean_squared_error(y_train, rf_tree.predict(x_train))))


# Test: 89.25%
print('Test Score:', rf_tree.score(x_test, y_test))
# RMSE: 0.13589
print('Test RMSE:' , math.sqrt(mean_squared_error(y_test, rf_tree.predict(x_test))))



#################################################################
# stochastic Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

# Tuning Gradient Boosting
gb_tree_cv = GradientBoostingRegressor(subsample = 2/3,
                                       random_state = 0,
                                       warm_start = True,
                                       verbose = 1)


# Param search 
gb_param_grid = [{
        'learning_rate' : [.1],
        'n_estimators' : [375], # [350,375,400,425]
        'max_depth' : [2], #np.linspace(1,5, 5)
        'min_samples_split' : [16], # np.arange(10,21,2)
        'min_samples_leaf' : [3], # np.arange(1,6,1)
        'max_features' : [21] # np.arange(16,25,1)
}]

gb_grid_search = GridSearchCV(gb_tree_cv,
                              param_grid = gb_param_grid,
                              cv = 5,
                              scoring= 'neg_mean_squared_error')

gb_grid_search.fit(x_train, y_train)

#
gb_grid_search.best_params_

# RMSE: .11693
math.sqrt(-gb_grid_search.best_score_)




# Refit Gradient Boosting
gb_tree = GradientBoostingRegressor(max_depth = 2, 
                                    learning_rate = 0.1, 
                                    n_estimators = 375,
                                    min_samples_split = 16,
                                    max_features = 21,
                                    min_samples_leaf = 3,
                                    subsample= 2/3, 
                                    random_state=0)

gb_tree.fit(x_train, y_train)

# Training R sq: 95.41%
print('Training Score:', gb_tree.score(x_train, y_train))
# Training RMSE: .08503
print('Training RMSE:', math.sqrt(mean_squared_error(y_train, gb_tree.predict(x_train))))

# Test R sq: 90.84%
print('Test Score:', gb_tree.score(x_test, y_test))
# Test RMSE: .12547
print('Test RMSE:', math.sqrt(mean_squared_error(y_test, gb_tree.predict(x_test))))


##############################################################################

import os 

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.3.0-posix-seh-rt_v5-rev0\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import xgboost as xgb
from xgboost.sklearn import XGBRegressor


xgb_tree_cv = XGBRegressor(subsample = 2/3,
                           random_state = 0,
                           warm_start = True,
                           verbose = 2)



# Param search 
xgb_param_grid = [{
        'learning_rate' : [0.10777777777777778], #np.linspace(.09,.17,10)
        'n_estimators' : [300], # [250,275,300,325,350]
        'min_child_weight' : [5], # np.linspace(1,10,10) # number of obs per node
        'max_depth' : [2], #np.arange(1,10)
        'gamma' : [0], # np.linspace(0,1,10) # minimum reduction in RSS required for a split
        'colsample_bytree' : [1], #np.arange(.1,1.1,.1) # How much percentage of columns used to split
        'reg_lambda': [1], #np.linspace(0,1.1,12) L2 regularization term (ridge)
        'reg_alpha' : [0.30000000000000004] #np.linspace(0,1.1,12) # L1 regularization term (lasso)
}]


xgb_grid_search = GridSearchCV(xgb_tree_cv,
                               param_grid = xgb_param_grid,
                               cv = 5,
                               scoring= 'neg_mean_squared_error',
                               verbose = 2)

# Fit Parameters to model
xgb_grid_search.fit(x_train, y_train)

# Best Parameters
xgb_grid_search.best_params_

# RMSE: .11492
math.sqrt(-xgb_grid_search.best_score_)



# Refit Xtreme Gradient Boosting
xgb_tree = XGBRegressor(max_depth = 2, 
                        learning_rate = 0.10777777777777778, 
                        n_estimators = 300,
                        min_child_weight = 5,
                        colsample_bytree = 1,
                        gamma = 0,
                        reg_lambda = 1,
                        reg_alpha = 0.30000000000000004,
                        subsample= 2/3, 
                        random_state=0)

xgb_tree.fit(x_train, y_train)

# Training R sq: 95.65%
print('Training Score:', xgb_tree.score(x_train, y_train))
# Training RMSE: .08271
print('Training RMSE:', math.sqrt(mean_squared_error(y_train, xgb_tree.predict(x_train))))

# Test R sq: 91.03%
print('Test Score:', xgb_tree.score(x_test, y_test))
# Test RMSE: .12411
print('Test RMSE:', math.sqrt(mean_squared_error(y_test, xgb_tree.predict(x_test))))



##############################################################################

# Random Forest Test R sq: 89.25%, RMSE: 0.13589
rf_test_pred = rf_tree.predict(house_test_x)
rf_test_pred = np.exp(rf_test_pred)
rf_test_pred

pd.DataFrame(rf_test_pred).to_csv('rf_test_pred_v2.csv')



# Gradient Boost Test R sq: 90.84%, RMSE: .12547
gb_test_pred = gb_tree.predict(house_test_x)
gb_test_pred = np.exp(gb_test_pred)
gb_test_pred

pd.DataFrame(gb_test_pred).to_csv('gb_test_pred_v2.csv')



# Xtreme Gradient Boost Test R sq: 91.03%, RMSE: .12411
xgb_test_pred = xgb_tree.predict(house_test_x)
xgb_test_pred = np.exp(xgb_test_pred)
xgb_test_pred

pd.DataFrame(xgb_test_pred).to_csv('xgb_test_pred_v2.csv')