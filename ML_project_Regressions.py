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
house_train_x = pd.read_csv('house_train_x_v3.csv').drop('Unnamed: 0', axis = 1)
house_train_y = pd.read_csv('house_train_y_v3.csv').drop('Unnamed: 0', axis = 1).values

# Loading in Actual Test Set
house_test_x = pd.read_csv('house_test_x_v3.csv').drop('Unnamed: 0', axis = 1)



###############################################################################

# Create sperate out original training set to the new training and test set
x_train, x_test, y_train, y_test = ms.train_test_split(house_train_x, \
                                                       house_train_y, \
                                                       test_size = 1/8, \
                                                       random_state = 0)

##############################################################################

# scale data for regularized regression 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)

# transform training set
X_train = scaler.transform(x_train)
X_train = pd.DataFrame(X_train, columns = x_train.columns)
X_train.head()

# transform train test set (validation)
X_test = scaler.transform(x_test)
X_test = pd.DataFrame(X_test, columns = x_test.columns)
X_test.head()


# transform actual test set (house_test)
X_Test = scaler.transform(house_test_x)
X_Test = pd.DataFrame(X_Test, columns = house_test_x.columns)
X_Test.head()



#####################################################################3######
from sklearn.linear_model import Ridge

# HYPERTUNED RIDGE
ridge_cv = Ridge(random_state=0)

ridge_param_grid = [{
        'alpha' : np.linspace(1,100,100) #(540,550, 11) # (400,600, 201) np.linspace(100,300,201)
        }]

#ridge_param_grid = [{
#        'alpha' : np.logspace(-1,3,100)
#        }]

# Grid Search alpha (lambda)
ridge_grid_search = GridSearchCV(ridge_cv,
                                 param_grid = ridge_param_grid,
                                 cv = 5,
                                 scoring = 'neg_mean_squared_error')

ridge_grid_search.fit(X_train, y_train)

# alpha = 11
ridge_grid_search.best_params_
# RMSE = .1205
math.sqrt(-ridge_grid_search.best_score_)

# Refit best params
ridge_reg2 = Ridge(alpha = 11, # 
                   random_state = 0)

ridge_reg2.fit(X_train,y_train)



# Training error: .0938
math.sqrt(mean_squared_error(y_train, ridge_reg2.predict(X_train)))
# Training R sq: 94.41% 
ridge_reg2.score(X_train, y_train)



# Test error: .12085
math.sqrt(mean_squared_error(y_test, ridge_reg2.predict(X_test)))
# Test R sq 91.50% 
ridge_reg2.score(X_test, y_test)



##################################################################################

from sklearn.linear_model import Lasso

lasso_cv = Lasso(random_state = 0)

lasso_param_grid = [{
        'alpha' : np.linspace(0.002 ,.004, 201) #np.linspace(0.001 ,.01, 10)
        }]

 
#lasso_param_grid = [{
#        'alpha' : np.logspace(-3,3,100)
#        }]

# Grid Search alpha (lambda)
lasso_grid_search = GridSearchCV(lasso_cv,
                                 param_grid = lasso_param_grid,
                                 cv = 5,
                                 scoring = 'neg_mean_squared_error')

lasso_grid_search.fit(X_train, y_train)

# alpha = 0.00318
lasso_grid_search.best_params_
# RMSE = .11382
math.sqrt(-lasso_grid_search.best_score_)

# Refit best params
lasso_reg2 = Lasso(alpha = 0.00318, 
                   random_state = 0)

lasso_reg2.fit(X_train,y_train)

# Training error (.10308)
math.sqrt(mean_squared_error(y_train, lasso_reg2.predict(X_train)))
# Training R sq (93.25%)
lasso_reg2.score(X_train, y_train)

# Test error (.11720) 
math.sqrt(mean_squared_error(y_test, lasso_reg2.predict(X_test)))
# Test R sq (92.00%)
lasso_reg2.score(X_test, y_test)


# THOUGHTS
# Overall, Lasso regression performs better than Ridge by a few percent
# Final Ridge = 91.50%, RMSE: .12085
# Final Lasso = 92.50%, RMSE: .11720 

#################################################################################

from sklearn.linear_model import ElasticNet

# HYPERTUNED ELASTIC NET
elastic_cv = ElasticNet()

elastic_param_grid = [{
        'alpha' : np.linspace(0.01 , .5, 50),
        'l1_ratio' : np.linspace(0.001 ,0.2, 50),
        'random_state': [0]
        }]

# Grid Search alpha (lambda)
elastic_grid_search = GridSearchCV(elastic_cv,
                                   param_grid = elastic_param_grid,
                                   cv = 5,
                                   verbose = 2,
                                   scoring = 'neg_mean_squared_error')

elastic_grid_search.fit(X_train, y_train)

# alpha = 0.02 & ratio = 0.16344897959183674
elastic_grid_search.best_params_
# CV Error: .11409
math.sqrt(-elastic_grid_search.best_score_)

# Refit best params
elastic_net2 = ElasticNet(alpha = 0.02,
                          l1_ratio = 0.16344897959183674,
                   random_state = 0)

elastic_net2.fit(X_train,y_train)

# Training error (.10347)
math.sqrt(mean_squared_error(y_train, elastic_net2.predict(X_train)))
# Training R sq (93.20%)
elastic_net2.score(X_train, y_train)

# Test error (.11694)
math.sqrt(mean_squared_error(y_test, elastic_net2.predict(X_test)))
# Test R sq (92.04%)
elastic_net2.score(X_test, y_test)



###################################################################

# Ridge Prediction (Val R sq: 91.50%, RMSE = .12085)

ridge_test_pred = ridge_reg2.predict(X_Test)
ridge_test_pred = np.exp(ridge_test_pred)
ridge_test_pred

pd.DataFrame(ridge_test_pred).to_csv('ridge_test_pred_v3.csv')




# Lasso Prediction (Val R sq: 92.50, RMSE = .11720)

lasso_test_pred = lasso_reg2.predict(X_Test)
lasso_test_pred = np.exp(lasso_test_pred)
lasso_test_pred

pd.DataFrame(lasso_test_pred).to_csv('lasso_test_pred_v3.csv')




# Elastic Prediction (Val R sq: 92.04%, RMSE = .11694)

elastic_test_pred = elastic_net2.predict(X_Test)
elastic_test_pred = np.exp(elastic_test_pred)
elastic_test_pred

pd.DataFrame(elastic_test_pred).to_csv('elastic_test_pred_v3.csv')



###### THOUGHTS #######
# Elastic > Lasso > Ridge
# Model perfers Lasso


