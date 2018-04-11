import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error

os.chdir('C:/Users/moyke/Desktop/NYCDSA/Housing_Working')

##############################################################################

# Loading in Training and Validation sets
x_train = pd.read_csv('house_train_x_v2.csv').drop('Unnamed: 0', axis = 1)
y_train = pd.read_csv('house_train_y_v2.csv').drop('Unnamed: 0', axis = 1).values
y_train = y_train.reshape(-1,)

# Loading in Actual Test Set
x_test = pd.read_csv('house_test_x_v2.csv').drop('Unnamed: 0', axis = 1)

###############################################################################

from sklearn.model_selection import KFold
from sklearn.base import clone

def transformer(y, func=None):
    """Transforms target variable and prediction"""
    if func is None:
        return y
    else:
        return func(y)


def stacking_regression(models, meta_model, X_train, y_train, X_test,
             transform_target=None, transform_pred=None,
             metric=None, n_folds=3, average_fold=True,
             shuffle=False, random_state=0, verbose=1):
    '''
    Function 'stacking' takes train data, test data, list of 1-st level
    models, meta_model for the 2-nd level and returns stacking predictions.

    Parameters
    ----------
    models : list
        List of 1-st level models. You can use any models that follow sklearn
        convention i.e. accept numpy arrays and have methods 'fit' and 'predict'.

    meta_model: model
        2-nd level model. You can use any model that follow sklearn convention

    X_train : numpy array or sparse matrix of shape [n_train_samples, n_features]
        Training data

    y_train : numpy 1d array
        Target values

    X_test : numpy array or sparse matrix of shape [n_test_samples, n_features]
        Test data


    transform_target : callable, default None
        Function to transform target variable.
        If None - transformation is not used.
        For example, for regression task (if target variable is skewed)
            you can use transformation like numpy.log.
            Set transform_target = numpy.log
        Usually you want to use respective backward transformation
            for prediction like numpy.exp.
            Set transform_pred = numpy.exp
        Caution! Some transformations may give inapplicable results.
            For example, if target variable contains zeros, numpy.log
            gives you -inf. In such case you can use appropriate
            transformation like numpy.log1p and respective
            backward transformation like numpy.expm1

    transform_pred : callable, default None
        Function to transform prediction.
        If None - transformation is not used.
        If you use transformation for target variable (transform_target)
            like numpy.log, then using transform_pred you can specify
            respective backward transformation like numpy.exp.
        Look at description of parameter transform_target

    metric : callable, default None
        Evaluation metric (score function) which is used to calculate
        results of cross-validation.
        If None, then by default:
            sklearn.metrics.mean_absolute_error - for regression

    n_folds : int, default 3
        Number of folds in cross-validation

    average_fold: boolean, default True
        Whether to take the average of the predictions on test set from each fold.
        Refit the model using the whole training set and predict test set if False

    shuffle : boolean, default False
        Whether to perform a shuffle before cross-validation split

    random_state : int, default 0
        Random seed for shuffle

    verbose : int, default 1
        Level of verbosity.
        0 - show no messages
        1 - for each model show single mean score
        2 - for each model show score for each fold and mean score

        Caution. To calculate MEAN score across all folds
        full train set prediction and full true target are used.
        So for some metrics (e.g. rmse) this value may not be equal
        to mean of score values calculated for each fold.

    Returns
    -------
    stacking_prediction : numpy array of shape n_test_samples
        Stacking prediction
    '''

    # Specify default metric for cross-validation
    if metric is None:
        metric = mean_squared_error

    # Print metric
    if verbose > 0:
        print('metric: [%s]\n' % metric.__name__)

    # Split indices to get folds
    kf = KFold(n_splits = n_folds, shuffle = shuffle, random_state = random_state)

    if X_train.__class__.__name__ == "DataFrame":
    	X_train = X_train.as_matrix()
    	X_test = X_test.as_matrix()

    # Create empty numpy arrays for stacking features
    S_train = np.zeros((X_train.shape[0], len(models)))
    S_test = np.zeros((X_test.shape[0], len(models)))

    # Loop across models
    for model_counter, model in enumerate(models):
        if verbose > 0:
            print('model %d: [%s]' % (model_counter, model.__class__.__name__))

        # Create empty numpy array, which will contain temporary predictions for test set made in each fold
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        # Loop across folds
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            # Clone the model because fit will mutate the model.
            instance = clone(model)
            # Fit 1-st level model
            instance.fit(X_tr, transformer(y_tr, func = transform_target))
            # Predict out-of-fold part of train set
            S_train[te_index, model_counter] = transformer(instance.predict(X_te), func = transform_pred)
            # Predict full test set
            S_test_temp[:, fold_counter] = transformer(instance.predict(X_test), func = transform_pred)

            # Delete temperatory model
            del instance

            if verbose > 1:
                print('    fold %d: [%.8f]' % (fold_counter, metric(y_te, S_train[te_index, model_counter])))

        # Compute mean or mode of predictions for test set
        if average_fold:
            S_test[:, model_counter] = np.mean(S_test_temp, axis = 1)
        else:
            model.fit(X_train, transformer(y_train, func = transform_target))
            S_test[:, model_counter] = transformer(model.predict(X_test), func = transform_pred)

        if verbose > 0:
            print('    ----')
            print('    MEAN RMSE:   [%.8f]\n' % np.sqrt((metric(y_train, S_train[:, model_counter]))))

    # Fit our second layer meta model
    meta_model.fit(S_train, transformer(y_train, func = transform_target))
    # Make our final prediction
    stacking_prediction = transformer(meta_model.predict(S_test), func = transform_pred)

    return stacking_prediction


##############################################################################
    


# scale data for regularized regression 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)

# transform training set
X_train = scaler.transform(x_train)
X_train = pd.DataFrame(X_train, columns = x_train.columns)
X_train.head()


# transform actual test set (house_test)
X_test = scaler.transform(x_test)
X_test = pd.DataFrame(X_test, columns = x_test.columns)
X_test.head()


###############################################################################

from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha = 137, # 
                   random_state = 0)

ridge_reg.fit(X_train,y_train)



##############################################################################

from sklearn.linear_model import Lasso

# Refit best params
lasso_reg = Lasso(alpha = 0.00299, 
                   random_state = 0)

lasso_reg.fit(X_train,y_train)



#############################################################################

from sklearn.linear_model import ElasticNet

# Refit best params
elastic_net = ElasticNet(alpha = 0.02,
                          l1_ratio = 0.15126530612244898,
                   random_state = 0)

elastic_net.fit(X_train,y_train)



##########################################################################3###

from sklearn.ensemble import RandomForestRegressor

# Refit randorm forest 
rf_tree = RandomForestRegressor(n_estimators = 1000,
                                max_features = 36,
                                max_depth = 15,
                                #min_samples_split = 6,
                                bootstrap = True,
                                oob_score = True,
                                random_state = 0)


rf_tree.fit(x_train, y_train)



##############################################################################

from sklearn.ensemble import GradientBoostingRegressor

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



##############################################################################

import os 
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.3.0-posix-seh-rt_v5-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb
from xgboost.sklearn import XGBRegressor

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


###############################################################################

#  'stacked_5mod.csv' (.12248)
models = [ridge_reg, elastic_net, rf_tree, gb_tree, xgb_tree]

meta_model = lasso_reg


y_predicted = stacking_regression(models, meta_model, X_train, y_train, X_test,
             metric=None, n_folds=5, average_fold=True,
             shuffle=False, random_state=0, verbose=2)


y_predicted = np.exp(y_predicted)
y_predicted

pd.DataFrame(y_predicted).to_csv('stacked_4mod_a.csv')


