import re
import pandas as pd
import os
import seaborn as sns
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')



os.chdir('C:/Users/moyke/Desktop/NYCDSA/Housing_Working')

house_train = pd.read_csv('train.csv')
house_test = pd.read_csv('test.csv')
house_train.columns.values



###############################################################################

#       GETTING ALL THE DATA FROM TEXT FILE

# FNDING FACTOR LEVELS 
factorLevel = {}
with open('data_description.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.find(':') != -1 and line.find(' ') != 0:
            col_name = re.sub(':.*', '', line).strip()
            factorLevel[col_name] = []
        else:
            if len(re.findall('[a-zA-Z0-9]', line)) > 0 :
                level = re.sub('\t.*', '', line).strip()
                if level !='':
                    factorLevel[col_name].append(level)             
                   
# outputs factorLevel as a dict without any blank spaces
factorLevel = {k:v for k,v in factorLevel.items() if len(v) > 0}


###############################################################################

# Nulling id column
del house_train['Id']
del house_test['Id']

##############################################################################

# LOOKING AT DIMENSIONS
house_train.shape
house_test.shape
##### THOUGHTS #####
# There are 79 predictors variables, that will be regressed by SalePrice



###############################################################################

# UNDERSTANDING DATATYPES
house_train.dtypes
house_test.dtypes

# Convert numerical variables that are actually categories
actually_cat = ['MSSubClass', 'MoSold', 'YrSold'] #  'MoSold', 'YrSold'
house_train[actually_cat] = house_train[actually_cat].astype('str')
house_test[actually_cat] = house_test[actually_cat].astype('str')
##### THOUGHTS #####
# Ordinal variables can be kept as numericals: OverallQual, OverallCond, BUT
# MSSubClass, MoSold, YrSold is listed as an integer, but are actually categories


data = pd.concat([house_train['SalePrice'], house_train['YearBuilt']], axis = 1)
fig, ax = plt.subplots(figsize=(30,10))
sns.boxplot(x = 'YearBuilt', y = 'SalePrice', data = data)
plt.xticks(rotation=45)
# SalePrice is higher in recent years




# Label Encode Ordinal variables
label_encode_var = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',
                    'KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']

house_train[label_encode_var] = house_train[label_encode_var].replace({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0 } )
house_test[label_encode_var] = house_test[label_encode_var].replace({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0 } )



###############################################################################
        
# UNDERSTANDING MISSINGNESS 

train_missing = house_train.isnull().sum().sort_values(ascending = False)
train_miss_per = (house_train.isnull().sum() / len(house_train)).sort_values(ascending = False)

train_miss_data = pd.concat([train_missing,train_miss_per], axis = 1, keys = ['Total','Percentage'] )
train_miss_data.head(20)

##### THOUGHTS #####
# Missing Completely At Random?
# PoolQC, MiscFeature, Alley, Fence, and FireplaceQu have over 20% missing data
# They dont tell us much so lets delete
# Lot Frontage only as 18% missing. Its borderline. Use Regression to fill missing data

# Deleting very missing data 
really_missing_var = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
house_train = house_train.drop(really_missing_var, axis = 1)
house_test = house_test.drop(really_missing_var, axis = 1)


# IMPUTE LOTFRONTAGE WITH RESGRESSION ON LOTAREA
fig, ax = plt.subplots(figsize=(15,15))
plt.scatter( np.log(house_train.LotArea), np.log(house_train.LotFrontage))
plt.xlabel('LotArea')
plt.ylabel('LotFrontage')

# Creating lot DF
lot_var = ['LotArea','LotFrontage']
lot_df = house_train[lot_var].dropna(axis = 0, how = 'any')
lot_df.shape
# Logging both to make relationship linear
lot_df.LotArea = np.log1p(lot_df.LotArea)
lot_df.LotFrontage = np.log1p(lot_df.LotFrontage)
# Create lm model 
from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept = False)
lm.fit(lot_df.LotArea.reshape(-1,1), lot_df.LotFrontage.reshape(-1,1))

beta1_lot = lm.coef_

def regress_frontage (df):
    if df.LotFrontage != df.LotFrontage:
        return np.exp(beta1_lot * np.log1p(df.LotArea))
    else:
        return df.LotFrontage

# Replacing missing values with regression
house_train['LotFrontage'] = house_train.apply(lambda x: regress_frontage(x), axis = 1)
house_test['LotFrontage'] = house_test.apply(lambda x: regress_frontage(x), axis = 1)

house_train.LotFrontage = house_train.LotFrontage.astype('int64')
house_test.LotFrontage = house_test.LotFrontage.astype('int64')



################################################################################

# IMPUTING MISSING FEATURES 

# Categorical variables that have NAs listed as possible value. Impute with 'None'.
cat_none = ["GarageFinish", 'GarageType', "BsmtExposure", "BsmtFinType1",
            "BsmtFinType2",'MasVnrType']
house_train[cat_none] = house_train[cat_none].replace(np.nan, "None")
house_test[cat_none] = house_test[cat_none].replace(np.nan, "None")



# Categories that a house MUST have. Impute with mode
cat_mode = ["LotShape","LandContour", "LandSlope", "BldgType","HeatingQC", 
            "CentralAir", "Functional",'MSZoning', 'Electrical', 'Utilities',
            "PavedDrive"]
house_train[cat_mode] = house_train[cat_mode].replace(np.nan, house_train[cat_mode].mode())
house_test[cat_mode] = house_test[cat_mode].replace(np.nan, house_train[cat_mode].mode())



# categories that have "Other" as option. Impute with 'Other'
cat_other = ['SaleType', 'Exterior1st', 'Exterior2nd'] 
house_train[cat_other] = house_train[cat_other].replace(np.nan, "Other")
house_test[cat_other] = house_test[cat_other].replace(np.nan, "Other")



# Impute the median for Numerical features in training
for feature in house_train.columns.values:
    if house_train.loc[:,feature].dtype.name == 'float64' or house_train.loc[:,feature].dtype.name == 'int64':
        if house_train.loc[:,feature].isnull().any() == True:
            house_train.loc[house_train[feature].isnull(),feature] = house_train.loc[:,feature].median()
            
            
   
# Impute the median for numerical features in testing using training median
for feature in house_test.columns.values:
    if house_test.loc[:,feature].dtype.name == 'float64' or house_test.loc[:,feature].dtype.name == 'int64':
        if house_test.loc[:,feature].isnull().any() == True:
            house_test.loc[house_test[feature].isnull(),feature] = house_train.loc[:,feature].median()
 
  
# Check for  null values in training data
house_train.isnull().any().sum() # 0 columns with null values
house_train.isnull().any().sort_values(ascending = False)

house_test.isnull().any().sum() # 0 columns with null values
house_test.isnull().any().sort_values(ascending = False)



#############################################################################

# FEATURE ENGINEERING ON FULL DATA SET

# Combining both house_train and house_test to clean data
house_train # 75 columns
house_test # 74 columns. need SalePrice placeholder
house_test['SalePrice'] = np.nan
# house_test starts on index row 1460 
house_full = pd.concat([house_train, house_test], axis = 0)
#house_full.iloc[1460,]



# IMPUTING GARAGEYRBLT (deleting and replacing with engineered features)

# Create new feature: hasGarage
house_full['hasGarage'] = [0 if x == True else 1 for x in house_full.GarageYrBlt.isnull()]
# replacing garageyrblt value 2207 to 2007
house_full.loc[house_full.GarageYrBlt == 2207, 'GarageYrBlt']  = 2007
# Create new feature GarageBlt (how many years after was the garage built after the house)
house_full['GarageBlt'] = house_full.GarageYrBlt - house_full.YearBuilt
# NA values will equal to zero assuming that these garages were built the same time as the house  according to Shu's graph
house_full.loc[house_full.GarageBlt.isnull(), 'GarageBlt'] = 0
# If garage year built is before house year built, make it the same year. Human Error?
house_full.loc[house_full['GarageBlt'] < 0, 'GarageBlt'] = 0
# Delete the GarageYrBlt variable. No longer needed 
del house_full['GarageYrBlt']
# Plot year built against garageblt
sns.lmplot(x = 'YearBuilt', y = 'GarageBlt', fit_reg = False, hue = 'hasGarage', data = house_full)



# TRANSFORMING YEARREMODADD 

# New var isRemod: 0 if year remodelled is the same as year built, 1 otherwise 
house_full['isRemod'] = [0 if x == True else 1 for x in house_full.YearRemodAdd == house_full.YearBuilt]
# New var RemodAdd: YearRemodAdd - YearBuilt: How many years later was hosue remodeled?
house_full['RemodAdd'] = house_full.YearRemodAdd - house_full.YearBuilt
# Make remodeled year the same as year built if year remodeled was before the year built. human error?
house_full.loc[house_full['RemodAdd'] < 0, 'RemodAdd'] = 0
# Delete YearRemodAdd variable
del house_full['YearRemodAdd'] 
# Plot YearBuilt against RemodAdd
sns.lmplot(x = 'YearBuilt', y = 'RemodAdd', data = house_full, fit_reg = False, hue = 'isRemod')



# TOTAL NUMBER OF BATHROOMS
house_full['TotBath'] = house_full.FullBath + .5*house_full.HalfBath
del house_full['FullBath']
del house_full['HalfBath']


# Replace zero bedrooms with 1 bedroom. maybe its a studio? This is so we can engineer a feature
house_full.BedroomAbvGr.replace(0, 1, inplace=True)
house_full.BedroomAbvGr.unique()
# Bath Capacity. The higher the ratio the better
house_full['Bath_Capacity'] = house_full.TotBath / house_full.BedroomAbvGr
house_full.Bath_Capacity
# Parking Capacity. The higher the better
house_full['Parking_Capacity'] = house_full.GarageCars / house_full.BedroomAbvGr
house_full.Parking_Capacity.unique()



# TOTAL NUMBER OF BASEMENT BATHROOMS
house_full['BsmtTotBath'] = house_full.BsmtFullBath + .5*house_full.BsmtHalfBath
del house_full['BsmtFullBath']
del house_full['BsmtHalfBath']



# GRLIVAREA IS JUST THE SUME OF 3 VARIABLES. DEL 3 VARIABLES
del house_full['1stFlrSF']
del house_full['2ndFlrSF']
del house_full['LowQualFinSF']



# TOTAL PORCH SQUARE FOOTAGE
house_full['TotalPorchSF'] = house_full['OpenPorchSF'] + house_full['EnclosedPorch'] + house_full['3SsnPorch'] + house_full['ScreenPorch']
del house_full['OpenPorchSF']
del house_full['EnclosedPorch']
del house_full['3SsnPorch']
del house_full['ScreenPorch']



# TOTAL BASEMENT SQUARE FOOTAGE IS GIVEN. DELETE VARIABLES THAT ADD UP TO IT
del house_full['BsmtFinSF1']
del house_full['BsmtFinSF2']
del house_full['BsmtUnfSF']


# split full data back to training and test before imputation
house_train = house_full.iloc[0:1460,:]
house_test = house_full.iloc[1460:,]


####################################################################################

# UNDERSTANDING LOW VARIANCE

#from sklearn.feature_selection import VarianceThreshold

# there are 42 numerical columns
#numerical_var = house_train.select_dtypes(exclude = 'category').columns.values
#numerical_var

#def var_selector (df, threshold):
#    columns = df.columns.values
#    selector = VarianceThreshold(threshold)
#    #scaler = StandardScaler()
#    #df = scaler.fit_transform(df)
#    selector.fit_transform(df)
#    columns_index = [columns[x] for x in selector.get_support(indices = True)]
#    diff = set(columns) - set(columns_index)
#    return diff

#var_selector(house_train[numerical_var], .16)

#house_train.BsmtHalfBath.describe()[2]**2
#house_train.KitchenAbvGr.describe()[2]**2
#house_train.hasGarage.describe()[2]**2
#house_train.GarageQual.describe()[2]**2
#house_train.ExterCond.describe()[2]**2

##### THOUGHTS #####
# This is only selecting out variables with small possible values. So of course
# dvariance will be smaller for them...
# Will keep as is and not use this


##############################################################################

# UNDERSTANDING LOW VARIANCE PART 2

# Deleting zero or near zero variance as done by Shu
near_zero_var = ["Street", "Utilities", "Condition2", "RoofMatl","PoolArea",
                 "MiscVal" ]

house_train = house_train.drop(near_zero_var, axis = 1)
house_test = house_test.drop(near_zero_var, axis = 1)

house_train.shape
house_test.shape

############################################################################

# Outliers

# Visualize outliers
fig, ax = plt.subplots(figsize=(30,10))
sns.pairplot(data = house_train,
             x_vars = house_train.select_dtypes(exclude = 'object').columns.values,
             y_vars =['SalePrice'],dropna = True)


outliers = [
        house_train[house_train['LotFrontage'] > 250].index.values,
        house_train[house_train['LotArea'] > 100000].index.values,
        house_train[house_train['TotalBsmtSF'] > 4000].index.values,
        house_train[house_train['GrLivArea'] > 4500].index.values,
        house_train[house_train['TotalPorchSF'] > 700].index.values
        ]

outliers_indices = [inner_num for sub_list in outliers for inner_num in sub_list]


house_train = house_train.drop(house_train.index[outliers_indices])
house_train.shape

#################################################################################

# Normalize SalePrice in Training data
fig, ax = plt.subplots(figsize=(10,10))
sns.distplot(house_train.SalePrice, bins = 20)
fig, ax = plt.subplots(figsize=(10,10))
sns.distplot([np.log(house_train.SalePrice + 1)], bins = 20)

# Converting SalePrice to Log
house_train.SalePrice = np.log1p(house_train.SalePrice)
house_test.SalePrice = np.log1p(house_test.SalePrice)



##############################################################################

# NORMALIZE SKEWED INDEPENDENT VARIABLES TO ENSURE LINEAR RELATIONSHIP AND REMOVE OUTLIERS

from scipy.stats import skew 

numerical_var = house_train.select_dtypes(exclude = 'object').columns.values
numerical_var = [x for x in numerical_var if x != 'SalePrice']

skewed_var = [x for x in numerical_var if abs(skew(house_train[x])) > .6]
skewed_var

# Only some of these variables SHOULD and CAN be log transformed
# Do not transform ordinal variables!
to_skew = ['LotArea', 'LotFrontage', 'MasVnrArea','GrLivArea', 
           'WoodDeckSF', 'GarageBlt', 'RemodAdd', 'Bath_Capacity', 'Parking_Capacity',
           'TotalPorchSF']

# Log transform features
for feature in to_skew:
        house_train[feature] = np.log1p(house_train[feature])
        house_test[feature] = np.log1p(house_test[feature])


# Confirm more normal skew
[skew(house_train[x]) for x in to_skew]
[skew(house_test[x]) for x in to_skew]

house_train[to_skew].dtypes

##############################################################################

# Combine house and test to dummify together
from copy import deepcopy


# house_test starts on index row 1460 
house_full = pd.concat([house_train, house_test], axis = 0)

house_full_copy = deepcopy(house_full)

house_full_dummy = pd.get_dummies(house_full_copy)

# split train and test again
house_train = house_full_dummy.iloc[0:1451,:]
house_test = house_full_dummy.iloc[1451:,]



############################################################################

# Split training independent and dependent variable

house_train_x = house_train.drop('SalePrice', axis = 1)
house_train_y = house_train.SalePrice.values.reshape(-1,1)
house_test_x = house_test.drop('SalePrice', axis = 1)


house_train_x.to_csv('house_train_x_v2.csv')
pd.DataFrame(house_train_y).to_csv('house_train_y_v2.csv')
house_test_x.to_csv('house_test_x_v2.csv')

