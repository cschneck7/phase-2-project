import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
from sklearn.feature_selection import RFECV
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
    
def initial_prep(df):
#     Maps new values for condition column
    condition_map = {'Poor': 1, 'Fair': 2, 'Average': 3, 'Good': 4, 'Very Good': 5}
    df.condition = df.condition.map(condition_map)
    
    unique_values = df.grade.unique()
    grade_map = {}
    
#     Creates grade map for later reference
    for unique_value in unique_values:
        grade_map[int(unique_value[:2])] = unique_value[2:].strip()

#    Function to take only numeric value of grade rating
    def grade_change(ds):
        return int(ds[:2])

#     Takes only numerical value of grade rating    
    df.grade = df.grade.map(grade_change)
    
    return (df, condition_map, grade_map)


def omit_outliers_dups(df):
    '''
    Removes outliers and duplicate entries
    
    Input: dataframe
    Output: Modified dataframe
    '''
    
#     Omits outliers
    df2 = df[(df.sqft_living <= 8000)
           & (df.price <= 4000000)
           & (df.bedrooms != 33)].copy()
    
#     Drops Duplicates
    df2.drop_duplicates(inplace=True)
    
    return df2


def cross_val(X_train, y_train):
    '''
    Returns the Rsquared value from 10 randomly selected training
    and test sets from the set provided
    
    inputs: X_train, y_train
    output: array of Train Rsquared values,
            array of Test Rsquared values,
            timing statistics
    '''
    
#     Start a linear regression model
    lr = LinearRegression()
    
#     Create a random splitter, randomly selects 10 sets of train:test sets
    splitter = ShuffleSplit(n_splits=10, test_size=0.20, random_state=1)

#     Creates models for the 10 sets, returns Rsqared score for each
    baseline_scores = cross_validate(
        estimator=lr,
        X=X_train,
        y=y_train,
        return_train_score=True,
        cv=splitter
    )

    return baseline_scores


def model_summary(X_train, y_train):
    '''
    Creates a linear regression model using statsmodels
    
    Inputs: X_train, y_train
    Output: linear regression model
    '''
    
#     Independant Variables with a column of 1's added for intercept
    predictors = sm.add_constant(X_train)
    
#     Creates linear regression model
    model = sm.OLS(y_train, predictors).fit()

    return model


def log_columns(data, columns=0):
    '''
    Takes in a dataseries or dataframe and logs columns in input variable columns.
    If dataseries, input variable columns isn't necessary
    If dataframe, columns is by default all columns
    
    Input: data, columns (default all columns)
    Ouptut: data with called out columns changed to log form
    '''
    
#     Checks if data is dataseries or dataframe, also if input column variable is empty
    try:
#         Sets columns variable to all columns if columns varible is empty
        if columns==0:
            columns = data.columns
#         Notes that data is not a dataseries
        ds = False
    except:
#         notes that data is a dataseries
        ds = True
#         Crates variable for dataseries name 
        name = data.name
    
    data_log = data.copy()
    
#     Logs columns, and changes column names
    if ds:
        new_name = 'log_' + name
        data_log = np.log(data_log)
        data_log.rename(new_name, inplace=True)
    else:
        new_names = {column: ('log_' + column) for column in columns}
        data_log[columns] = np.log(data_log[columns])
        data_log.rename(columns=new_names, inplace=True)
    
    return data_log


def correlation_check(X_train):
    '''
    Returns correlation dataseries
    
    input: X
    output: correlation dataseries for correlations > .3
    '''
#     Create correlation matrix, manipulate them into one column
    df = X_train.corr().stack().reset_index().sort_values(0, ascending=False)
    
#     Creates variables for new index
    df['pairs'] = list(zip(df.level_0, df.level_1))

#     Sets new index to pairs
    df.set_index(['pairs'], inplace = True)

    #drops level columns
    df.drop(columns=['level_1', 'level_0'], inplace = True)

    # rename correlation column as cc rather than 0
    df.columns = ['cc']

    # drop duplicates.
    df.drop_duplicates(inplace=True)
    return df[(df.cc>.3) & (df.cc <1)]

def mse(model, X, y):
    '''
    Returns mean squared error
    
    inputs: model, X, y
    output: mean squared error
    '''
    pred = model.predict(sm.add_constant(X))
    
    return (mean_squared_error(y, pred))