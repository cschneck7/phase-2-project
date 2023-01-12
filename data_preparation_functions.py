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


def rsquared_df(X, y, i, df=pd.DataFrame()):
    '''
    Returns dataframe with new model and old models rsquared values
    
    Inputs: X = model features
            y = target variable
            i = index
            df = df of previus model results,
                 (default = empty dataframe),
                 if first entry leave out
    
    Output: DataFrame of new and old models rsquared values
    '''
#     Performs cross validation on dataset
    cross_val_results = cross_val(X, y)
    
#     creates dictionary of current models rsquared values
    data = {'train_rsquared': [round(cross_val_results['train_score'].mean(), 5)],
            'test_rsquared': [round(cross_val_results['test_score'].mean(), 5)]}
#     Creates dataframe containing new models data
    new_data_df = pd.DataFrame(data, index=[i])
    
#     Creates dataframe containing old and new models rsquared values
#     Checks if this is first entry
    if df.empty:
        df = new_data_df
    else: 
#         if not, concantenates new and old data together
        df = pd.concat([df, new_data_df])
    
#     Returns DataFrame
    return df


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
#         Sets columns variable to all columns if columns variable is empty
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
        if type(columns) == str:
            new_names = {columns: ('log_' + columns)}
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
#     Gets predictions using X
    pred = model.predict(sm.add_constant(X))

#     returns mean squared error
    return (mean_squared_error(y, pred))



def outlier_percentage(X ,y, model, resid_cutoff, columns):
    '''
    Prints out the percentage of the dataframe that are outliers
    depending on the residual cutoff
    
    Inputs: X = features
            y = target variable
            model = linear regression model
            resid_cutoff = high residual mark
            columns = columns to be checked
    '''
#     Concatenates features and target varibale    
    df = pd.concat([X, y], axis=1)
    
#     Creates residual column in dataframe    
    df['resid'] = model.resid
    
#     Creates DataFrame of  only entries with residuals greater 
#     than resid_cutoff
    out_df = df[df.resid > resid_cutoff]
    
#     Finds number of entries in original dataframe
    len_df = len(y)

#     Iterates through columns, printing out percentage of outliers
    for c in columns:
#         Finds minimum value of outlier entries
        min_val = min(out_df[c])
#     Calculates percentage of outliers 
        perc = round(100*(len(df[df[c] >= min_val]))/len_df, 2)
#     Prints percentage
        print(c + ': ' + str(perc) + '%')
    

def model5_data(X_train, y_train, X_test, y_test, 
                model, resid_cutoff, column):
    '''
    Creates new X and y data for entries lower than the minimum
    value of column that is above or equal to the the resid_cutoff
    value
    
    Inputs: X_train = training set features
            y_train =  training set target variable
            X_test = test set features
            y_test =  test set target variable
            model = linear regression model
            resid_cutoff = high residual mark
            column = column deciding cutoff
             
    Output: Dictionary containing new training sets, test sets,
            and cutoff value for deciding column
    '''
    
#     Concatenates features and target varibale    
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    
#     Creates residual column in training dataframe    
    df_train['resid'] = model.resid
    
#     Find minimum value dependent on residual cutoff in training dataframe
    min_val = min(df_train[df_train.resid >= resid_cutoff][column])

#     Creates datafromes of all entries lower than min_val
    df_train = df_train[df_train[column] < min_val]
    df_test = df_test[df_test[column] < min_val]
    
#     Drops residual column from training set   
    df_train.drop(columns='resid', axis=1, inplace=True)
    
#     Creates new target variable data
    y_name = y_train.name
    new_y_train = df_train[y_name]
    new_y_test = df_test[y_name]
    
#     Creates new feature data
    new_X_train = df_train.drop(columns=y_name, axis=1)
    new_X_test = df_test.drop(columns=y_name, axis=1)
    
#     Creates dictionary containing all data
    data_dict = {'X_train': new_X_train,
                 'y_train': new_y_train,
                 'X_test': new_X_test,
                 'y_test': new_y_test,
                 'cutoff': min_val}
#     returns dictionary of new data and min_val
    return data_dict