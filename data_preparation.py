import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
from sklearn.feature_selection import RFECV
    
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


def base_model(x_train, y_train):
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
        X=x_train,
        y=y_train,
        return_train_score=True,
        cv=splitter
    )

    return baseline_scores

def omit_outliers(X_train, y_train):
    df = pd.concat([X_train, y_train], axis=1)
    
    df = df[(df.sqft_living <= 8000)
           & (df.bedrooms != 33)]
    
    y = df.price
    X = df.drop(columns='price', axis=1)
    
    return (X, y)


def second_model_prep(X_train, y_train, categoricals):
    '''
    Creates dummy variables for categorical data.
    Standardizes continuous variables.
    Concatenates the dummy variables and continuous variables
    into one dataframe.
    
    inputs: (X_train, y_train, Categorical_columns)
    '''
    
#     Create dataframe with just category variables and changes
#     them to type category
    cat_df = X_train[categoricals].astype('category')

#     Creates dummy variables and drops firts column of each
#     set of dummy variables to help prevent multicollinearity
    cat_dummies = pd.get_dummies(cat_df, drop_first=True)

#     Drops categorical variables from original dataset
    X_train_cont = X_train.drop(categoricals, axis=1).copy()
    
#     Stardardizes numerical columns
    X_train_stand = StandardScaler().fit_transform(X_train_cont)
    X_train_stand = pd.DataFrame(X_train_stand, 
                                 columns=X_train_cont.columns,
                                 index=X_train_cont.index)
    
#     Concatinates dummy variables and standardized continuous variables
    X_train_prep = pd.concat([X_train_stand, cat_dummies], axis=1)    
    
    return X_train_prep


def multi_model(X_train, y_train):
    '''
    Returns the Rsquared value from 10 randomly selected training
    and test sets from the set provided

    inputs: (X_train (standardized and OHE), 
             y_train)
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

    return (baseline_scores)


def create_cont_interactions(X_train, X_train_OHE, continuous):
    '''
    Creates interactions between continuous variables in X_train
    
    input: (Original X_train dataframe with unstandardized variables, 
            X_train dataframe with OHE variables,
            continuous variables to have features created)
            
    output: Dataframe of original columns and interactions
    '''
    
#     creatures dataframe for continuous variables to have interactions created between
    features_for_poly = X_train[continuous]
    
#     Drops continuous columns to later be concatenated to new variables
    OHE_features = X_train_OHE.drop(columns=continuous, axis=1)
    
#     Creatures PolynomialFeatures model, of degree 2 and
#     only between features
    poly = PolynomialFeatures(degree=2, interaction_only=True)

#     Creates interactions
    X_train_interactions = poly.fit_transform(features_for_poly)
 
#     Standardizes Interaction columns
    X_train_interactions_stand = StandardScaler().fit_transform(X_train_interactions)

#     Gets feature names described by fit
    feature_names = poly.get_feature_names()

#     Creates list of feature names described in fit that describe each individual feature
    interaction_ind_features = [('x' + str(i)) for i, 
                                column in enumerate(features_for_poly.columns)]

#     Creates mapping of fit feature name to actual feature name
    interaction_map = {x: name for x, name in zip(interaction_ind_features,
                                                  features_for_poly.columns)}

#     Iterates through all features created by polynomial fit and assigns
#     understandable names
    column_names = []
    for feature in feature_names:
        if feature == '1':
            new_name = 'const'
        else:
            split_feature = feature.split(' ')
            new_name = interaction_map[split_feature[0]]
            if (len(split_feature) > 1):
                new_name = new_name + '*' + interaction_map[split_feature[1]]
        column_names.append(new_name)

#      Creates a dataframe of new features and drops the intercept column of 1's
    interaction_df = (pd.DataFrame(X_train_interactions_stand, columns=column_names, 
                                   index=X_train.index).drop(columns='const', axis=1))
    
#     returns concatenated standardized interaction df and OHE df
    return pd.concat([interaction_df, OHE_features], axis=1)


def create_all_interactions(X_train, X_train_OHE, continuous):
    '''
    Creates interactions between all variables in X_train
    
    input: (Original X_train dataframe with unstandardized variables, 
            X_train dataframe with OHE variables,
            continuous variables to be standardized)
            
    output: Dataframe with interactions between features
    '''
    
#     creatures dataframe of unstandardized continuous variables
    unstandardized_cont_features = X_train[continuous]
    
#     Pulls OHE features from dataframe
    OHE_features = X_train_OHE.drop(columns=continuous, axis=1)

#     Creates dataframe of features for interactions
    features_for_poly = pd.concat([unstandardized_cont_features, OHE_features], axis=1)
    
#     Creatures PolynomialFeatures model, of degree 2 and
#     only between features
    poly = PolynomialFeatures(degree=2, interaction_only=True)

#     Creates interactions
    X_train_interactions = poly.fit_transform(features_for_poly)

#     Gets feature names described by fit
    feature_names = poly.get_feature_names()

#     Creates list of feature names described in fit that describe each individual feature
    interaction_ind_features = [('x' + str(i)) for i, 
                                column in enumerate(features_for_poly.columns)]

#     Creates mapping of fit feature name to actual feature name
    interaction_map = {x: name for x, name in zip(interaction_ind_features,
                                                  features_for_poly.columns)}

#     Iterates through all features created by polynomial fit and assigns
#     understandable names
    column_names = []
    for feature in feature_names:
        if feature == '1':
            new_name = 'const'
        else:
            split_feature = feature.split(' ')
            new_name = interaction_map[split_feature[0]]
            if (len(split_feature) > 1):
                new_name = new_name + '*' + interaction_map[split_feature[1]]
        column_names.append(new_name)

#      Creates a dataframe of new features and drops the intercept column of 1's
    interactions_df = (pd.DataFrame(X_train_interactions, columns=column_names, 
                                   index=X_train.index).drop(columns='const', axis=1))

#     Drops all columns of all zeros
    interactions_df = interactions_df.loc[:, (interactions_df!=0).any(axis=0)]

#     Gets column names of continuous variables to standardize
    combs = combinations(continuous, 2)
    cols_to_standardize = []
    for comb in combs:
        cols_to_standardize.append(comb[0] + '*' + comb[1])
    for cont in continuous:
        cols_to_standardize.append(cont)
        for ohe in OHE_features.columns:
            cols_to_standardize.append(cont + '*' + ohe)
    
#     Standardizes continuous columns
    cols_to_standardize_df = interactions_df[cols_to_standardize]
    X_train_interactions_stand = StandardScaler().fit_transform(cols_to_standardize_df)
    X_train_interactions_stand_df =  (pd.DataFrame(X_train_interactions_stand, 
                                                   columns=cols_to_standardize_df.columns, 
                                                   index=X_train.index))
    
#     Gets only interactions between OHE variables
    X_train_OHE_interactions = interactions_df.drop(columns=cols_to_standardize, axis=1)
    
#     returns concatenated standardized interaction df and OHE df
    return pd.concat([X_train_interactions_stand_df, X_train_OHE_interactions], axis=1)


def recursive_elimination(X_train, y_train):
    '''
    Performs Recursive elimination on X_train to find features with best resulting
    Rsquared value
    
    inputs: (X_train, y_train)
    output: Prints columns that survived for final model
    '''

    lr = LinearRegression()
    splitter = ShuffleSplit(n_splits=5, test_size=0.25, random_state=1)
    
    # Instantiate and fit the selector
    selector = RFECV(lr, cv=splitter)
    results = selector.fit(X_train, y_train)
    
    for i, column in enumerate(X_train.columns):
        if results.support_[i] == True:
            print(column)
            
            
            