import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
import math

plt.style.use('bmh')
    
def base_plots(x, y):
    '''
    Figure to plot scatter and histograms for price and sqft_living for base model
    Inputs: x=price, y=sqft_living
    Output: Scatter plot of price vs. sqft_living,
            histograms of price and sqft_living
    '''
    
    fig, axes = plt.subplots(figsize=(18,6), ncols=3, nrows=1)
    
#     Price vs. Square Foot Living Scatter Plot
    axes[0].scatter(x,y)
    axes[0].set_ylabel('Price')
    axes[0].set_xlabel('Square Foot Living')
    axes[0].set_title('Square Foot Living vs. Price')
    
#     Square Foot Living Histogram
    axes[1].hist(x, bins=30)
    axes[1].set_xlabel('Sqare Foot Living')
    axes[1].set_ylabel('count')
    axes[1].set_title('Square Foot Living')
    
#     Price Histogram
    axes[2].hist(y, bins=30)
    axes[2].set_xlabel('Price')
    axes[2].set_ylabel('count')
    axes[2].set_title('Price')
    
    
def base_plots_log(x, y):
    '''
    Figure to plot log scatter and histograms for price and sqft_living for base model
    Inputs: x=price, y=sqft_living
    Output: Scatter plot of log price vs. log sqft_living,
            histograms of log price and log sqft_living
    '''
    
    fig, axes = plt.subplots(figsize=(18,6), ncols=3, nrows=1)
    
#     takes log of variable
    x_log = np.log(x)
    y_log = np.log(y)
    
#     log Price vs log Square Foot Living Scatter Plot
    axes[0].scatter(x_log,y_log)
    axes[0].set_ylabel('log(Price)')
    axes[0].set_xlabel('log(Square Foot Living)')
    axes[0].set_title('Square Foot Living vs. Price')
    
#     Log Square Foot Living Histogram
    axes[1].hist(x_log, bins=30)
    axes[1].set_xlabel('log(Sqare Foot Living)')
    axes[1].set_ylabel('count')
    axes[1].set_title('Log Square Foot Living')
    
#     Log Price Histrogram
    axes[2].hist(y_log, bins=30)
    axes[2].set_xlabel('log(Price)')
    axes[2].set_ylabel('count')
    axes[2].set_title('Log Price')
    
    
def all_scatters(X_train, y_train):
    '''
    Creates scatterplots for all independant variables compared
    to our independent variable price
    
    inputs: (X_train, y_train)
    outputs: x vs. y Scatter plots
    '''
    
#     Calculates number of rows for figure, maxing at 3 columns
    nrows = math.ceil(len(X_train.columns)/3)
    
#     Creates figure
    fig = plt.figure(constrained_layout=True, figsize=(12, 4*nrows))
    
#     Creats figure layout
    gs = GridSpec(nrows, 3, figure=fig)
    
#     Gets name of target variable
    target_var = y_train.name
    
    axes = []
    for i, column in  enumerate(X_train.columns):
        axes.append(fig.add_subplot(gs[i//3, i%3]))
        axes[i].scatter(X_train[column], y_train)
        axes[i].set_xlabel(column)
        axes[i].set_ylabel(target_var)
        axes[i].set_title(target_var + ' vs. ' + column)
        

def high_resid_plots(X, y, model, resid_cutoff):
    '''
    Creates scatter plots for all features and target variable containing 
    entries with residuals greater then assigned resid_cutoff
    
    Inputs: X = features
            y = target variable
            model = linear regression model
            resid_cutoff = high residual mark
    
    Outputs: Scatter plots of all features and target variable of entries
             greater than the resid_cutoff
    '''
#     Concatenates features and target varibale
    df = pd.concat([X, y], axis=1)
    
#     Calculates number of rows in figure, maxing at 3 columns
    nrows = math.ceil(len(df.columns)/3)
#     Take column names of features to be plotted against the residuals
    columns = df.columns
    
#     Creates residual column in dataframe
    df['resid'] = model.resid
#     Takes only entries with residuals greater than resid_cutoff
    df = df[df.resid > resid_cutoff]
    
#     Creates figure
    fig = plt.figure(constrained_layout=True, figsize=(12 ,nrows*4))
    
#     Creates figure layout
    gs = GridSpec(nrows, 3, figure=fig)
    
    axes = []
#     Iterates through columns, plotting feature vs. resid
    for i, column in  enumerate(columns):
        axes.append(fig.add_subplot(gs[i//3, i%3]))
        axes[i].scatter(df[column], df.resid)
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Residuals')
        axes[i].set_title('High Residuals vs. ' + column)    
        


def normality_plots(resid):
    '''
    Plots histogram and QQ-plot for residuals
    
    Input: resid = model residuals
    '''
#     Creates figure
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 8))
    
#     Creates residual histogram
    sns.histplot(resid, kde=True, ax=axes[0])
    axes[0].set_title('Residual Histogram')
    axes[0].set_xlabel('Residuals')
    axes[0].set_ylabel('Count')
    
#     Creates QQ-plot
    sm.graphics.qqplot(resid, dist=stats.norm,
                      line='45', fit='True', ax=axes[1])
    axes[1].set_title('Residual QQ-plot')        
        
        
        
def homoskedasticity_plot(y, model):
    '''
    Creates scatterplot of residuals against expected
    outcome to check for homoskedasticity
    
    Inputs: y = target variable/real outcomes
            model = linear regression model
    '''
    
#     Finds residuals of model
    residuals = model.resid
#     Names dataseries
    residuals.name = 'resid'
#     Concatenates y and residuals
    df = pd.concat([y, residuals], axis=1)
    
#     Creates scatterplot
    sns.scatterplot(x=df[y.name], y=df.resid)
    plt.title('Residuals vs. Target')