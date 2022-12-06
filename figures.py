import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import numpy as np

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
    
    fig = plt.figure(constrained_layout=True, figsize=(12, 12))
    
    gs = GridSpec(3, 3, figure=fig)
    
#     Gets name of target variable
    target_var = y_train.name
    
    axes = []
    for i, column in  enumerate(X_train.columns):
        axes.append(fig.add_subplot(gs[i//3, i%3]))
        axes[i].scatter(X_train[column], y_train)
        axes[i].set_xlabel(column)
        axes[i].set_ylabel(target_var)
        axes[i].set_title(target_var + ' vs. ' + column)