import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.feature_selection import f_regression, SelectKBest, RFE
from sklearn.preprocessing import PolynomialFeatures

from math import sqrt

import warnings
warnings.filterwarnings('ignore')




def plot_residuals(df, y, yhat):
    '''
    This function takes in a dataframe, the y target variable 
    and the yhat (model predictions) and creates columns for residuals
    and baseline residuals. It returns a graph of both residual columns.
    '''

    # create a residual column
    df['residual'] = (yhat - y)

    # create a residual baseline column
    df['residual_baseline'] = (y.mean() - y)
    
    fig, ax = plt.subplots(figsize=(13,7))

    ax.hist(df.residual_baseline, label='baseline residuals', alpha=.6)
    ax.hist(df.residual, label='model residuals', alpha=.6)
    ax.legend()





def regression_errors(df, y, yhat):
    '''
    
    '''
    
    SSE = mean_squared_error(y, yhat)*len(df)


    MSE = mean_squared_error(y, yhat)


    RMSE = sqrt(mean_squared_error(y, yhat))


    ESS = sum((yhat - y.mean())**2)
    TSS = sum((y - y.mean())**2)

    # compute explained variance
    R2 = ESS / TSS
    
    print('SSE is:', SSE)
    print('ESS is:', ESS)
    print('TSS is:', TSS)
    print('R2 is:', R2)
    print('MSE is:', MSE)
    print('RMSE is:', RMSE)



def baseline_mean_errors(df, y, yhat_baseline):
    
    SSE_baseline = mean_squared_error(y, yhat_baseline)*len(df)
    
    MSE_baseline = mean_squared_error(y, yhat_baseline)
    
    RMSE_baseline = sqrt(mean_squared_error(y, yhat_baseline))
    
    
    print('Baseline SSE is:', SSE_baseline)
    print('Baseline MSE is:', MSE_baseline)
    print('Baseline RMSE is:', RMSE_baseline)




def better_than_baseline(df, y, yhat, yhat_baseline):

        RMSE = sqrt(mean_squared_error(y, yhat))
    
        RMSE_baseline = sqrt(mean_squared_error(y, yhat_baseline))
        
        if RMSE < RMSE_baseline:
            print('The model performs better than the baseline')
        elif RMSE > RMSE_baseline:
            print('The baseline performs better than the model')
        else:
            print('error, it is possible they are equal')
        
        
        return RMSE, RMSE_baseline





def select_kbest(X_train, y_train, no_features):
    
    # using kbest
    f_selector = SelectKBest(score_func=f_regression, k=no_features)
    
    # fit
    f_selector.fit(X_train, y_train)

    # display the two most important features
    mask = f_selector.get_support()
    
    return X_train.columns[mask]



def rfe(x_train, y_train, no_features):
    # now using recursive feature elimination
    lm = LinearRegression()
    rfe = RFE(estimator=lm, n_features_to_select=no_features)
    rfe.fit(x_train, y_train)

    # returning the top chosen features
    return x_train.columns[rfe.support_]