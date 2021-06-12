import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt



def scatterplot(X, y, df):
    for i in X:
        sns.scatterplot(x=i, y=y, data=df)
        plt.show()


def lmplot(X, y, df):
    '''
    Takes in an an X, y and dataframe and returns a loop of lmplots.
    Tip: Add y as string
    '''
    for i in X:
        sns.lmplot(x=i, y=y, data=df, line_kws={'color': 'red'})
        plt.show()