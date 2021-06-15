import pandas as pd
import numpy as np
import os

from env import host, user, password
from pydataset import data

from scipy import stats
from math import sqrt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression, SelectKBest, RFE

## ACQUIRE MY DATA

# Create helper function to get the necessary connection url.

def get_connection(db_name):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    from env import host, user, password
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'


def get_zillow():
    '''
    This function pulls in the zillow dataframe from my sql query. I specified
    columns from sql to bring in, I narrowed the transaction dates during the
    hot dates (May-August) for 2017, and only uses single unit dwellings.
    '''

    sql_query = '''
    SELECT bedroomcnt, 
    bathroomcnt,
    calculatedfinishedsquarefeet, 
    taxvaluedollarcnt, 
    yearbuilt, 
    taxamount, 
    fips,
    parcelid
    FROM properties_2017
    JOIN predictions_2017 USING(parcelid)
    JOIN propertylandusetype USING(propertylandusetypeid)
    WHERE transactiondate BETWEEN '2017-05-01' AND '2017-08-31' 
    AND propertylandusetypeid IN (260,261,262,263,264,266,273,276,279);
    '''
    
    return pd.read_sql(sql_query, get_connection('zillow'))


def get_info(df):
    '''
    This function takes in a dataframe and prints out information about the dataframe.
    '''

    print(df.info())

    print('------------------------')

    print('------------------------')
    print('This dataframe has', df.shape[0], 'rows and', df.shape[1], 'columns.')
    print('------------------------')
        
    print('Null count in dataframe:')
    print('------------------------')
    print(df.isnull().sum())

    print(' Dataframe sample:')
    return df.sample(3)



## GET VALUE COUNTS
def value_counts(df, column):
    '''
    This function takes in a dataframe and list of columns and prints value counts for each column.
    '''
    for col in column:
        print(col)
        print(df[col].value_counts())
        print('-------------')




# PREPARE MY DATA

def clean_zillow(df):
    '''
    This function takes in the zillow dataframe and cleans and prepares it by dropping nulls, dropping
    duplicates, replacing whitespaces, renaming columns and creating a new tax rate column.
    '''
    # change whitespaces to nan
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # drop all nulls
    df.dropna(inplace=True)

    # drop any duplicates from the dataframe
    df.drop_duplicates(inplace=True)

    # now that we've been able to drop any houses with duplicate parcel ids, we can drop the column
    df.drop(columns='parcelid', inplace=True)

    # this section addresses my fips code and 
    df['fips'] = df['fips'].astype(str)
    df.loc[df['fips'].str[0] == '6','state'] = 'California'
    df.loc[df['fips'].str.contains('111'),'county'] = 'Ventura'
    df.loc[df['fips'].str.contains('037'),'county'] = 'Los Angeles'
    df.loc[df['fips'].str.contains('059'),'county'] = 'Orange'
    df['fips'] = df['fips'].astype(float)
    
    # let's rename our columns so they are more clear
    df.rename(columns={'bedroomcnt': 'num_bedroom', 
                     'bathroomcnt': 'num_bathroom',
                     'calculatedfinishedsquarefeet': 'finished_sqft',
                     'taxvaluedollarcnt': 'tax_value',
                     'yearbuilt': 'build_year',
                     'taxamount': 'tax_amount'}, inplace=True)

    df['tax_rate'] = (df['tax_amount']/df['tax_value'] * 100)

    return df




# Address any outliers in my data with no less than 0

def outlier_bounds(df, columns):
    '''
    calculates the lower and upper bound to locate outliers in variables and then removes them.
    '''
    for i in columns:
        quartile1, quartile3 = np.percentile(df[i], [25,75])
        IQR_value = quartile3 - quartile1
        lower_bound = max((quartile1 - (1.5 * IQR_value)),0)
        upper_bound = min((quartile3 + (1.5 * IQR_value)),max(df[i]))
        print(f'For {i} the lower bound is {lower_bound} and  upper bound is {upper_bound}')
        
        df = df[(df[i] <= upper_bound) & (df[i] > 0)]
    
    return df




def split_data(df):
    '''
    This function takes in a dataframe and splits it into train, test, and validate dataframes for my model
    '''

    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)

    print('train--->', train.shape)
    print('validate--->', validate.shape)
    print('test--->', test.shape)
    return train, validate, test




## MY MINMAX SCALER FUNCTION
def min_max_scaler(X_train, X_validate, X_test, numeric_cols):
    """
    this function takes in 3 dataframes with the same columns,
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler.
    it returns 3 dataframes with the same column names and scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    #
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=numeric_cols).set_index(
        [X_train.index.values]
    )

    X_validate_scaled = pd.DataFrame(
        X_validate_scaled_array, columns=numeric_cols
    ).set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=numeric_cols).set_index(
        [X_test.index.values]
    )

    # Overwriting columns in our input dataframes for simplicity
    for i in numeric_cols:
        X_train[i] = X_train_scaled[i]
        X_validate[i] = X_validate_scaled[i]
        X_test[i] = X_test_scaled[i]

    return X_train, X_validate, X_test





# KBEST FUNCTION

def select_kbest(X_train_scaled, y_train, no_features):
    '''
    This function takes in scaled data and number of features and returns the top features
    '''
    
    # using kbest
    f_selector = SelectKBest(score_func=f_regression, k=no_features)
    
    # fit
    f_selector.fit(X_train_scaled, y_train)

    # display the two most important features
    mask = f_selector.get_support()
    
    return X_train_scaled.columns[mask]



## RFE FUNCTION

def rfe(X_train_scaled, y_train, no_features):
    '''
    This function takes in scaled data and number of features and returns the top features
    '''
    
    # now using recursive feature elimination
    lm = LinearRegression()
    rfe = RFE(estimator=lm, n_features_to_select=no_features)
    rfe.fit(X_train_scaled, y_train)

    # returning the top chosen features
    return X_train_scaled.columns[rfe.support_]