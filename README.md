# View my completed powerpoint presentation [here](https://drive.google.com/file/d/1l5s-7Zic5we-YdFOXZaMecIo65LaYLRg/view?usp=sharing)!



## Project Goals
1. Create a model that predicts tax values of single unit properties
2. Determine the states and counties the properties are located in
3. Determine the distribution of tax rates for each county

## Replicate my Project
    1. python
    2. pandas
    3. scipy
    4. sci-kit learn
    5. numpy
    6. matplotlib.pyplot
    7. seaborn
* Steps to recreate
    1. Clone this repository
    - https://github.com/DesireeMcElroy/zillow-regression-project
    2. Import dataframe from SQL

## Key Findings
1. Finished squarefeet is the most important driver of tax value with the highest correlation
2. County surprisingly did not have a major correlation to value
3. Bedrooms and bathrooms had no correlation with each other and were able to be separated and binned accordingly
4. Model needs improvement since it performed badly on test set

## Drawing Board
View my trello board [here](https://trello.com/b/T8B0pTSp/zillow-regression-project).

------------

I want to examine these possibilities:
1. Does the tax value increase as the number of bathrooms increase?
2. Does the tax value increase as the number of bedrooms increase?
3. Does the tax value increase as the total square feet increases?
4. Does the tax value decrease as build year decrease?
5. Is there a difference between tax values based on the FIPS county?

I will verify my hypotheses using statistical testing and where I can move forward with the alternate hypothesis, I will use those features in exploration. By the end of exploration, I will have identified which features are the best for my model.

During the modeling phase I will establish two baseline models and then use my selected features to generate a regression model. I will evaluate each model with the highest correlated features to minimize error and compare each model's performance to the baseline. Once I have selected the best modeling method, I will subject it to the training sample and evaluate the results.


## Data Dictionary

#### Target
Name | Description | Type
:---: | :---: | :---:
tax_value | The assesed value of the property for tax purposes | float
#### Features
Name | Description | Type
:---: | :---: | :---:
num_bathrooms | The number of bathrooms a property has | float
num_bedrooms | The number of bedrooms a property has | float
total_sqft | The square footage of a property | float
county | The county the property is located in | int
fips | The FIPS county code of the property location | float
#### Other data
Name | Description | Data Type
:---: | :---: | :---:
tax_rate | Observation tax amount divided by the tax value  | float
state | The state the property is located in | object

## Results
The two tweedieregressor models with degrees of 0 or 1 outperformed the baseline:
1. Lowest RMSE value of 186,000, almost 25,000 less than my baseline
2. Performed poorly on test dataset

Need to improve model since it performed so poorly on test dataset.

## Recommendations
1. I would conduct a lot more statistical testing to be sure I binned my data correctly.
2. Adjust the amount of my features used in my model to see which helps my model perform better.
3. Dive thoroughly into why my model did not test as well.
4. Take a deeper look at additional features such as garages and pools to see if they help my model perform better.
5. Find ways to fill in missing data to see if it helps model improvement.


Resources:
https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt