# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:29:57 2017

@author: 753914
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import StandardScaler

os.chdir('C:/Users/753914/Desktop/my_technical/machine_learning/practical_machine_learning/linear_regression/code/house_price_pred')

#Read the csv file and store it in pandas dataframe
df_train_data = pd.read_csv('C:/Users/753914/Desktop/my_technical/machine_learning/practical_machine_learning/linear_regression/code/house_price_pred/train.csv')
df_test_data = pd.read_csv('C:/Users/753914/Desktop/my_technical/machine_learning/practical_machine_learning/linear_regression/code/house_price_pred/test.csv')

#===================================================================================
#Analyse the values present in the dependent variable
#===================================================================================

print(df_train_data['SalePrice'].describe())
#count      1460.000000
#mean     180921.195890
#std       79442.502883
#min       34900.000000
#25%      129975.000000
#50%      163000.000000
#75%      214000.000000
#max      755000.000000

#plot historgram to see the result visually
sns.distplot(df_train_data['SalePrice']);
#result shows whehter it has peakedness and positive skewness / negative skewness

#skewness tells whether test data has any skewness. If it's 1 the no peak, >1 posivite pead, <1 negative peak.
print("Skewness: %f" % df_train_data['SalePrice'].skew())

#===================================================================================
#find the relationship between each IV(numerical/categorical value) and DV.
#===================================================================================

#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train_data['SalePrice'], df_train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#Relation is linear i.e when x increases y also increases

var = 'TotalBsmtSF'
data = pd.concat([df_train_data['SalePrice'], df_train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
                 
#find the relationship betwen each IV(categorical value) and DV
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train_data['SalePrice'], df_train_data[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

        
var = 'YearBuilt'
data = pd.concat([df_train_data['SalePrice'], df_train_data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
          
          
#===================================================================================
#Correlation heap map shows how variables are co-related.
#Helps to find relation between IV and DV
#Also find relation between IV to avoid multicollinearity
#===================================================================================
#correlation matrix
corrmat = df_train_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
           
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#'OverallQual', 'GrLivArea' and 'TotalBsmtSF' are strongly correlated with 'SalePrice'.
#'GarageCars' and 'GarageArea' are also some of the most strongly correlated variables.
#Therefore, we just need one of these variables in our analysis 
#'TotalBsmtSF' and '1stFloor' also seem to be twin brothers. We can keep 'TotalBsmtSF' 
#'TotRmsAbvGrd' and 'GrLivArea', twin brothers again.
#It seems that 'YearBuilt' is slightly correlated with 'SalePrice'. 

#Take only the selected variables and plot scatter plot betwen IV and DV and find the relationship.
#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
#sns.pairplot(df_train_data[cols], size = 2.5)
plt.show();
        
#===============================================================================
#Missing Data
#===============================================================================
total = df_train_data.isnull().sum().sort_values(ascending=False)
percent = (df_train_data.isnull().sum()/df_train_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))
#when more than 15% of the data is missing, we should delete the corresponding variable

#dealing with missing data
df_train_data = df_train_data.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train_data = df_train_data.drop(df_train_data.loc[df_train_data['Electrical'].isnull()].index)
print(df_train_data.isnull().sum().max()) #just checking that there's no missing data missing...
print(df_train_data)


#===============================================================================
#Outliars
#===============================================================================
#Univariate analysis
#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train_data['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

#Low range values are similar and not too far from 0.
#High range values are far from 0 and the 7.something values are really out of range.

#Bivariate analysis
#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train_data['SalePrice'], df_train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deleting points
df_train_data.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train_data = df_train_data.drop(df_train_data[df_train_data['Id'] == 1299].index)
df_train_data = df_train_data.drop(df_train_data[df_train_data['Id'] == 524].index)

#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train_data['SalePrice'], df_train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
                 

           
'''
#Display the list of columns present in the dataframe
print(df_train_data.columns)

#Split the training data into test and train using sklearn split method
#return dataframe with test and train split data
#random_state=0 always take same set of data for test and train
df_train, df_test = train_test_split(df_train_data, test_size = 0.2,random_state=0)

#Assign only the needed x(independent variable) to X_train
#Assign only the needed y(dependent variable) to Y_train
#Assign the test data to X_test and Y_test
X_train = df_train.loc[:,('GrLivArea','YearBuilt','Id')]
Y_train = df_train.loc[:,('SalePrice','Id')]
X_test = df_test.loc[:,('GrLivArea','YearBuilt','Id')]
Y_test = df_test.loc[:,('SalePrice','Id')]

#Create the object for linear regression algorithm
regr = linear_model.LinearRegression()

#fit the train data into linear regression model.
regr.fit(X_train.loc[:,('GrLivArea','YearBuilt')], Y_train.SalePrice)

#It tells for increase in each x for one unit how much y increases. i.e (slope)
print('Coefficients: \n', regr.coef_)

#Predict Y for  test data with two x variable
y_pred_df=pd.DataFrame(regr.predict(X_test.loc[:,('GrLivArea','YearBuilt')]),df_test.Id)

#Predict the result for actual test data
y_pred_df_real_test=pd.DataFrame(regr.predict(df_test_data.loc[:,('GrLivArea','YearBuilt')]),df_test_data.Id)

#Assign columns name to final result dataframe
y_pred_df.columns = ['predicted_price']
y_pred_df_real_test.columns = ['SalePrice']

#Select saleprice and id to check the actual price and predicted price in same row
x_df=pd.DataFrame({'actual_sales':df_test.SalePrice.values},df_test.Id)

#Print actual price and predicted price in one row.
result=pd.concat([y_pred_df,x_df],axis=1,join='inner')
print(result)

#Final the accuracy of the model using root mean square method.
#Sqrt of mean of (predicted y - actual y)^^2
#it gives the error. Minimise the error to get more accuracy.
print("Root Mean squared error: %.2f"
      % np.sqrt(np.mean((y_pred_df.predicted_price.values.astype(float)- Y_test.SalePrice.astype(float)) ** 2)))


# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_train.loc[:,('GrLivArea','YearBuilt')], Y_train.SalePrice))

#Write the output to csv
y_pred_df_real_test.to_csv('output.csv')
# Plot outputs
plt.scatter(X_test.GrLivArea, Y_test.SalePrice,  color='black')
plt.plot(X_test.GrLivArea, y_pred_df.predicted_price, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()
'''