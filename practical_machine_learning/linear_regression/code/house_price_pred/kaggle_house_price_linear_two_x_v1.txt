# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:29:57 2017

@author: 753914
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import numpy as np
from sklearn.model_selection import train_test_split
import os

os.chdir('C:/Users/753914/Desktop/my_technical/machine_learning/practical_machine_learning/linear_regression/code/house_price_pred')

#Read the csv file and store it in pandas dataframe
df_train_data = pd.read_csv('C:/Users/753914/Desktop/my_technical/machine_learning/practical_machine_learning/linear_regression/code/house_price_pred/train.csv')
df_test_data = pd.read_csv('C:/Users/753914/Desktop/my_technical/machine_learning/practical_machine_learning/linear_regression/code/house_price_pred/test.csv')

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
