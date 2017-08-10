# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:03:20 2017

@author: 753914
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:29:57 2017

@author: 753914
"""
import pandas as pd
import seaborn as sns
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

df_train_data = pd.read_csv('C:/Users/753914/Desktop/my_technical/machine_learning/practical_machine_learning/linear_regression/code/house_price_pred/train.csv')
df_test_data = pd.read_csv('C:/Users/753914/Desktop/my_technical/machine_learning/practical_machine_learning/linear_regression/code/house_price_pred/test.csv')


df_train, df_test = train_test_split(df_train_data, test_size = 0.2,random_state=0)


X_train = df_train.loc[:,('GrLivArea')]
Y_train = df_train.loc[:,('SalePrice')]
X_test = df_test.loc[:,('GrLivArea')]
Y_test = df_test.loc[:,('SalePrice')]

length = len(X_train)
test_length=len(X_test)

X_train = X_train.reshape(length,1)
Y_train = Y_train.reshape(length,1)
X_test = X_test.reshape(test_length,1)
Y_test = Y_test.reshape(test_length,1)





regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

print('Coefficients: \n', regr.coef_)


y_pred=regr.predict(X_test)

y_pred_df=pd.DataFrame(y_pred)
x_df=pd.DataFrame(df_test.Id)

print(x_df)

result=pd.concat([x_df,y_pred_df,pd.DataFrame(Y_test)],axis=1,join='inner')
#result=x_df.join(y_pred_df,how='inner')
#print(result.size)

print(y_pred)

print("Root Mean squared error: %.2f"
      % np.sqrt(np.mean((y_pred.astype(float) - Y_test.astype(float)) ** 2)))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_train, Y_train))

# Plot outputs
plt.scatter(X_test, Y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()
