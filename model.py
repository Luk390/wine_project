# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:52:31 2020

@author: lukeb
"""

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
df = pd.read_csv('datasets_4458_8204_winequality-red.csv')

print(df.columns)

x = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
y = df['quality']

#sklearn Multiple Linear regression

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

mlr = LinearRegression()
mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)
y_predict2 = mlr.predict(x_train)
print(mlr.score(x_train, y_train))
print(mlr.score(x_test, y_test))

residuals = y_predict - y_test
residuals2 = y_predict2 - y_train
"""
plt.title('Residual Analysis')
#plt.scatter(y_predict, residuals, alpha=0.4)
plt.scatter(y_predict2, residuals2, alpha=0.4)
plt.show()
"""
"""
Residual graphs show approximate homoscedasticity. So it doesn't violate BLUE

"""
#OLS regression
x_train = sm.add_constant(x_train)
model = sm.OLS(y_train, x_train)

results = model.fit()
print(results.summary())

"""OLS model r-squared 0.368, sklearn r-squared 0.368(train), 0.15(test data)
So not a great deal of predictive power held within these features. Logistic 
regression based on "good" vs "bad" wine may give better results.
"""


