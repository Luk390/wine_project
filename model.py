# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:52:31 2020

@author: lukeb
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats as stats

df = pd.read_csv('datasets_4458_8204_winequality-red.csv')

x = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
y = df['quality']

#Check whether distrobution of quality is normal to satisfy linear regression assumptions
#create figure
figure = plt.figure(figsize=(10,8))
figure.subplots_adjust(hspace=1.5)

gs = figure.add_gridspec(6,6)
#Create histogram of quality
ax1 = figure.add_subplot(gs[0:5, :])
ax1.set_title("Histogram of quality")
ax1.set_ylabel("Frequency")
sns.distplot(y, bins=6, kde=False)

#create boxplot of quality
ax2 = figure.add_subplot(gs[5, :])
ax2.set_title("Box plot of quality")
ax2.set_ylabel("Quality")
sns.boxplot(data=y, orient='h')

#Quality exhibits normal distribution.
#Investigate relationships between independent variables and dependent variable

for variable in x:
    print(variable)
    plt.scatter(x[variable], y)
    plt.show()

#Logistic Regression


"""
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

plt.title('Residual Analysis')
#plt.scatter(y_predict, residuals, alpha=0.4)
plt.scatter(y_predict2, residuals2, alpha=0.4)
plt.show()


Residual graphs show approximate homoscedasticity. So it doesn't violate BLUE


#OLS regression
x_train_w_constant = sm.add_constant(x_train)
model = sm.OLS(y_train, x_train_w_constant)

results = model.fit()
print(results.summary())

OLS model r-squared 0.368, sklearn r-squared 0.368(train), 0.15(test data)
So not a great deal of predictive power held within these features. Logistic 
regression based on "good" vs "bad" wine may give better results.


#Logistic Regression

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

log_model = LogisticRegression()
log_model.fit(x_train_scaled, y_train)

print(log_model.score(x_train_scaled, y_train))
print(log_model.score(x_test_scaled, y_test))


"""





















