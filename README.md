# Project Overview

* Created a model which can classify whether a wine is good (>= 7/10) or bad (< 7/10), with 95% accuracy from the physiochemical properties.
* trained and optimised several models including Logistic Regression, Linear SVC, SVC with rbf-kernel, Random Forest Classifier and Gradient Boosting Classifier
* Built a basic API using Flask

The best models were the Random Forest Classifier and Gradient Boosting Classifier

![alt text](https://github.com/Luk390/wine_project/blob/master/images/roc_curve.png "ROC curve for RFC and GBC")

## Code and Resources used

* Python: 3.7
* Packages: Pandas, Numpy, Scikit-Learn, Matplotlib, Seaborn, flask, json, pickle, missingno
* Flask API: https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2
* Project formatting: https://www.youtube.com/playlist?list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t
* Data source: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

## Data collection/cleaning

The data was extracted directly from kaggle and required no cleansing.

## EDA

The distribution of the target variable revealed that there were small numbers of wines at the extreme ends of the dataset so the decision was made to split the dataset into good and bad wines. 

![alt text](https://github.com/Luk390/wine_project/blob/master/images/quality%20bar%20plot.png "Bar plot of quality")

The EDA revealed that there was some slight clustering of the target variable indicating there was the possibility of predictive power in the dataset.

![alt text](https://github.com/Luk390/wine_project/blob/master/images/scatter%20matrix%20of%20independent%20variables.png "Scatter Matrix of Independent Variables")

It also revealed that there were some issues with normality but little issue with multicollinearity. Both provided an accuracy of 95% but the Gradient Boosting Classifier provided a slightly better AUC score.



![alt text](https://github.com/Luk390/wine_project/blob/master/images/heatmap%20of%20correlations.png "Heatmap of correlations between variables")

## Models

The dataset was preprocessed by splitting the data into a training and testing set and scaled using sklearn's StandardScaler().

The following models were built with default parameters:
* Logistic Regression
* Linear SVC
* SVC with rbf-kernel
* Random Forest Classifier
* Gradient Boosting Classifier

|  Model                             | Accuracy(%) |
| :--------------------------------: | :---------: |
| Logistic regression                | 88          |
| Linear SVC(Support vector machine) | 89          |
| SVC with rbf kernel                | 88          |
| Random Forest Classifier           | 91          |
| Gradient Boosting Classifier       | 89          |

The SVC with rbf-kernel Random Forest Classifier and Gradient Boosting Classifier were then optimised to see which provided the best performance.

![alt text](https://github.com/Luk390/wine_project/blob/master/images/gbc_confusion_matrix.png "Confusion matrix of Gradient Boosting Classifier")
![alt text](https://github.com/Luk390/wine_project/blob/master/images/roc_curve.png "ROC curve for RFC and GBC")

## Productionisation

A flask API endpoint was then created and hosted on a local server. This takes in a request with a list of physiochemical properties and returns a prediction of whether the wine is good or bad.
