# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 13:00:53 2020

@author: lukeb
"""
import pickle
import pandas as pd

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

df = pd.read_csv('datasets_4458_8204_winequality-red.csv')
df = df.drop('quality', axis=1)
print(model.predict(df.iloc[0].values.reshape(1,-1)))
