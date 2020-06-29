# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 09:09:33 2020

@author: lukeb
"""
import pandas as pd

df = pd.read_csv('../datasets_4458_8204_winequality-red.csv')
df = df.drop('quality', axis=1)
input = list(df.iloc[0,:])