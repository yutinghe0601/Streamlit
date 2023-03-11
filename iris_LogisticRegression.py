# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 21:19:48 2023

@author: 96546
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
# Model Report Card
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_iris


iris = load_iris()

# df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
# df.head()
# df['target'] = iris.target

X = iris.data
y = iris.target


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1, test_size=0.30)


print(f"Shape of Xtrain: {Xtrain.shape}\n\
Shape of Xtest: {Xtest.shape}\n\
Shape of ytrain: {ytrain.shape}\n\
Shape of ytest: {ytest.shape}")


model = LogisticRegression()
model.fit(Xtrain, ytrain)

y_model = model.predict(Xtest)
y_model


# Create a new DataFrame to store the results
classification_results = pd.DataFrame(columns=['classifier_Name', 'Accuracy_Score'])

accuracy_score(ytest, y_model)
classification_results = classification_results.append({'classifier_Name': 'Logistic Regression', 'Accuracy_Score': accuracy_score(ytest, y_model)}, ignore_index=True)
classification_results


print(classification_report(ytest, y_model))


# save model in a pickle format .pkl
import pickle
pickle_out = open("C:/Users/96546/Documents/何宇婷/学习/master/DSSI/LRmodel.pkl","wb")
pickle.dump(model, pickle_out)
pickle_out.close()









