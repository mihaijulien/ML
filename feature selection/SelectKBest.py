import pandas as pd
import numpy as np
import os 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

file = os.path.join(os.getcwd() , "datasets_228_482_diabetes.csv")
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
df = pd.read_csv(file, names=names)

print(df.head())

array = df.values
X = array[:,0:8]
y = array[:,8]

# Feature extraction
# SelectKBest will remove all but the K highest scoring features
test = SelectKBest(score_func=f_classif, k=4)
fit=test.fit(X,y)
features = fit.transform(X)

print(features)

# Looking at how all the features are correlated to the target variable
correlation_matrix = df.corr()
print(correlation_matrix['class'].sort_values(ascending=False))