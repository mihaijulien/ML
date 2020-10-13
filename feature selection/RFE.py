import pandas as pd
import numpy as np
import os 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

file = os.path.join(os.getcwd() , "datasets_228_482_diabetes.csv")
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
df = pd.read_csv(file, names=names)

print(df.head())

array = df.values
X = array[:,0:8]
y = array[:,8]


# RFE - recursevly removes attributes by performing a greedy search to find the best performing
# feature subset. In an itterative fashion it creates models, from your feature set and determines 
# the best or wrose performing features at each iteration and constructs the next iterative model 
# with the features that are left until it finally converges to the best set of features
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=3)
fit = rfe.fit(X,y)

print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
