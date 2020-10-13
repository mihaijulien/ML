import pandas as pd
import numpy as np
import os 
from sklearn.ensemble import ExtraTreesClassifier

file = os.path.join(os.getcwd() , "datasets_228_482_diabetes.csv")
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
df = pd.read_csv(file, names=names)

print(df.head())

array = df.values
X = array[:,0:8]
y = array[:,8]


model = ExtraTreesClassifier(n_estimators=10)
model.fit(X,y)

print(model.feature_importances_)