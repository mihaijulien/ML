import pandas as pd
import numpy as np
from sklearn import linear_model


fuel_consumption_co2 = pd.read_csv("dataset/FuelConsumptionCo2.csv")
feature_names = ['ENGINESIZE', 'CYLINDERS', 'CO2EMISSIONS']
X = fuel_consumption_co2[feature_names]

# train/test split
msk = np.random.rand(len(fuel_consumption_co2)) < 0.8
train = X[msk]
test = X[~msk]

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

print('Coefficients: ', regr.coef_)

#test

y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS']])
test_x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

print("Residual sum of squared: ", np.mean(y_hat - test_y) ** 2)
print("Variance score: ", regr.score(test_x, test_y))