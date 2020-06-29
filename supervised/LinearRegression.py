import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

fuel_consumption_co2 = pd.read_csv("dataset/FuelConsumptionCo2.csv")

# create the list of features below
feature_names = ['ENGINESIZE', 'CO2EMISSIONS']

# select data corresponding to features in feature_names
X = fuel_consumption_co2[feature_names]

# print description of the statistics from X
print(X.describe())
# print the top few lines
print(X.head())

# split the dataset into train and test sets, 80% for training, 20% for test
msk = np.random.rand(len(fuel_consumption_co2)) < 0.8
train = X[msk]
test = X[~msk]

# use linear regression
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
#train the model
regr.fit(train_x, train_y)

print("Slope of the line: ", regr.coef_)
print("Bias: ", regr.intercept_)

# plot outputs
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, c = 'blue')
plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emissions")
plt.show()

# evaluation
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: ", np.mean(np.absolute(test_y_hat - test_y)))
print("Mean squared error: ", np.mean(np.absolute(test_y_hat - test_y)) ** 2)
print("R2-score: ", r2_score(test_y_hat, test_y))