import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

path = os.getcwd() + '\\data\\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

print(data.head(10))

print("Number of missing data in data frame: ", data.isnull().sum().sum())

# matplotlib
plt.scatter(data['Population'], data['Profit'])
plt.xlabel('Population')
plt.ylabel('Profit')

#pandas plot
#data.plot(kind='scatter', x='Population', y='Profit')
#plt.show()

print(data.describe())

X = data['Population'].values
y = data['Profit'].values

X = X.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Mean absolute error: ", np.mean(np.absolute(prediction - y_test)))
print("Mean squared error: ", np.mean(np.absolute(prediction - y_test)) ** 2)
print("R2-score: ", r2_score(prediction, y_test))
