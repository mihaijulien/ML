import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# create dataset
fizzbuzz = {}

for i in range(1, 1000):
	if(i%3 == 0 and i%5 == 0):
		fizzbuzz[i] = 'fizzbuzz'
	elif(i%3 == 0):
		fizzbuzz[i] = 'fizz'
	elif(i%5 == 0):
		fizzbuzz[i] = 'buzz'
	else:
		fizzbuzz[i] = i

data = {'Number': list(fizzbuzz.keys()), 'FizzBuzz': list(fizzbuzz.values())}
df = pd.DataFrame(data)


df['FizzBuzz'].replace(['fizz', 'buzz', 'fizzbuzz'], [3,5,35], inplace=True)
print(df.head())

X = df['Number'].values
y = df['FizzBuzz'].values
X = StandardScaler().fit_transform(X.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

print(X_train.shape)
print(y_train.shape)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print('Train set accuracy: ', accuracy_score(y_train, model.predict(X_train)))
print('Test set accuracy: ', accuracy_score(y_test, pred))
#print("MSE: ", np.mean(((y_test - prediction) ** 2)))
