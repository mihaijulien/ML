import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# create dataset
fizzbuzz = {}

for i in range(101, 1000):
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
print(df.head())

def fizzbuzz_encode(i):
	if   (i == 'fizzbuzz') : return np.array([0,0,0,1])
	elif (i == 'fizz')     : return np.array([0,0,1,0])
	elif (i == 'buzz')     : return np.array([0,1,0,0])
	else                 : return np.array([1,0,0,0])

X = df['Number'].values
y = np.array([fizzbuzz_encode(i) for i in df['FizzBuzz'].values])
X = StandardScaler().fit_transform(X.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

print(X_train.shape)
print(y_train.shape)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print('Train set accuracy: ', accuracy_score(y_train, model.predict(X_train)))
print('Test set accuracy: ', accuracy_score(y_test, pred))

# test model with new data
def predict_number(n):
	pred = model.predict(np.array([n]).reshape(1,-1))
	
	if (pred[0][0] == 1.0) : 
		print('fizzbuzz')
		return 'fizzbuzz'
	elif (pred[0][1] == 1.0) : 
		return 'fizz'
	elif (pred[0][2] == 1.0) :
		return 'buzz'
	else: 
		return n

print('1', predict_number(1))
print('3', predict_number(3))
print('5', predict_number(5))
print('15', predict_number(15))
