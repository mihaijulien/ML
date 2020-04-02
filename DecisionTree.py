from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

plt.figure(figsize=(12,12))
model = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
tree.plot_tree(model.fit(X,y))
plt.show()

prediction = model.predict(X_test)
score = accuracy_score(y_test, prediction)
print('Test set accuracy score: ', score)