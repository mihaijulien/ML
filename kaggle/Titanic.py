import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

pd_titanic = pd.read_csv("data/train.csv")
print(pd_titanic.head(10))

survived = pd_titanic[pd_titanic['Survived'] == 1]
no_survived = pd_titanic[pd_titanic['Survived'] == 0]

print(survived.head(5))
print(survived.head(5))

# Count the survived and deceased
print("Total =", len(pd_titanic))

print("Number of Survived passengers =", len(survived))
print("Percentage Survived =",  len(survived) / len(pd_titanic) * 100, "%")

print("Did not Survive =", len(no_survived))
print("Percentage who did not survive =", len(no_survived) / len(pd_titanic) * 100, "%")

# # Bar Chart to indicate the number of people survived based on their class
# # If you are a first class, you have a higher chance of survival
plt.figure()
sns.countplot(x = 'Pclass', data = pd_titanic)
plt.show()

plt.figure()
sns.countplot(x = 'Pclass', hue = 'Survived', data=pd_titanic)
plt.show()
#
# # Bar Chart to indicate the number of people survived based on their siblings status
# # If you have 1 siblings (SibSp = 1), you have a higher chance of survival compared to being alone (SibSp = 0)
plt.figure(figsize=[6,12])
plt.subplot(211)
sns.countplot(x = 'SibSp', data=pd_titanic)
plt.subplot(212)
sns.countplot(x = 'SibSp', hue = 'Survived', data=pd_titanic)
#
# # Bar Chart to indicate the number of people survived based on their Parch status (how many parents onboard)
# # If you have 1, 2, or 3 family members (Parch = 1,2), you have a higher chance of survival compared to being alone (Parch = 0)
plt.figure(figsize=[6,12])
plt.subplot(211)
sns.countplot(x = 'Parch', data=pd_titanic)
plt.subplot(212)
sns.countplot(x = 'Parch', hue = 'Survived', data=pd_titanic)
#
# # Bar Chart to indicate the number of people survived based on the port they emparked from
# # Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
# # If you embarked from port "C", you have a higher chance of survival compared to other ports!
plt.figure(figsize=[6,12])
plt.subplot(211)
sns.countplot(x = 'Embarked', data=pd_titanic)
plt.subplot(212)
sns.countplot(x = 'Embarked', hue = 'Survived', data=pd_titanic)
#
# # Bar Chart to indicate the number of people survived based on their sex
# # If you are a female, you have a higher chance of survival compared to other ports!
plt.figure(figsize=[6,12])
plt.subplot(211)
sns.countplot(x = 'Sex', data=pd_titanic)
plt.subplot(212)
sns.countplot(x = 'Sex', hue = 'Survived', data=pd_titanic)
#
# # Bar Chart to indicate the number of people survived based on their age
# # If you are a baby, you have a higher chance of survival
plt.figure(figsize=(40,30))
sns.countplot(x = 'Age', hue = 'Survived', data=pd_titanic)

# Age Histogram
pd_titanic['Age'].hist(bins = 40)
plt.show()

# Data cleaning
pd_titanic.drop(['Name', 'Ticket', 'Embarked', 'PassengerId', 'Cabin'],axis=1,inplace=True)
print(pd_titanic.head())


# There are some people with unknown ages. This function will fill the Age series with the average age where null based on gender
def Fill_Age(data):
    age = data[0]
    sex = data[1]

    if pd.isnull(age):
        if sex is 'male':
            return 29
        else:
            return 25
    else:
        return age

pd_titanic['Age'] = pd_titanic[['Age','Sex']].apply(Fill_Age,axis=1)
pd_titanic['Age'].hist(bins=40)
plt.show()

# Replace 'male' and 'female' with 0 and 1
pd.get_dummies(pd_titanic['Sex'])
# We just need one column
male = pd.get_dummies(pd_titanic['Sex'], drop_first = True)

# first let's drop the sex
pd_titanic.drop(['Sex'], axis=1, inplace=True)

# Now let's add the encoded column male (we replaced the sex column with male column)
pd_titanic = pd.concat([pd_titanic, male], axis=1)


# Create the model

X = pd_titanic.drop('Survived', axis = 1).values # all columns except the Survived one wich is our label
y = pd_titanic['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print(classification_report(y_test, prediction))