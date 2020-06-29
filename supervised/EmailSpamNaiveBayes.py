import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

spam_df = pd.read_csv("dataset/emails.csv")
print(spam_df.head())

spam = spam_df[spam_df['spam'] == 1]
print(spam)
ham = spam_df[spam_df['spam]' == 0]]
print(ham)

print( 'Spam percentage =', (len(spam) / len(spam_df) )*100,"%")
print( 'Ham percentage =', (len(ham) / len(spam_df) )*100,"%")

vectorizer = CountVectorizer()
spamham_countervectorizer = vectorizer.fit_transform(spam_df['text'])
print(vectorizer.get_feature_names())

label = spam_df['spam'].values
X = spamham_countvectorizer
y = label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)

# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict_test))