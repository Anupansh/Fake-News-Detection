import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('news.csv')
label = df.label
x_train, x_test, y_train, y_test = train_test_split(df['text'], label, test_size=0.15, random_state=5)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)
print(df.shape)
print(x_train.shape)
print(x_test.shape)
print(tfidf_train[0].shape)
pac = PassiveAggressiveClassifier(max_iter=5000)
pac.fit(tfidf_train, y_train)
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score * 100, 2)}%')

#DataFlair - Build confusion matrix
conf = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
conf_2 = confusion_matrix(y_test,y_pred, labels=['REAL','FAKE'])
print(conf_2)
print("Confusion Matrix", conf)
