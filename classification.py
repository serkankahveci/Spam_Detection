import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('all')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pandas as pd

data = pd.read_excel('/content/spam7.xlsx')

def preprocess_text(text):

    tokens = word_tokenize(text.lower())

    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in set(stopwords.words('english'))]
    return " ".join(tokens)

data['text'] = data['text'].astype(str)
data['text'] = data['text'].apply(preprocess_text)

sid = SentimentIntensityAnalyzer()
data['compound'] = data['text'].apply(lambda x: sid.polarity_scores(x)['compound'])

sarcasm_threshold = 0.1
data['label'] = np.where(data['compound'] < sarcasm_threshold, 1, 0)

X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

spam_classifier = MultinomialNB()
spam_classifier.fit(X_train_tfidf, y_train)

y_pred = spam_classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(report)
