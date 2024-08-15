import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re

def remove_words_with_numbers(text):
    filtered_words = [word for word in text.split() if not re.search(r'\d', word)]
    return ' '.join(filtered_words)


df = pd.read_csv('food_data.csv')

# print(df.head)
# print(df.info())
# print(df['brand_name'].value_counts())

df['brand_name'] = df['brand_name'].str.lower()
df['brand_name'] = df['brand_name'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
# print(df['brand_name'])
df['brand_name'] = df['brand_name'].apply(remove_words_with_numbers)
df.dropna(inplace=True)
# print(df['brand_name'])

df['tokens'] = df['brand_name'].apply(lambda x: x.split())
X_train, X_test, y_train, y_test = train_test_split(df['brand_name'], df['generic_name'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

df.to_csv('preprocessed_data.csv', index=False)
