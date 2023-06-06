#!/usr/bin/env python
# coding: utf-8

# Import libraries and data frames 

# In[1]:


import string
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cProfile
import seaborn as sns

from scipy.sparse import coo_matrix # this is the sparse matrix format discussed in lecture

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# In[2]:


general = pd.read_csv("../Lesson 7/general-tweets.csv")
keyword = pd.read_csv("../Lesson 7/keyword-tweets.csv")


# In[3]:


general


# In[4]:


keyword


# Concatinate the General and Keyword data frames + updating coumn names.

# In[5]:


df = pd.concat([general,keyword])
newColumnNames = ['Sentiment', 'Tweet']
df.columns = newColumnNames


# In[6]:


df


# Replace Sentiment labels for numeric values.

# In[7]:


replacement_dict = {'POLIT': 1, 'NOT': 0}
df['Sentiment'] = df['Sentiment'].replace(replacement_dict)


# In[8]:


df


# Clean Tweets

# In[9]:


from nltk.stem import WordNetLemmatizer

def preprocess(text, list_of_steps):
    
    for step in list_of_steps:
        if step == 'remove_non_ascii':
            text = ''.join([x for x in text if ord(x) < 128])
        elif step == 'removehttps':
            text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)
        elif step == 'remove@':   
            text = re.sub(r'\@w+|\#','',text)
        elif step == 'lowercase':
            text = text.lower()
        elif step == 'remove_punctuation':
            punct_exclude = set(string.punctuation)
            text = ''.join(char for char in text if char not in punct_exclude)
        elif step == 'remove_numbers':
            text = re.sub("\d+", "", text)
        elif step == 'strip_whitespace':
            text = ' '.join(text.split())
        elif step == 'stem_words':
            lmtzr = WordNetLemmatizer()
            word_list = text.split(' ')
            stemmed_words = [lmtzr.lemmatize(word) for word in word_list]
            text = ' '.join(stemmed_words)
    return text

step_list = ['remove_non_ascii', 'lowercase', 'remove_punctuation', 'remove_numbers',
            'strip_whitespace', 'remove_stopwords', 'stem_words']


# In[10]:


import cProfile
#tweet_df['clean_tweet'] = tweet_df['tweet_text'].map(lambda s: preprocess(s, steps))
cProfile.run("df['clean_tweet'] = df['Tweet'].map(lambda s: preprocess(s, step_list))");


# In[11]:


df.head()


# TD-IDF Vector

# In[12]:


# vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, max_features = 50, stop_words = 'english')
# clean_texts = df['clean_tweet']
# tf_idf_tweets = vectorizer.fit_transform(clean_texts)

# tf_idf_tweets


# In[13]:


vectorizer = TfidfVectorizer(max_features=50)
X = vectorizer.fit_transform(df['clean_tweet'])
y = df['Sentiment']


# In[14]:


# from sklearn.model_selection import train_test_split
# y_targets = np.array([y[0] for y in tf_idf_tweets])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# X_train, X_test, y_train, y_test = train_test_split(tf_idf_tweets, y_targets, test_size = 0.25, random_state=42)

print(type(X_train), X_train.shape)


# In[15]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[16]:


train_results = lr.predict(X_train)
test_results = lr.predict(X_test)

train_acc = np.mean(y_train == train_results)
test_acc = np.mean(y_test == test_results)

print('Train accuracy: {}'.format(train_acc))
print('Test accuracy: {}'.format(test_acc))
print('Baseline accuracy: {}'.format(np.max([np.mean(y_test == 1), np.mean(y_test == 0)])))


# In[19]:


from sklearn.metrics import accuracy_score
max_features_values = [5, 500, 5000, 50000]

for max_features in max_features_values:
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df['clean_tweet'])
    y = df['Sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    baseline_accuracy = max(y.mean(), 1 - y.mean())

    print("Max Features:", max_features)
    print("Training Accuracy:", train_accuracy)
    print("Testing Accuracy:", test_accuracy)
    print("Baseline Accuracy:", baseline_accuracy)
    print()


# In[20]:


from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
precision, recall, f1, support = precision_recall_fscore_support(y_test, test_results)
tn, fp, fn, tp = confusion_matrix(y_test, test_results).ravel()

print(confusion_matrix(y_test, test_results))
print('='*35)
print('             Class 1   -   Class 0')
print('Precision: {}'.format(precision))
print('Recall   : {}'.format(recall))
print('F1       : {}'.format(f1))
print('Support  : {}'.format(support))


# In[ ]:




