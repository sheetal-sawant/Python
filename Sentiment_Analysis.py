#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #function required to split data for training and testing data sets

import nltk #natural language toolkit - library to work with human language data
from nltk.corpus import stopwords #to remove stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output


# In[48]:


tweet_data = pd.read_csv('C:/Users/Sheetal/Sentiment.csv') #import dataset
tweet_data = tweet_data[['text','sentiment']] #keeping required columns only
tweet_data.head()


# In[25]:


train, test = train_test_split(tweet_data,test_size = 0.1) #split dataset into train and test data
train = train[train.sentiment != "Neutral"] #exclude neutral sentiments


# In[37]:


positive_train = train[ train.sentiment == 'Positive']
positive_train = positive_train.text
negative_train = train[ train.sentiment == 'Negative']
negative_train = negative_train.text

def wordcloud_design(tweet_data, color):
    words = ' '.join(tweet_data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("Positive words")
wordcloud_design(positive_train,'white')
print("Negative words")
wordcloud_design(negative_train,'black')


# In[44]:


tweets = []
stopwords_list = set(stopwords.words("english"))

for index, row in train.iterrows():
    filtered_words = [each.lower() for each in row.text.split() if len(each) >= 3]
    clean_words = [word for word in filtered_words
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    words_without_stopwords = [word for word in clean_words if not word in stopwords_list]
    tweets.append((words_without_stopwords, row.sentiment))

positive_test = test[ test.sentiment == 'Positive']
positive_test = positive_test.text
negative_test = test[ test.sentiment == 'Negative']
negative_test = negative_test.text


# In[49]:


def keep_words_from_tweets(tweets):
    all = []
    for (words, sentiment) in tweets:
        all.extend(words)
    return all

def word_features(wordlist): 
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features
w_features = word_features(keep_words_from_tweets(tweets))

def extract_word_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


# In[50]:


wordcloud_design(w_features,'black')


# In[53]:


training_set = nltk.classify.apply_features(extract_word_features,tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)


# In[54]:


negative_count = 0
positive_count = 0
for obj in negative_test: 
    result =  classifier.classify(extract_features(obj.split()))
    if(result == 'Negative'): 
        negative_count = negative_count + 1
for obj in positive_test: 
    result =  classifier.classify(extract_features(obj.split()))
    if(result == 'Positive'): 
        positive_count = positive_count + 1
        
print('[Negative]: %s/%s '  % (len(negative_test),negative_count))        
print('[Positive]: %s/%s '  % (len(positive_test),positive_count))   


# In[ ]:




