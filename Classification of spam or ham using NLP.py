# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 00:00:32 2019

@author: madaa
"""
import pandas as pd

import re
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer



#Reading the data

dataset= pd.read_csv("C:\\Users\\madaa\\Downloads\\Ex_Files_NLP_Python_ML_EssT\\Exercise Files\\Ch02\\02_04\Start\\SMSSpamCollection.tsv", sep="\t",header=None)
print(dataset.head())

dataset.columns= ['label', 'body_text']
print(dataset.head())  #Displays first five rows of the dataset

#Exploring the dataset

#1 what is shape of data
print("Input data has {} rows and {} columns".format(len(dataset),len(dataset.columns)))


#2 How many spam/ham are there?
print("Out of {} rows {} are spam and {} ham".format(len(dataset),len(dataset[dataset['label']=="spam"]),len(dataset[dataset['label']=="ham"])))


#3 How many missing data is there?
print("Number of nulls in label: {}".format(dataset['label'].isnull().sum()))
print("Number of nulls in label: {}".format(dataset['body_text'].isnull().sum()))

#Using regular expressions to split the sentences of body text into list of words
#tokenize data
pd.set_option('display.max_colwidth',100)
print(dataset.head())

#Remove punctuation
def remove_punct(text):
    text_nonpunct= "".join([char for char in text if char not in string.punctuation])
    return text_nonpunct

dataset['body_text_clean']=dataset['body_text'].apply(lambda x: remove_punct(x))
print(dataset.head())


#Tokenization of text

def tokenize(text):
        tokens=re.split('\W+',text)
        return tokens

dataset['body_text_tokenize']=dataset['body_text_clean'].apply(lambda x: tokenize(x.lower()))
print(dataset.head())

#remove stopwords
stopwords=nltk.corpus.stopwords.words('english')

def remove_stopwords(tokenised_list):
    text=[word for word in tokenised_list if word not in stopwords]
    return text

dataset['body_text_nostop']=dataset['body_text_tokenize'].apply(lambda x: remove_stopwords(x))
print(dataset.head())


#Stemming and Lemmitization

ps=nltk.PorterStemmer()
def stemming(tokenized_list):
    text=[ps.stem(word) for word in tokenized_list]
    return text
dataset['body_text_stemmed']=dataset['body_text_nostop'].apply(lambda x: stemming(x))
print(dataset.head())

#Lemmatizing
wn=nltk.WordNetLemmatizer()
def lemmatizing(tokenized_text):
    text=[wn.lemmatize(word) for word in tokenized_text]
    return text
dataset['body_text_lemmatize']=dataset['body_text_nostop'].apply(lambda x: lemmatizing (x))

print(dataset.head(10))

def clean_text(text):
    text="".join([word.lower() for word in text if word not in string.punctuation])
    tokens=re.split("\W+",text)
    text="".join([ps.stem(word) for word in tokens if word not in stopwords])
    return text
dataset['cleaned_text']=dataset['body_text'].apply(lambda x: clean_text(x))
print(dataset.head())

#Vectorization of data

count_vect=CountVectorizer(analyzer=clean_text)
X_count=count_vect.fit_transform(dataset['body_text'])
print(X_count.shape)
print(count_vect.get_feature_names())

print("-------------------------------------------------------------------")

data_sample=dataset[0:20]
count_vect_sample=CountVectorizer(analyzer=clean_text)
X_count_sample=count_vect_sample.fit_transform(data_sample['body_text'])
print(X_count_sample.shape)
print(count_vect_sample.get_feature_names())


#in case of sparse matrix data should be stored in an array format

#X_counts_df=pd.DataFrame(X_count_sample.toarray())
#print(X_counts_df)

#X_counts_df.columns=count_vect_sample.get_feature_names()
#print(X_counts_df)


#N-grams
n_gram_vect=CountVectorizer(ngram_range=(2,2))
X_counts=n_gram_vect.fit_transform(dataset['cleaned_text'])
print(X_counts.shape)
print(X_counts.get_feature_names())
