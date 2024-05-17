# python3 -m pip install -r requirements.txt# -*- coding: utf-8 -*-
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log, sqrt
import pandas as pd
import numpy as np
import re
import random
import tweepy
import csv  # Import csv
import preprocessing
# from . import model

"""# Loading the Data"""

tweets = pd.read_csv('templates/Data/sentiment_tweets3.csv', encoding='latin-1')

"""# Splitting the Data in Training and Testing Sets

As you can see, I used almost all the data for training: 98% and the rest for testing.
"""

totalTweets = 8000 + 2314
trainIndex, testIndex = list(), list()
for i in range(tweets.shape[0]):
    if np.random.uniform(0, 1) < 0.98:
        trainIndex += [i]
    else:
        testIndex += [i]
trainData = tweets.iloc[trainIndex]
testData = tweets.iloc[testIndex]

"""#Pre-processing the data for the training: Tokenization, stemming, and removal of stop words"""


def process_message(message, lower_case=True, stem=True, stop_words=True, gram=2):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words


class TweetClassifier(object):
    def __init__(self, trainData, method='tf-idf'):
        self.tweets, self.labels = trainData['message'], trainData['label']
        self.method = method

    def train(self):
        self.calc_TF_and_IDF()
        if self.method == 'tf-idf':
            self.calc_TF_IDF()
        else:
            self.calc_prob()

    def calc_prob(self):
        self.prob_depressive = dict()
        self.prob_positive = dict()
        for word in self.tf_depressive:
            self.prob_depressive[word] = (self.tf_depressive[word] + 1) / (self.depressive_words + \
                                                                           len(list(self.tf_depressive.keys())))
        for word in self.tf_positive:
            self.prob_positive[word] = (self.tf_positive[word] + 1) / (self.positive_words + \
                                                                       len(list(self.tf_positive.keys())))
        self.prob_depressive_tweet, self.prob_positive_tweet = self.depressive_tweets / self.total_tweets, self.positive_tweets / self.total_tweets

    def calc_TF_and_IDF(self):
        noOfMessages = self.tweets.shape[0]
        self.depressive_tweets, self.positive_tweets = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_tweets = self.depressive_tweets + self.positive_tweets
        self.depressive_words = 0
        self.positive_words = 0
        self.tf_depressive = dict()
        self.tf_positive = dict()
        self.idf_depressive = dict()
        self.idf_positive = dict()
        for i in range(noOfMessages):
            message_processed = process_message(self.tweets.iloc[i])
            count = list()  # To keep track of whether the word has ocured in the message or not.
            # For IDF
            for word in message_processed:
                if self.labels.iloc[i]:
                    self.tf_depressive[word] = self.tf_depressive.get(word, 0) + 1
                    self.depressive_words += 1
                else:
                    self.tf_positive[word] = self.tf_positive.get(word, 0) + 1
                    self.positive_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels.iloc[i]:
                    self.idf_depressive[word] = self.idf_depressive.get(word, 0) + 1
                else:
                    self.idf_positive[word] = self.idf_positive.get(word, 0) + 1

    def calc_TF_IDF(self):
        self.prob_depressive = dict()
        self.prob_positive = dict()
        self.sum_tf_idf_depressive = 0
        self.sum_tf_idf_positive = 0
        for word in self.tf_depressive:
            self.prob_depressive[word] = (self.tf_depressive[word]) * log(
                (self.depressive_tweets + self.positive_tweets) \
                / (self.idf_depressive[word] + self.idf_positive.get(word, 0)))
            self.sum_tf_idf_depressive += self.prob_depressive[word]
        for word in self.tf_depressive:
            self.prob_depressive[word] = (self.prob_depressive[word] + 1) / (
                    self.sum_tf_idf_depressive + len(list(self.prob_depressive.keys())))

        for word in self.tf_positive:
            self.prob_positive[word] = (self.tf_positive[word]) * log((self.depressive_tweets + self.positive_tweets) \
                                                                      / (self.idf_depressive.get(word, 0) +
                                                                         self.idf_positive[word]))
            self.sum_tf_idf_positive += self.prob_positive[word]
        for word in self.tf_positive:
            self.prob_positive[word] = (self.prob_positive[word] + 1) / (
                    self.sum_tf_idf_positive + len(list(self.prob_positive.keys())))

        self.prob_depressive_tweet, self.prob_positive_tweet = self.depressive_tweets / self.total_tweets, self.positive_tweets / self.total_tweets

    def classify(self, processed_message):
        pDepressive, pPositive = 0, 0
        for word in processed_message:
            if word in self.prob_depressive:
                pDepressive += log(self.prob_depressive[word])
            else:
                if self.method == 'tf-idf':
                    pDepressive -= log(self.sum_tf_idf_depressive + len(list(self.prob_depressive.keys())))
                else:
                    pDepressive -= log(self.depressive_words + len(list(self.prob_depressive.keys())))
            if word in self.prob_positive:
                pPositive += log(self.prob_positive[word])
            else:
                if self.method == 'tf-idf':
                    pPositive -= log(self.sum_tf_idf_positive + len(list(self.prob_positive.keys())))
                else:
                    pPositive -= log(self.positive_words + len(list(self.prob_positive.keys())))
            pDepressive += log(self.prob_depressive_tweet)
            pPositive += log(self.prob_positive_tweet)
        return pDepressive >= pPositive

    def predict(self, testData):
        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = process_message(message)
            result[i] = int(self.classify(processed_message))
        return result


def metrics(labels, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels.iloc[i] == 1 and predictions[i] == 1)
        true_neg += int(labels.iloc[i] == 0 and predictions[i] == 0)
        false_pos += int(labels.iloc[i] == 0 and predictions[i] == 1)
        false_neg += int(labels.iloc[i] == 1 and predictions[i] == 0)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("F-score: ", Fscore)
    # print("Accuracy: ", accuracy)


sc_tf_idf = TweetClassifier(trainData, 'tf-idf')
sc_tf_idf.train()
preds_tf_idf = sc_tf_idf.predict(testData['message'])
metrics(testData['label'], preds_tf_idf)

# sc_bow = TweetClassifier(trainData, 'bow')
# sc_bow.train()
# preds_bow = sc_bow.predict(testData['message'])
# metrics(testData['label'], preds_bow)

'''Getting tweets or User Text'''

"""# Predictions with TF-IDF

"""

def Manual_Data_prediction(data):
    pm = process_message(data)
    # print(sc_tf_idf.classify(pm))
    return sc_tf_idf.classify(pm)


def Twitter_Data_Prediction():
    api_key = 'pSBDpA4liaEtizwzgXgkFhkAd'
    api_secret = 'r60Wzx8B84Swgz6qQ7tZrAE4DTPi9ouIBXpXCqmJOqWFWLprnP'
    access_token_secret = 'glhn4tQXeZr1WZ7a5qb7n1iT2RCInBvbCq4W3uW8hGLln'
    access_token = '1293177979699224577-t98wgDiyhd7ZqWOW5Ikv14L0cd17W7'

    # auth = tweepy.auth.OAuthHandler(api_key, api_secret)
    # auth.set_access_token(access_token, access_token_secret)

    # api = tweepy.API(auth)

    # Open/create a file to append data to
    # csvFile = open('result.csv', 'w')
    counter = 0
    c = 50

    # choice_list = [['mood','happy',"depressed",'sad'],['awsome','beautiful','cry','depraved'],['celebration','peaceable','peaceful','envious']]
    # query = random.choice(choice_list)
    query = ['mood','happy','depressed','sad']
    print("Query is ::\n",query)
    # query = ['mood','happy',"depressed",'sad']
    # Use csv writer
    # csvWriter = csv.writer(csvFile)
    data = []
    final_result = []
    print("Fetch data from twitter:\n")
    # for tweet in tweepy.Cursor(api.search_tweets,
    #                            q=query,
    #                            lang="en",
    #                            count=c).items():
    
    # Minimun 3 tweets required in below list
    tweets = ["Got PPO guys",
              "Once i Started Waking up early it's over for u guys ",
              "feeling very bad for hardik pandya seeing the crowd booming just because he is captain of MI" ,
              "IPL is IPLing", 
              "@klopstock Agreed. My computer is my servant, _not_ the other way around! ",
              "Good Morning",
              "I Love You",
              "I am feeling happy today",
              "Happy 'main mar jaungi fir pata chalega' day",
              ""
              ]
    for tweet in tweets:
        # Write a row to the CSV file. I use encode UTF-8
        # text = tweet.text
        data.append(tweet)
        # text = preprocessing.preprocess(text)
        # text = preprocessing(text)
        result = sc_tf_idf.classify(process_message(tweet))
        print(tweet)
        final_result.append(result)
        # csvWriter.writerow([text])
        # print(tweet.text)
        counter = counter + 1
        if counter == c:
            break
    # csvFile.close()
    return data, final_result
