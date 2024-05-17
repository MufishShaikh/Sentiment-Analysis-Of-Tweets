#!/usr/bin/python
# Fetch the tweets and cleaning using imported preprocessing file and strored in result.csv file

import tweepy
import csv  # Import csv
import preprocessing
from . import model


def get_twitter_data():
    # api_key = 'hyc3fX2hVLlReSufR6zfzoLzj'
    api_key = 'ZGxEd0J3YUI2NG5wQzlGTDZQZU86MTpjaQ'
    # api_secret = 'teyEyGFFGSYdwNwKrTmsImEKvs1NdIZpDj7YOUGRZB5tFp39Dp'
    api_secret = 'jZrVqPH1pCsohV1Yxa4GlHdqEIVcMcRuV7AFf6WBTCOzq5x5_f'
    access_token_secret = 'Amu84izsqwpla9jtvXkVRJE1mx8FMRFEUoUseNcy50ak3'
    
    access_token = '1292462289426059264-WDXvxLy9qKMHYCxtdcnEBetUHWiWtk'
    # auth = tweepy.auth.OAuthHandler(api_key, api_secret)
    # auth.set_access_token(access_token, access_token_secret)

    # api = tweepy.API(auth)

    # Open/create a file to append data to
    # csvFile = open('result.csv', 'w')
    counter = 0
    c = 5
    query = ["mood", "Happy", "bad"]
    # Use csv writer
    # csvWriter = csv.writer(csvFile)
    data = []
    final_result = []
    # for tweet in tweepy.Cursor(api.search,
    #                            q=query,
    #                            lang="en",
    #                            count=c).items():
    tweets = ["I am good", "I am happy","Good Morning","I Love You","I am feeling happy today"]
    for tweet in tweets:
        # Write a row to the CSV file. I use encode UTF-8
        # text = tweet.text
        data.append(tweet)
        text = preprocessing.preprocess(text)
        # text = preprocessing(tweet.text)
        print("Text ::", text)
        result = model.Get_Data(text)
        print(result)
        final_result.append(result)
        # csvWriter.writerow([text])
        # print(tweet.text)
        counter = counter + 1
        if counter == c:
            break
    # csvFile.close()
    return final_result
