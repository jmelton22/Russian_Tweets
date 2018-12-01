#!/usr/bin/env python3

import re
import pandas as pd
from textblob import TextBlob

tweets = pd.read_csv('../../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'])
tweets.dropna(subset=['lemmas'], inplace=True)
tweets.reset_index(drop=True, inplace=True)


def clean_text(text):
    from bs4 import BeautifulSoup

    # Remove URLs
    www_exp = r'www.[^ ]+'
    http_exp = r'http?s?[^\s]+'
    clean = re.sub('|'.join((www_exp, http_exp)), '', text)

    # Remove HTML encoded text (ampersand)
    soup = BeautifulSoup(clean, 'lxml')
    clean = soup.get_text()
    try:
        clean = clean.encode().decode().replace(u'\ufffd', '?')
    except UnicodeEncodeError or UnicodeDecodeError:
        clean = clean

    return clean


def sentiment(x):
    blob = TextBlob(x).sentiment
    return pd.Series({'polarity': blob.polarity,
                      'subjectivity': blob.subjectivity})


print('Removing links from tweet text')
tweets['text'] = tweets.text.apply(clean_text)

print('Calculating sentiment polarity and subjectivity')
sentiment_df = tweets.text.apply(sentiment)

sentiment_df.to_csv('../../results_csv/tweet_sentiments.csv', index=False)
