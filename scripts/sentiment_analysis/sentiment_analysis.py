#!/usr/bin/env python3

import re
import pandas as pd
from textblob import TextBlob

# Read in file with tweets, drop rows with NaN
tweets = pd.read_csv('../../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'])
tweets.dropna(subset=['lemmas'], inplace=True)
tweets.reset_index(drop=True, inplace=True)


def clean_text(text):
    """
    Mini-preprocess function to remove links and split 'not' contractions

    :param text: Raw text
    :return: Processed text
    """
    # Remove URLs
    www_exp = r'www.[^ ]+'
    http_exp = r'http?s?[^\s]+'
    clean = re.sub('|'.join((www_exp, http_exp)), '', text)

    # Split 'not' contractions
    contraction_dict = {'can\'t': 'can not', 'won\'t': 'will not',
                        'isn\'t': 'is not', 'aren\'t': 'are not',
                        'wasn\'t': 'was not', 'weren\'t': 'were not',
                        'haven\'t': 'have not', 'hasn\'t': 'has not',
                        'wouldn\'t': 'would not', 'don\'t': 'do not',
                        'doesn\'t': 'does not', 'didn\'t': 'did not',
                        'couldn\'t': 'could not', 'shouldn\'t': 'should not',
                        'mightn\'t': 'might not', 'mustn\'t': 'must not',
                        'had\'t': 'had not'}
    contraction_exp = re.compile(r'\b(' + '|'.join(contraction_dict.keys()) + r')\b')
    clean = contraction_exp.sub(lambda x: contraction_dict[x.group()], clean)

    return clean


def sentiment(x):
    """
    Function to return the polarity and subjectivity of Textblob sentiment analysis

    :param x: Text string
    :return: Pandas Series with the polarity and subjectivity of text sentiment
    """
    blob = TextBlob(x).sentiment
    return pd.Series({'polarity': blob.polarity,
                      'subjectivity': blob.subjectivity})


print('Removing links from tweet text')
tweets['text'] = tweets.text.apply(clean_text)

print('Calculating sentiment polarity and subjectivity')
sentiment_df = tweets.text.apply(sentiment)

sentiment_df.to_csv('../../results_csv/tweet_sentiments.csv', index=False)
