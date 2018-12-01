#!/usr/bin/env python3

import pandas as pd
import re
import csv
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

tweets = pd.read_csv('../tweets/tweets.csv', header=0)
print(tweets.isna().sum())
print()

tweets.drop(['user_id', 'created_at', 'retweet_count', 'retweeted',
             'favorite_count', 'source', 'expanded_urls', 'posted',
             'retweeted_status_id', 'in_reply_to_status_id'],
            axis=1, inplace=True)

# Remove rows with NaN text field
tweets.dropna(subset=['text'], inplace=True)
tweets.reset_index(drop=True, inplace=True)
tweets.rename(columns={'created_str': 'date'}, inplace=True)

tweets.info()
print()
print('Unique account keys:', len(tweets.user_key.unique()))
print()
print(tweets.groupby(['user_key']).size().sort_values(ascending=False).head(10))
print()


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

    # Remove @ mentions and hashtags
    # at_exp = r'@[A-Za-z0-9_]+'
    # hashtag_exp = r'#[A-Za-z0-9_]+'
    # clean = re.sub('|'.join((at_exp, hashtag_exp)), '', clean)

    # Remove non-letter chars, 'RT'/'MT' from retweets, enclitics from split contractions
    clean = re.sub('[^a-zA-Z0-9]', ' ', clean)
    tails = [r'\bRT\b', r'\bMT\b', r'\bve\b', r'\bre\b', r'\bll\b']
    clean = re.sub('|'.join(tails), '', clean)
    # Convert to lower case
    clean = clean.lower()

    return ' '.join([word for word in clean.split(' ') if word is not ''])


def find_hashtags(text):
    return re.findall('#(\w+)', text)


# Find hashtags in tweets
print('Finding hashtags in tweets')
tweets['hashtags'] = tweets['text'].apply(lambda text: re.findall('#(\w+)', text))

# Apply preprocessing to tweet text
print('Cleaning tweet text')
tweets['clean_text'] = tweets['text'].apply(clean_text)
tweets.dropna(subset=['clean_text'], inplace=True)

tweet_text = list(tweets.clean_text)

# Tokenize tweets and remove stop words
stop_words = set(stopwords.words('english'))
my_stops = ['go', 'be', 'also', 'get', 'do', 'thing', 'use', 'let', 'would', 'say', 'could', 'yet']
stop_words.update(my_stops)
print('Tokenizing and removing stop words')
tweet_text = [[word for word in word_tokenize(tweet) if word not in stop_words and len(word) > 2]
              for tweet in tweet_text]

nlp = spacy.load('en', disable=['parser', 'ner'])


def lemmatization(texts, allowed_postags=('NOUN', 'ADJ', 'VERB', 'ADV')):
    lemmas = []
    for i, tweet in enumerate(texts):
        if i % 10000 == 0 or i == len(tweets)-1:
            print('Lemmatized {} tweets'.format(i))
        lemmas.append([token.lemma_ for token in nlp(' '.join(tweet))
                       if token.pos_ in allowed_postags and token.lemma_ not in stop_words])

    return lemmas


tweet_lemmas = lemmatization(tweet_text)
tweets['lemmas'] = [' '.join(words) for words in tweet_lemmas]
tweets.dropna(subset=['lemmas'], inplace=True)
tweets.reset_index(drop=True, inplace=True)

tweets.info()

tweets.to_csv('../tweets/tweets_clean.csv', index=False, quoting=csv.QUOTE_ALL)
