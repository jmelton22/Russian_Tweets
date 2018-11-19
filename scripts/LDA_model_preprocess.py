#!/usr/bin/env python3

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    import imp
warnings.simplefilter('ignore')

import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import spacy
import gensim
import gensim.corpora as corpora

tweets = pd.read_csv('../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'])
print('Reading in tweets')
# Drop tweets reduced to NaN by preprocessing
tweets = list(tweets.clean_text.dropna().unique())

# Tokenize tweets and remove stop words
stop_words = set(stopwords.words('english'))
print('Tokenizing and removing stop words')
tweets = [[word for word in word_tokenize(tweet)
           if word not in stop_words]
          for tweet in tweets]

nlp = spacy.load('en', disable=['parser', 'ner'])


def lemmatization(texts, allowed_postags=('NOUN', 'ADJ', 'VERB', 'ADV')):
    lemmas = []
    for i, tweet in enumerate(texts):
        if i % 10000 == 0 or i == len(tweets)-1:
            print('Lemmatized {} tweets'.format(i))
        lemmas.append([token.lemma_ for token in nlp(' '.join(tweet)) if token.pos_ in allowed_postags])

    return lemmas


tweet_lemmas = lemmatization(tweets)

# Save tweet lemmas to pickle file
with open('./topic_modeling_objects/lemmas.pkl', 'wb') as f_out:
    pickle.dump(tweet_lemmas, f_out)

print('Making Dictionary')
id2word = corpora.Dictionary(tweet_lemmas)
# Save Dictionary to pickle file
id2word.save('./topic_modeling_objects/dictionary.pkl')

print('Converting docs to bags of words corpus')
corpus = [id2word.doc2bow(tweet) for tweet in tweet_lemmas]

# Save corpus to pickle file
with open('./topic_modeling_objects/corpus.pkl', 'wb') as f_out:
    pickle.dump(corpus, f_out)
