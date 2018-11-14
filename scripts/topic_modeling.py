#!/usr/bin/env python3

import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import spacy

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

tweets = pd.read_csv('../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'])
tweets = list(tweets.clean_text.dropna())[:100]

# Remove stop words
stop_words = set(stopwords.words('english'))
tweets = [[word for word in word_tokenize(tweet)
           if word not in stop_words]
          for tweet in tweets]

bigram = gensim.models.Phrases(tweets,
                               min_count=5,
                               threshold=100)

trigram = gensim.models.Phrases(bigram[tweets], threshold=100)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

tweet_bigrams = [bigram_mod[tweet] for tweet in tweets]
tweet_trigrams = [trigram_mod[bigram_mod[tweet]] for tweet in tweets]

nlp = spacy.load('en', disable=['parser', 'ner'])


def lemmatization(texts, allowed_postags=('NOUN', 'ADJ', 'VERB', 'ADV')):
        return [[token.lemma_ for token in nlp(' '.join(tweet))
                 if token.pos_ in allowed_postags]
                for tweet in texts]


tweet_lemmas = lemmatization(tweets)
print(tweet_lemmas[:1])

id2word = corpora.Dictionary(tweet_lemmas)
corpus = [id2word.doc2bow(tweet) for tweet in tweet_lemmas]

print(corpus[:1])

read = [[(id2word[i], freq) for i, freq in cp] for cp in corpus[:1]]
for each in read:
    print(each)
