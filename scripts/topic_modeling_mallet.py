#!/usr/bin/env python3

import warnings
warnings.simplefilter('ignore')

import pandas as pd
from pprint import pprint

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import spacy

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel


def main():
    tweets = pd.read_csv('../tweets/tweets_clean.csv',
                         header=0,
                         parse_dates=['date'])
    print('Reading in tweets')
    tweets = list(tweets.clean_text.dropna())
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    print('Tokenizing and Removing Stop Words')
    tweets = [[word for word in word_tokenize(tweet)
               if word not in stop_words]
              for tweet in tweets]
    nlp = spacy.load('en', disable=['parser', 'ner'])

    def lemmatization(texts, allowed_postags=('NOUN', 'ADJ', 'VERB', 'ADV')):
        lemmas = []
        for i, tweet in enumerate(texts):
            if i % 10000 == 0 or i == len(tweets):
                print('Lemmatized {} tweets'.format(i))
            lemmas.append([token.lemma_ for token in nlp(' '.join(tweet)) if token.pos_ in allowed_postags])

        return lemmas

    tweet_lemmas = lemmatization(tweets)

    id2word = corpora.Dictionary(tweet_lemmas)
    corpus = []
    for i, tweet in enumerate(tweet_lemmas):
        if i % 10000 == 0:
            print('Converted {} docs to bag of words'.format(i))
        corpus.append(id2word.doc2bow(tweet))
    corpus = [id2word.doc2bow(tweet) for tweet in tweet_lemmas]

    # TODO: MALLET path not working

    mallet_path = 'C:\\mallet-2.0.8\\bin\\mallet'
    lda_mallet = gensim.models.wrappers.LdaMallet(mallet_path,
                                                  corpus=corpus,
                                                  num_topics=20,
                                                  id2word=id2word)
    pprint(lda_mallet.show_topics(formatted=False))
    print()

    coherence_model_ldamallet = CoherenceModel(model=lda_mallet,
                                               texts=tweet_lemmas,
                                               dictionary=id2word,
                                               coherence='c_v')
    coherence_mallet = coherence_model_ldamallet.get_coherence()
    print('Coherence Mallet:', coherence_mallet)


if __name__ == "__main__":
    main()
