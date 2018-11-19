#!/usr/bin/env python3

import pandas as pd
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# with open('./topic_modeling_objects/corpus.pkl', 'rb') as f:
#     corpus = pickle.load(f)
#
# with open('./topic_modeling_objects/dictionary.pkl', 'rb') as f:
#     vocab = pickle.load(f)
#
# with open('./topic_modeling_objects/lemmas.pkl', 'rb') as f:
#     tweet_lemmas = pickle.load(f)
#     tweet_docs = [' '.join(doc) for doc in tweet_lemmas if doc]

tweets = pd.read_csv('../tweets/tweets_clean.csv')
tweets = list(tweets.clean_text.dropna().unique())

# TODO: Print original tweet text

stop_words = set(stopwords.words('english'))
tweets_no_stop = [[word for word in word_tokenize(tweet)
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


tweet_lemmas = lemmatization(tweets_no_stop)
tweet_docs = [' '.join(doc) for doc in tweet_lemmas]

# tweet_docs = list(pd.Series(tweet_docs).unique())

print(len(tweet_docs))

tfidif_vect = TfidfVectorizer(max_df=0.95,
                              min_df=2,
                              stop_words='english')
tfidf = tfidif_vect.fit_transform(tweet_docs)
tfidf_names = tfidif_vect.get_feature_names()

cv = CountVectorizer(max_df=0.95,
                     min_df=2,
                     stop_words='english')
tf = cv.fit_transform(tweet_docs)
tf_names = cv.get_feature_names()

lda = LatentDirichletAllocation(n_components=20,
                                max_iter=5,
                                learning_method='online',
                                learning_decay=0.7,
                                random_state=0)
lda_model = lda.fit(tf)

# Topic to document matrix (W) for LDA model
lda_W = lda_model.transform(tf)
# Word to topics matrix (H) for LDA model
lda_H = lda_model.components_

nmf = NMF(n_components=20,
          alpha=0.1,
          l1_ratio=0.5,
          random_state=1,
          init='nndsvd')
nmf_model = nmf.fit(tfidf)

# Topic to document matrix (W) for NMF model
nmf_W = nmf_model.transform(tfidf)
# Word to topics matrix (H) for NMF model
nmf_H = nmf.components_


def display_topics(H, W, feature_names, docs, orig_docs, n_words=15, n_docs=20):
    for i, topic in enumerate(H):
        print('Topic {}: '.format(i) + ' '.join([feature_names[word] for word in topic.argsort()[: (-n_words - 1): -1]]))
        # print(' '.join([feature_names[word] for word in topic.argsort()[: (-n_words - 1): -1]]))
        print()
        top_doc_ids = np.argsort(W[:, i])[:: -1][0: n_docs]
        for doc_id in top_doc_ids:
            print('Tweet:', orig_docs[doc_id])
            print('Lemmas:', docs[doc_id])
        print()


display_topics(lda_H, lda_W, tf_names, tweet_docs, tweets)
print('. . .')
display_topics(nmf_H, nmf_W, tfidf_names, tweet_docs, tweets)
