#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

tweet_text = pd.read_csv('../tweets/tweets_text.csv', header=0)
tweet_text.dropna(subset=['lemmas'], inplace=True)

tweet_docs = tweet_text.lemmas.tolist()
tweet_orig = tweet_text.text.tolist()

tfidif_vect = TfidfVectorizer(max_df=0.95,
                              min_df=2,
                              stop_words='english')
tfidf = tfidif_vect.fit_transform(tweet_docs)
tfidf_names = tfidif_vect.get_feature_names()

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
        print('Topic {}: '.format(i) + ' '.join([feature_names[word]
                                                 for word in topic.argsort()[: (-n_words - 1): -1]]))
        print()
        top_doc_ids = np.argsort(W[:, i])[:: -1][0: n_docs]
        for doc_id in top_doc_ids:
            print('Tweet:', orig_docs[doc_id])
            print('Lemmas:', docs[doc_id])
        print()


display_topics(nmf_H, nmf_W, tfidf_names, tweet_docs, tweet_orig)
