#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

tweet_text = pd.read_csv('../../../tweets/tweets_clean.csv', header=0)

tweet_docs = tweet_text.lemmas.tolist()
tweet_orig = tweet_text.text.tolist()

cv = CountVectorizer(max_df=0.95,
                     min_df=100)
tf = cv.fit_transform(tweet_docs)
tf_names = cv.get_feature_names()

lda = LatentDirichletAllocation(n_components=15,
                                max_iter=10,
                                learning_method='online',
                                learning_decay=0.7,
                                random_state=123)
lda_model = lda.fit(tf)

print('Perplexity: {:.3f}'.format(lda_model.perplexity(tf)))
print('Log likelihood: {:.3f}'.format(lda_model.score(tf)))
print()

# Save fitted model
with open('../../../topic_modeling_objects/sklearn_LDA_model.joblib', 'wb') as f_out:
    joblib.dump(lda_model, f_out)

# Topic to document matrix (W) for LDA model
lda_W = lda_model.transform(tf)
# Word to topics matrix (H) for LDA model
lda_H = lda_model.components_


def display_topics(H, W, feature_names, orig_docs, n_words=15, n_docs=25):
    for i, topic in enumerate(H):
        print('Topic {}: '.format(i) + ' '.join([feature_names[word]
                                                 for word in topic.argsort()[: (-n_words - 1): -1]]))
        print()
        top_doc_ids = np.argsort(W[:, i])[:: -1][0: n_docs]
        for doc_id in top_doc_ids:
            print('Tweet:', repr(orig_docs[doc_id]))
        print('-' * 80)


display_topics(lda_H, lda_W, tf_names, tweet_orig, n_words=20)
