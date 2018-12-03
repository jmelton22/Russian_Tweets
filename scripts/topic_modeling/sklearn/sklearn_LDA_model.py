#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Read in file of tweets, drop rows with NaN
tweets = pd.read_csv('../../../tweets/tweets_clean.csv', header=0)
tweets.dropna(subset=['lemmas'], inplace=True)
tweets.reset_index(drop=True, inplace=True)

tweet_docs = tweets.lemmas.tolist()  # Lemmatized tweets as list
tweet_orig = tweets.text.tolist()  # Original tweets text as list

# Create term frequency matrix for tweet lemmas
cv = CountVectorizer(max_df=0.95,
                     min_df=100)
tf = cv.fit_transform(tweet_docs)
tf_names = cv.get_feature_names()  # English term names

# Fit a LDA model to the term frequency matrix (use parameters selected based on cross-validation results)
lda = LatentDirichletAllocation(n_components=15,
                                max_iter=10,
                                learning_method='online',
                                learning_decay=0.7,
                                random_state=123)
lda_model = lda.fit(tf)

print('Perplexity: {:.3f}'.format(lda_model.perplexity(tf)))
print('Log likelihood score: {:.3f}'.format(lda_model.score(tf)))
print()

# Save the fitted model to disk
with open('../../../topic_modeling_objects/sklearn_LDA_model.joblib', 'wb') as f_out:
    joblib.dump(lda_model, f_out)

# Topic to document matrix (W) for LDA model
lda_W = lda_model.transform(tf)
# Word to topics matrix (H) for LDA model
lda_H = lda_model.components_


def display_topics(H, W, feature_names, orig_docs, n_words=15, n_docs=25):
    """
    Function to print the top words and top tweets for each topic

    :param H: Topic to document matrix
    :param W: Word to topics matrix
    :param feature_names: English term names
    :param orig_docs: Original tweet texts
    :param n_words: Number of top words to print
    :param n_docs: Number of top tweets to print
    """
    for i, topic in enumerate(H):
        print('Topic {}: '.format(i) + ' '.join([feature_names[word]
                                                 for word in topic.argsort()[: (-n_words - 1): -1]]))
        print()
        top_doc_ids = np.argsort(W[:, i])[:: -1][0: n_docs]
        for doc_id in top_doc_ids:
            print('Tweet:', repr(orig_docs[doc_id]))
        print('-' * 80)


display_topics(lda_H, lda_W, tf_names, tweet_orig, n_words=20)
