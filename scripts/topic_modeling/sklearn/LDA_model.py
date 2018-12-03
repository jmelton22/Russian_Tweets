#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib

# Read in file of tweets, drop rows with NaN
tweets = pd.read_csv('../../../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'])
tweets.dropna(subset=['lemmas'], inplace=True)
tweets.reset_index(drop=True, inplace=True)

tweet_docs = tweets.lemmas.tolist()  # Original tweets text as list
tweet_orig = tweets.text.tolist()  # Lemmatized tweets as list

# Load fitted count vectorizer
with open('../../../topic_modeling_objects/sklearn_vect.joblib', 'rb') as f:
    cv = joblib.load(f)

# Load term frequency matrix
with open('../../../topic_modeling_objects/sklearn_CV.joblib', 'rb') as f:
    tf = joblib.load(f)

# Load feature names
with open('../../../topic_modeling_objects/sklearn_feature_names.joblib', 'rb') as f:
    tf_names = joblib.load(f)

# Load fitted LDA model
with open('../../../topic_modeling_objects/sklearn_LDA_model.joblib', 'rb') as f:
    lda_model = joblib.load(f)

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

topic_names = ['Topic' + str(i) for i in range(len(lda_model.components_))]
doc_names = ['Doc' + str(i) for i in range(len(tweet_docs))]
word_names = ['Word ' + str(i) for i in range(len(tf_names))]

# Create df with the topic probabilities (cols) for each doc (rows)
doc_topic_df = pd.DataFrame(np.round(lda_W, 2), columns=topic_names, index=doc_names)

# Add column with dominant topic for each doc
doc_topic_df['dominant_topic'] = np.argmax(doc_topic_df.values, axis=1)
print(doc_topic_df.head(25))

# Create df with document topic distribution (num docs per topic)
topic_dist_df = doc_topic_df['dominant_topic'].value_counts().reset_index(name='Num docs')
topic_dist_df.columns = ['Topic_num', 'Num_docs']
topic_dist_df['Proportion'] = topic_dist_df.Num_docs.apply(lambda x: x / len(tweet_docs))
print()
print(topic_dist_df)


def top_keywords(feature_names, H):
    keywords = np.array(feature_names)
    topic_keywords = []
    for weights in H:
        topic_keywords_locs = (-weights).argsort()
        topic_keywords.append(keywords.take(topic_keywords_locs))

    return pd.DataFrame(topic_keywords)


# Create df with top words (cols) per topic (rows)
topic_words_df = top_keywords(tf_names, lda_H)
topic_words_df.columns = word_names
topic_words_df.index = topic_names
print(topic_words_df.iloc[:, :5])

doc_topic_df.to_csv('../../../results_csv/topics_per_doc_LDA.csv', index=True)
topic_dist_df.to_csv('../../../results_csv/docs_per_topic_LDA.csv', index=False)
topic_words_df.to_csv('../../../results_csv/words_per_topic_LDA.csv', index=True)
