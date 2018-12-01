#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

tweet_text = pd.read_csv('../../../tweets/tweets_clean.csv', header=0)

tweet_docs = tweet_text.lemmas.tolist()
tweet_orig = tweet_text.text.tolist()

tfidif_vect = TfidfVectorizer(max_df=0.95,
                              min_df=100,
                              use_idf=True)
tfidf = tfidif_vect.fit_transform(tweet_docs)
tfidf_names = tfidif_vect.get_feature_names()

nmf = NMF(n_components=15,
          solver='cd',
          # alpha=0.1,
          # l1_ratio=0.5,
          init='nndsvd',
          random_state=123)
nmf_model = nmf.fit(tfidf)

with open('../../../topic_modeling_objects/sklearn_NMF_model.joblib', 'wb') as f_out:
    joblib.dump(nmf_model, f_out)

# Topic to document matrix (W)
nmf_W = nmf_model.transform(tfidf)
# Word to topics matrix (H)
nmf_H = nmf.components_


def display_topics(H, W, feature_names, orig_docs, n_words=15, n_docs=20):
    for i, topic in enumerate(H):
        print('Topic {}: '.format(i) + ' '.join([feature_names[word]
                                                 for word in topic.argsort()[: (-n_words - 1): -1]]))
        print()
        top_doc_ids = np.argsort(W[:, i])[:: -1][0: n_docs]
        for doc_id in top_doc_ids:
            print('Tweet:', repr(orig_docs[doc_id]))
        print()


display_topics(nmf_H, nmf_W, tfidf_names, tweet_orig, n_words=25, n_docs=50)

# Create df with the topic probabilities (cols) for each doc (rows)
topic_names = ['Topic' + str(i) for i in range(len(nmf_model.components_))]
doc_names = ['Doc' + str(i) for i in range(len(tweet_docs))]

doc_topic_df = pd.DataFrame(np.round(nmf_W, 2), columns=topic_names, index=doc_names)

# Add column with dominant topic for each doc
doc_topic_df['dominant_topic'] = np.argmax(doc_topic_df.values, axis=1)
print(doc_topic_df.head(25))

# Create df with document topic distribution (num docs per topic)
topic_dist_df = doc_topic_df['dominant_topic'].value_counts().reset_index(name='Num docs')
topic_dist_df.columns = ['Topic_num', 'Num_docs']
print()
print(topic_dist_df)

doc_topic_df.to_csv('../../../results_csv/topics_per_doc_NMF.csv', index=False)
topic_dist_df.to_csv('../../../results_csv/docs_per_topic_NMF.csv', index=False)
