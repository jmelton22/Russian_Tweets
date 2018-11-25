#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

tweet_text = pd.read_csv('../tweets/tweets_text.csv', header=0)
tweet_text.dropna(subset=['lemmas'], inplace=True)

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
with open('./topic_modeling_objects/sklearn_LDA_model.joblib', 'wb') as f_out:
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

doc_topic_df.to_csv('./topic_modeling_objects/topics_per_doc_LDA.csv', index=True)
topic_dist_df.to_csv('./topic_modeling_objects/docs_per_topic_LDA.csv', index=False)
topic_words_df.to_csv('./topic_modeling_objects/words_per_topic_LDA.csv', index=True)
