#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tweets = pd.read_csv('../../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'])
tweets.dropna(subset=['lemmas'], inplace=True)
tweets.reset_index(drop=True, inplace=True)

topics = pd.read_csv('../../results_csv/topics_per_doc_LDA.csv',
                     header=0)
topics.rename(columns={'dominant_topic': 'topic'}, inplace=True)

sentiments = pd.read_csv('../../results_csv/tweet_sentiments.csv',
                         header=0)

topic_sentiment = pd.concat([topics['topic'], sentiments], axis=1)
topic_sentiment.set_index(tweets['date'], inplace=True)

topic_sentiment['polarity_cat'] = np.where(topic_sentiment['polarity'] > 0.1, 'pos',
                                           np.where(topic_sentiment['polarity'] < -0.1, 'neg', 'neut'))

topic_sentiment = topic_sentiment.groupby([topic_sentiment['topic'],
                                           topic_sentiment['polarity_cat']]).agg(['count', 'mean', 'std'])['polarity']

sentiments.set_index(tweets['date'], inplace=True)
sentiments['polarity_cat'] = np.where(sentiments['polarity'] > 0, 'pos',
                                      np.where(sentiments['polarity'] < 0, 'neg', 'neut'))

daily_sentiment = sentiments.groupby([sentiments.index.date, sentiments['polarity_cat']]).mean()['polarity'].unstack(level=1)

fig = daily_sentiment.plot()

fig.get_figure().savefig('../../visuals/daily_mean_sentiment.png')
plt.show()

# topic_sentiment.boxplot()
# plt.show()

# topic_sentiment = topic_sentiment.unstack()
#
# # print(topic_sentiment.head(20))
#
# topic_polarity = topic_sentiment.groupby('topic')['polarity'].agg([('negative', lambda x: x[x < 0].count()),
#                                                                     ('neutral', lambda x: x[x == 0].count()),
#                                                                     ('positive', lambda x: x[x > 0].count())])
# topic_polarity.loc['total'] = topic_polarity.sum()
# print(topic_polarity)
