#!/usr/bin/env python3

import pandas as pd
import numpy as np

tweets = pd.read_csv('../../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'])
tweets.dropna(subset=['lemmas'], inplace=True)
tweets.reset_index(drop=True, inplace=True)

topics = pd.read_csv('../../results_csv/topics_per_doc_LDA.csv',
                     header=0)

sentiments = pd.read_csv('../../results_csv/tweet_sentiments.csv',
                         header=0)

topic_sentiment = pd.concat([topics['dominant_topic'], sentiments], axis=1)
topic_sentiment.set_index(tweets['date'], inplace=True)

topic_sentiment['group'] = np.where(topic_sentiment['polarity'] > 0, 'pos',
                                    np.where(topic_sentiment['polarity'] < 0, 'neg', 'neut'))

print(topic_sentiment.head(20))

topic_polarity = topic_sentiment.groupby('dominant_topic')['polarity'].agg([('negative', lambda x: x[x < 0].count()),
                                                                            ('neutral', lambda x: x[x == 0].count()),
                                                                            ('positive', lambda x: x[x > 0].count())])
topic_polarity.loc['total'] = topic_polarity.sum()

print(topic_polarity)
