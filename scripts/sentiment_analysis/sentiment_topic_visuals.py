#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

# Read in file with tweets, drop rows with NaN
tweets = pd.read_csv('../../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'])
tweets.dropna(subset=['lemmas'], inplace=True)
tweets.reset_index(drop=True, inplace=True)

# Read in file containing the topic probabilities for each tweet
topics = pd.read_csv('../../results_csv/topics_per_doc_LDA.csv', header=0)
topics.rename(columns={'dominant_topic': 'topic'}, inplace=True)

# Read in file containing the sentiment polarity/subjectivity for each tweet
sentiments = pd.read_csv('../../results_csv/tweet_sentiments.csv', header=0)
sentiments.set_index(tweets['date'], inplace=True)

# Create a categorical column that separates tweets into positive, neutral, or negative categories by polarity
sentiments['polarity_cat'] = np.where(sentiments['polarity'] > 0.1, 'Positive',
                                      np.where(sentiments['polarity'] < -0.1, 'Negative', 'Neutral'))

# Create new df with the dominant topic and sentiment for each tweet
# Set tweets' dates as df index
sentiments.reset_index(drop=True, inplace=True)  # Reset index for concatenation
topic_sentiment = pd.concat([topics['topic'], sentiments], axis=1)
topic_sentiment.set_index(tweets['date'], inplace=True)

# Create a categorical column that separates tweets into positive, neutral, or negative categories
topic_sentiment['polarity_cat'] = np.where(topic_sentiment['polarity'] > 0.1, 'Positive',
                                           np.where(topic_sentiment['polarity'] < -0.1, 'Negative', 'Neutral'))

# Group tweets by their topic and sentiment category
# Count the number of tweets and the mean and standard deviation of the sentiment polarity
topic_sentiment = topic_sentiment.groupby([topic_sentiment['topic'],
                                           topic_sentiment['polarity_cat']]).agg(['count', 'mean', 'std'])['polarity']

fig = topic_sentiment.unstack()['count'].plot(kind='bar',
                                              figsize=(16, 10),
                                              rot=0,
                                              title='Number of Tweets per Topic by Sentiment Polarity',
                                              fontsize=12)
fig.legend(title='Category', frameon=True, fontsize=11)
fig.set_xlabel('Topic', fontsize=12)
fig.set_ylabel('Number of tweets', fontsize=12)

fig = topic_sentiment.unstack()['mean'].plot(kind='barh',
                                             figsize=(16, 10),
                                             rot=0,
                                             title='Mean Polarity of Tweets by Topic')
fig.legend(title='Category', frameon=True)
fig.set_xlabel('Polarity')
fig.set_ylabel('Topic')

plt.show()
