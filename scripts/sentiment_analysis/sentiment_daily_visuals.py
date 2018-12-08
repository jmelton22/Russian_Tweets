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
topics = pd.read_csv('../../results_csv/topics_per_doc_LDA.csv',
                     header=0)
topics.rename(columns={'dominant_topic': 'topic'}, inplace=True)

# Read in file containing the sentiment polarity/subjectivity for each tweet
sentiments = pd.read_csv('../../results_csv/tweet_sentiments.csv',
                         header=0)
sentiments.set_index(tweets['date'], inplace=True)

# Create a categorical column that separates tweets into positive, neutral, or negative categories by polarity
sentiments['polarity_cat'] = np.where(sentiments['polarity'] > 0.1, 'Positive',
                                      np.where(sentiments['polarity'] < -0.1, 'Negative', 'Neutral'))

# Group tweets by date and sentiment category and calculate the mean polarity score
daily_sentiment = sentiments.groupby([sentiments.index.date,
                                      sentiments['polarity_cat']]).mean().unstack(level=1)

# Plot the daily mean polarity of tweets by sentiment category
fig1 = daily_sentiment['polarity'].plot(figsize=(12, 8),
                                        use_index=True,
                                        style='.-',
                                        title='Mean Polarity Score of Daily Tweet Sentiment')
fig1.legend(title='Category')
plt.xlabel('Date')
plt.ylabel('Sentiment Polarity')

fig1.get_figure().savefig('../../visuals/daily_mean_sentiment.png')

# Plot the daily mean subjectivity of tweets by sentiment category
fig2 = daily_sentiment['subjectivity'].plot(figsize=(12, 8),
                                            use_index=True,
                                            style='.-',
                                            title='Mean Subjectivity Score of Daily Tweet Sentiment')
fig2.legend(title='Category')
plt.xlabel('Date')
plt.ylabel('Sentiment Subjectivity')

fig2.get_figure().savefig('../../visuals/daily_mean_subjectivity.png')
plt.show()
