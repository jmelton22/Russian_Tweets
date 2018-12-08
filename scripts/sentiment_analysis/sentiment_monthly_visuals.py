#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TODO: Replace matplotlib plot with interactive bokeh plot
# TODO: Visualizations of sentiment per topic

# TODO: move legend outside of plot

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

# Group tweets by month and sentiment category and calculate the mean polarity score
monthly_sentiment = sentiments.groupby([sentiments.index.year, sentiments.index.month,
                                       sentiments['polarity_cat']]).mean().unstack(level=2)
monthly_sentiment.index.rename(['year', 'month'], inplace=True)

# Collapse index back into a single date
monthly_sentiment.reset_index(inplace=True)
monthly_sentiment.set_index(pd.to_datetime(dict(year=monthly_sentiment.year,
                                                month=monthly_sentiment.month,
                                                day=[1] * len(monthly_sentiment))))
monthly_sentiment.drop(['year', 'month'], axis=1, inplace=True)

# Plot the monthly mean polarity of tweets by sentiment category
fig1 = monthly_sentiment['polarity'].plot(figsize=(12, 8),
                                          use_index=True,
                                          style='.-',
                                          title='Mean Polarity Score of Monthly Tweet Sentiment')
fig1.legend(title='Category', loc='upper left')
plt.xlabel('Date')
plt.ylabel('Sentiment Polarity')

fig1.get_figure().savefig('../../visuals/monthly_mean_sentiment.png')

# Plot the daily mean subjectivity of tweets by sentiment category
fig2 = monthly_sentiment['subjectivity'].plot(figsize=(12, 8),
                                              use_index=True,
                                              style='.-',
                                              title='Mean Subjectivity Score of Monthly Tweet Sentiment')
fig2.legend(title='Category')
plt.xlabel('Date')
plt.ylabel('Sentiment Subjectivity')

fig2.get_figure().savefig('../../visuals/monthly_mean_subjectivity.png')

plt.show()
