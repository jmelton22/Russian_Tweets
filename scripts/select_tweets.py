#!/usr/bin/env python3

import pandas as pd

# Read in tweets, drop rows with NaN
tweets = pd.read_csv('../tweets/tweets_clean.csv', header=0,
                     parse_dates=['date'])
tweets.dropna(subset=['lemmas'], inplace=True)
tweets.reset_index(drop=True, inplace=True)

# Read in topic probabilities per doc
topics = pd.read_csv('../results_csv/topics_per_doc_LDA.csv', header=0, index_col=0)
topics.reset_index(drop=True, inplace=True)

# Add row with probability for dominant topic
topics['prob'] = topics.drop('dominant_topic', axis=1).max(axis=1)

# Concat tweets with dominant_topic and probability columns
tweets = pd.concat([tweets, topics[['dominant_topic', 'prob']]], axis=1)
tweets.set_index('date', inplace=True)


def select_tweets(day, topic):
    day_tweets = tweets.loc[day]
    print('{}: {}'.format(day, topic))

    subset = day_tweets[day_tweets['dominant_topic'] == topic][['text', 'prob']].sort_values('prob', ascending=False)
    for _, row in subset.iterrows():
        print(row['prob'], repr(row['text']))


# Top dates based on proportion of tweets per topic
dates_list = [('2016-10-04', 0), ('2016-05-12', 1), ('2016-05-08', 2), ('2015-12-16', 3), ('2016-03-12', 4),
              ('2016-12-14', 5), ('2016-02-29', 6), ('2016-02-11', 7), ('2017-08-03', 8), ('2016-07-21', 9),
              ('2016-10-03', 10), ('2016-09-22', 11), ('2017-06-23', 12), ('2015-06-13', 13), ('2016-03-22', 14)]

# Top dates based on count of tweets per topic
# dates_list = [('2016-10-06', 0), ('2016-11-08', 1), ('2016-12-21', 2), ('2016-10-06', 3), ('2016-09-18', 4),
#               ('2016-12-14', 5), ('2016-02-29', 6), ('2016-10-06', 7), ('2016-10-19', 8), ('2016-07-21', 9),
#               ('2016-10-06', 10), ('2016-12-26', 11), ('2017-01-25', 12), ('2016-11-14', 13), ('2016-03-22', 14)]

for d, top in dates_list:
    select_tweets(d, top)
    print('-' * 80)
