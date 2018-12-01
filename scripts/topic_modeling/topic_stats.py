#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tweets = pd.read_csv('../../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'],
                     index_col='date')
tweets.dropna(subset=['lemmas'], inplace=True)
# tweets = tweets.loc['2016-07-01': '2017-03-31']
tweets.reset_index(inplace=True)

doc_topics = pd.read_csv('../../results_csv/topics_per_doc_LDA.csv',
                         header=0)

tweets = pd.concat([tweets['date'], doc_topics['dominant_topic']], axis=1)

topics_daily = tweets.groupby([tweets.date.dt.date, tweets.dominant_topic]).size().to_frame().unstack(level=1,
                                                                                                      fill_value=0)
topics_daily.columns = topics_daily.columns.droplevel(0)
topics_daily.columns = topics_daily.columns.values.astype('str')

topics_daily['sum'] = topics_daily.sum(axis=1)
topics_daily = topics_daily[topics_daily['sum'] >= 100]
topics_daily.drop('sum', axis=1, inplace=True)

topics_daily_props = topics_daily.apply(lambda x: x / x.sum(), axis=1)

# topics_daily.loc['mean'] = topics_daily.mean()
# topics_daily.loc['var'] = topics_daily.var()

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(12, 6))
topics_daily_props.boxplot()
plt.xlabel('Topic')
plt.ylabel('Proportion of tweets')
plt.title('Boxplot of Topics for Daily Tweets')

fig.savefig('../../visuals/tweets_prop_boxplot.png')

print(topics_daily_props.idxmax())
"""
Topic 0: 2016-10-04: https://en.wikipedia.org/wiki/Portal:Current_events/2016_October_4
Topic 1: 2016-05-12: https://en.wikipedia.org/wiki/Portal:Current_events/2016_May_12
Topic 2: 2016-05-08: https://en.wikipedia.org/wiki/Portal:Current_events/2016_May_8
Topic 3: 2015-12-16: https://en.wikipedia.org/wiki/Portal:Current_events/2015_December_16
Topic 4: 2016-03-12: https://en.wikipedia.org/wiki/Portal:Current_events/2016_March_12
Topic 5: 2016-12-14: https://en.wikipedia.org/wiki/Portal:Current_events/2016_December_14
Topic 6: 2016-02-29: https://en.wikipedia.org/wiki/Portal:Current_events/2016_February_29
Topic 7: 2016-02-11: https://en.wikipedia.org/wiki/Portal:Current_events/2016_February_11
Topic 8: 2017-08-03: https://en.wikipedia.org/wiki/Portal:Current_events/2017_August_3
Topic 9: 2016-07-21: https://en.wikipedia.org/wiki/Portal:Current_events/2016_July_21
Topic 10: 2016-10-03: https://en.wikipedia.org/wiki/Portal:Current_events/2016_October_3
Topic 11: 2016-09-22: https://en.wikipedia.org/wiki/Portal:Current_events/2016_September_22
Topic 12: 2017-06-23: https://en.wikipedia.org/wiki/Portal:Current_events/2017_June_23
Topic 13: 2015-06-13: https://en.wikipedia.org/wiki/Portal:Current_events/2015_June_13
Topic 14: 2016-03-22: https://en.wikipedia.org/wiki/Portal:Current_events/2016_March_22
"""

fig = plt.figure(figsize=(12, 6))
topics_daily.boxplot()
plt.xlabel('Topic')
plt.ylabel('Number of tweets')
plt.title('Boxplot of Topics for Daily Tweets')

fig.savefig('../../visuals/tweets_boxplot.png')

print(topics_daily.idxmax())
"""
Topic 0: 2016-10-06: https://en.wikipedia.org/wiki/Portal:Current_events/2016_October_6
Topic 1: 2016-11-08: https://en.wikipedia.org/wiki/Portal:Current_events/2016_November_8
    - US Election
Topic 2: 2016-12-21: https://en.wikipedia.org/wiki/Portal:Current_events/2016_December_21
Topic 3: 2016-10-06: https://en.wikipedia.org/wiki/Portal:Current_events/2016_October_6
Topic 4: 2016-09-18: https://en.wikipedia.org/wiki/Portal:Current_events/2016_September_18
Topic 5: 2016-12-14: https://en.wikipedia.org/wiki/Portal:Current_events/2016_December_14
Topic 6: 2016-02-29: https://en.wikipedia.org/wiki/Portal:Current_events/2016_February_29
Topic 7: 2016-10-06: https://en.wikipedia.org/wiki/Portal:Current_events/2016_October_6
Topic 8: 2016-10-19: https://en.wikipedia.org/wiki/Portal:Current_events/2016_October_19
Topic 9: 2016-07-21: https://en.wikipedia.org/wiki/Portal:Current_events/2016_July_21
Topic 10: 2016-10-06: https://en.wikipedia.org/wiki/Portal:Current_events/2016_October_6
Topic 11: 2016-12-26: https://en.wikipedia.org/wiki/Portal:Current_events/2016_December_26
Topic 12: 2017-01-25: https://en.wikipedia.org/wiki/Portal:Current_events/2017_January_25
Topic 13: 2016-11-14: https://en.wikipedia.org/wiki/Portal:Current_events/2016_November_14
Topic 14: 2016-03-22: https://en.wikipedia.org/wiki/Portal:Current_events/2016_March_22
"""

tweet_counts = tweets.groupby([tweets.date.dt.date]).sum(axis=0)
tweet_counts.columns = ['counts']
tweet_counts = tweet_counts[tweet_counts['counts'] > 0]

tweet_counts['log_counts'] = tweet_counts.counts.apply(lambda x: np.log(x))


def logarithm(x):
    if x > 0:
        return np.log(x)
    else:
        return np.NaN


topics_daily = topics_daily.applymap(lambda x: logarithm(x))

fig = topics_daily.plot.kde(figsize=(12, 8))
plt.xlabel('Log of Number of Tweets')
plt.ylabel('Proportion of Days')
plt.title('KDE Plot for Log of Daily Tweets per Topic')

fig.get_figure().savefig('../../visuals/topics_kde.png')

plt.show()
