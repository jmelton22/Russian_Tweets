#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

tweets = pd.read_csv('../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'],
                     index_col='date')
tweets.dropna(subset=['lemmas'], inplace=True)
# tweets = tweets.loc['2016-07-01': '2017-03-31']
tweets.reset_index(inplace=True)

doc_topics = pd.read_csv('./topic_modeling_objects/topics_per_doc_LDA.csv',
                         header=0)

tweets = pd.concat([tweets['date'], doc_topics['dominant_topic']], axis=1)

topics_daily = tweets.groupby([tweets.date.dt.date, tweets.dominant_topic]).size().to_frame().unstack(level=1,
                                                                                                      fill_value=0)
topics_daily.columns = topics_daily.columns.droplevel(0)
topics_daily.columns = topics_daily.columns.values.astype('str')

topics_daily['sum'] = topics_daily.sum(axis=1)
topics_daily = topics_daily[topics_daily['sum'] >= 100]
topics_daily.drop('sum', axis=1, inplace=True)

topics_daily = topics_daily.apply(lambda x: x / x.sum(), axis=1)

# topics_daily.loc['mean'] = topics_daily.mean()
# topics_daily.loc['var'] = topics_daily.var()

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(12, 6))
topics_daily.boxplot()
plt.xlabel('Topic')
plt.ylabel('Proportion of tweets')
plt.title('Boxplot of Topics for Daily Tweets')

fig.savefig('../visuals/tweets_boxplot.png')

print(topics_daily.idxmax())
"""
Topic 0: November 02, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_November_2
Topic 1: February 29, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_February_29
Topic 2: July 21, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_July_21
Topic 3: November 23, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_November_23
Topic 4: September 26, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_September_26
Topic 5: October 05, 2016:  https://en.wikipedia.org/wiki/Portal:Current_events/2016_October_5
Topic 6: December 19, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_December_19
Topic 7: December 16, 2015: https://en.wikipedia.org/wiki/Portal:Current_events/2015_December_16
Topic 8: October 17, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_October_17
Topic 9: November 08, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_November_8
Topic 10: August 03, 2017: https://en.wikipedia.org/wiki/Portal:Current_events/2017_August_3
Topic 11: June 23, 2017: https://en.wikipedia.org/wiki/Portal:Current_events/2017_June_23
Topic 12: September 01, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_September_1
Topic 13: March 22, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_March_22
Topic 14: June 20 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_June_20
"""

plt.show()
