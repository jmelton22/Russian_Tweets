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
Topic 0: December 14, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_December_14
Topic 1: May 05, 2015: https://en.wikipedia.org/wiki/Portal:Current_events/2015_May_5
Topic 2: February 11, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_February_11
    - North/South Korean relations
Topic 3: December 21, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_December_21
    - Berlin van terror attack
Topic 4: November 15, 2015: https://en.wikipedia.org/wiki/Portal:Current_events/2015_November_15
    - Paris terror attack
Topic 5: February 29. 2016:  https://en.wikipedia.org/wiki/Portal:Current_events/2016_February_29
    - Migrant crisis in Europe
    - Final batch of Clinton emails recovered from server
Topic 6: August 13, 2015: https://en.wikipedia.org/wiki/Portal:Current_events/2015_August_13
Topic 7: August 03, 2017: https://en.wikipedia.org/wiki/Portal:Current_events/2017_August_3
    - Trump transcripts with Mexican President Nieto and Australian PM Turnbull
    - Mueller impanels grand jury
Topic 8: February 20, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_February_20
    - Caucuses in Nevada and SC
        - Hillary beats Bernie in Nevada
        - Trump wins SC; Jeb Bush suspends campaign
Topic 9: July 21, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_July_21
    - Trump accepts GOP nomination
    - Russia ban in Rio Olympics
Topic 10: June 23, 2017: https://en.wikipedia.org/wiki/Portal:Current_events/2017_June_23
Topic 11: May 08, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_May_8
Topic 12: March 31, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_March_31
Topic 13: March 12, 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_March_12
    - Primaries
Topic 14: March 12 2016: https://en.wikipedia.org/wiki/Portal:Current_events/2016_March_12
    - Primaries
"""

plt.show()
