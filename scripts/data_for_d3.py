#!/usr/bin/env python3

import pandas as pd

tweets = pd.read_csv('../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'])
tweets.dropna(subset=['lemmas'], inplace=True)
tweets.reset_index(drop=True, inplace=True)

doc_topics = pd.read_csv('../results_csv/topics_per_doc_LDA.csv',
                         header=0)

tweets = pd.concat([tweets['date'], doc_topics['dominant_topic']], axis=1)

topics_per_month = tweets.groupby([tweets.date.dt.year, tweets.date.dt.month, tweets.dominant_topic]).size().to_frame('counts')
topics_per_month.index.rename(['year', 'month'], level=[0, 1], inplace=True)

# Get data in format for D3 interactive stream chart, write to csv
test_df = topics_per_month.reset_index()
test_df['date'] = pd.to_datetime(dict(year=test_df.year,
                                      month=test_df.month,
                                      day=[1] * len(test_df)))
test_df.drop(['year', 'month'], axis=1, inplace=True)
test_df.columns = ['key', 'value', 'date']
test_df.sort_values(by=['key', 'date'], inplace=True)
test_df.reset_index(drop=True, inplace=True)

test_df.to_csv('../results_csv/data.csv', index=False, date_format='%m/%d/%y')
