#!/usr/bin/env python3

import pandas as pd

tweets = pd.read_csv('../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'])

tweets = tweets[tweets.date.isin(pd.date_range('2016-01-01', '2017-05-31'))]
daily_tweets = tweets.groupby(tweets.date.dt.date).size().to_frame('counts')
