#!/usr/bin/env python3

import pandas as pd
import math
import matplotlib.colors as mc
import seaborn as sns
from bokeh.plotting import figure, output_file, output_notebook, show
from bokeh.models import HoverTool, ColumnDataSource, Range1d

sns.set(style='whitegrid')
tweets = pd.read_csv('../tweets/tweets.csv',
                     header=0,
                     parse_dates=['date'])

# Group by user key and year/month
monthly_tweets = tweets.groupby([tweets['user_key'], tweets.date.dt.year, tweets.date.dt.month]).size().to_frame('count')
monthly_tweets.index.rename(['user_key', 'year', 'month'], inplace=True)

# Unstack df so each column is tweets per month for a user_key
monthly_tweets = monthly_tweets.unstack(level=0, fill_value=0).sort_values(by=['year', 'month'], axis=0)
monthly_tweets.columns = monthly_tweets.columns.droplevel(level=0)
# Collapse index back into a single date
monthly_tweets.reset_index(inplace=True)
monthly_tweets['date'] = pd.to_datetime(dict(year=monthly_tweets.year,
                                             month=monthly_tweets.month,
                                             day=[1] * len(monthly_tweets)))
monthly_tweets.set_index(monthly_tweets['date'], inplace=True)
monthly_tweets.drop(['year', 'month', 'date'], axis=1, inplace=True)

# Totals per user
top_users = monthly_tweets.sum(axis=0).sort_values(ascending=False).to_frame('counts')

# Total tweets
monthly_tweets['Total'] = monthly_tweets.sum(axis=1)
