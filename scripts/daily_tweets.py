#!/usr/bin/env python3

import pandas as pd
import matplotlib.colors as mc
import seaborn as sns
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource

tweets = pd.read_csv('../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'],
                     index_col='date')
tweets = tweets.loc['2016-07-01': '2017-03-31']
daily_tweets = tweets.groupby(tweets.index.date).size().to_frame('counts')


def daily_counts(tweets_df):
    # Create color palette for tweet values
    palette = sns.color_palette('YlOrRd', len(tweets_df))
    # Assign a color to each value of tweets by rank
    tweets_df['color'] = [mc.rgb2hex(palette[int(rank - 1)]) for rank in tweets_df.counts.rank()]

    src = ColumnDataSource(data=dict(date=tweets_df.index.values,
                                     counts=tweets_df.counts,
                                     color=tweets_df.color))

    output_file('../visuals/daily_tweets.html')
    p = figure(width=1000, height=600,
               x_axis_type='datetime',
               x_axis_label='Date',
               y_axis_label='Number of tweets',
               title='Tweets per day')

    p.line(x='date',
           y='counts',
           line_width=4,
           color='lightgray',
           source=src)
    p.circle(x='date',
             y='counts',
             fill_color='color',
             size=12,
             source=src)

    hover = HoverTool(tooltips=[('Date', '@date{%m-%d-%Y}'),
                                ('Num tweets', '@counts')],
                      formatters={'date': 'datetime'},
                      mode='mouse')
    p.add_tools(hover)
    p.xgrid.grid_line_color = None

    return p


daily_plot = daily_counts(daily_tweets)
show(daily_plot)
