#!/usr/bin/env python3

import pandas as pd
import matplotlib.colors as mc
import seaborn as sns
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource

tweets = pd.read_csv('../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'])

# Group into number of tweets per month
monthly_tweets = tweets.groupby([tweets.date.dt.year, tweets.date.dt.month]).size().to_frame('counts')
monthly_tweets.index.rename(['year', 'month'], inplace=True)

# Collapse index back into a single date
monthly_tweets.reset_index(inplace=True)
monthly_tweets['date'] = pd.to_datetime(dict(year=monthly_tweets.year,
                                             month=monthly_tweets.month,
                                             day=[1] * len(monthly_tweets)))
monthly_tweets.set_index(monthly_tweets['date'], inplace=True)
monthly_tweets.drop(['year', 'month', 'date'], axis=1, inplace=True)


def monthly_counts(tweets_df):
    # Create color palette for tweet values
    palette = sns.color_palette('YlOrRd', len(tweets_df))
    # Assign a color to each value of tweets by rank
    tweets_df['color'] = [mc.rgb2hex(palette[int(rank - 1)]) for rank in tweets_df.counts.rank()]

    src = ColumnDataSource(data=dict(date=tweets_df.index.values,
                                     counts=tweets_df.counts,
                                     color=tweets_df.color))

    output_file('../visuals/monthly_tweets.html')
    p = figure(width=1000, height=600,
               x_axis_type='datetime',
               x_axis_label='Date',
               y_axis_label='Number of tweets',
               title='Tweets per month')

    p.line(x='date',
           y='counts',
           line_width=4,
           color='lightgray',
           source=src)
    p.circle(x='date',
             y='counts',
             fill_color='color',
             size=15,
             source=src)

    hover = HoverTool(tooltips=[('Date', '@date{%B-%Y}'),
                                ('Num tweets', '@counts')],
                      formatters={'date': 'datetime'},
                      mode='vline')
    p.add_tools(hover)
    p.xgrid.grid_line_color = None

    return p


monthly_plot = monthly_counts(monthly_tweets)
show(monthly_plot)
