#!/usr/bin/env python3

import pandas as pd
import math
import matplotlib.colors as mc
import seaborn as sns
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource, Range1d

tweets = pd.read_csv('../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'])

# Group into user keys
user_tweets = tweets.groupby('user_key').size().sort_values(ascending=False).to_frame('counts')


def user_counts(tweets_df, num_users=15):
    users_to_plot = tweets_df.iloc[:num_users, :]

    palette = [mc.rgb2hex(col) for col in sns.color_palette('Blues_r', num_users)]

    src = ColumnDataSource(data=dict(user=users_to_plot.index.values,
                                     counts=users_to_plot.counts.values,
                                     color=palette))

    output_file('../visuals/top_users.html')
    p = figure(plot_width=950, plot_height=600,
               x_range=users_to_plot.index.values,
               y_range=Range1d(0, max(tweets_df.counts) + 500),
               toolbar_location='above',
               x_axis_label='User',
               y_axis_label='Number of tweets',
               title='Tweets per user')
    p.vbar(x='user', top='counts',
           width=0.5,
           fill_color='color',
           source=src)

    hover = HoverTool(tooltips=[('User', '@user'),
                                ('Num tweets', '@counts')])
    p.add_tools(hover)
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = math.pi / 4

    return p


user_plot = user_counts(user_tweets)
show(user_plot)
