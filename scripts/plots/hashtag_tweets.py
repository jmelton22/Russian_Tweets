#!/usr/bin/env python3

import pandas as pd
import ast
from collections import Counter
import math
import matplotlib.colors as mc
import seaborn as sns
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource, Range1d

tweets = pd.read_csv('../../tweets/tweets_clean.csv', header=0)

# Get a list of all hashtags from pandas column containing a list of hashtags per tweet
hashtags = [tag.lower() for tag_list in tweets.hashtags for tag in ast.literal_eval(tag_list)]

# Create df containing the counts of each hashtag, sort by number of occurences
hashtag_df = pd.DataFrame.from_dict(Counter(hashtags),
                                    orient='index',
                                    columns=['counts']).sort_values(by='counts', ascending=False)


def hashtag_counts(tweets_df, num_hashtags=25):
    """
    Function to generate a bar chart of top hashtags by number of usages in tweets

    :param tweets_df: sorted dataframe with index of each hashtag in data, column is count of occurences
    :param num_hashtags: Number of top hashtags to plot
    :return: bokeh plot
    """
    hashtags_to_plot = tweets_df.iloc[:num_hashtags, :]

    palette = [mc.rgb2hex(col) for col in sns.color_palette('Purples_r', num_hashtags)]

    src = ColumnDataSource(data=dict(hashtag=hashtags_to_plot.index.values,
                                     counts=hashtags_to_plot.counts.values,
                                     color=palette))

    output_file('../../visuals/top_hashtags.html')
    p = figure(plot_width=950, plot_height=700,
               x_range=hashtags_to_plot.index.values,
               y_range=Range1d(0, max(tweets_df.counts) + 500),
               toolbar_location='above',
               x_axis_label='Hashtag',
               y_axis_label='Number of tweets',
               title='Tweets per hashtag')
    p.vbar(x='hashtag', top='counts',
           width=0.5,
           fill_color='color',
           source=src)

    hover = HoverTool(tooltips=[('Hashtag', '@hashtag'),
                                ('Num tweets', '@counts')])
    p.add_tools(hover)
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = math.pi / 4
    p.xaxis.major_label_text_font_size = "14pt"

    return p


fig = hashtag_counts(hashtag_df)
show(fig)
