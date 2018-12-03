#!/usr/bin/env python3

import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource

# Read in tweets file, drop rows with NaNs
tweets = pd.read_csv('../../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'])
tweets.dropna(subset=['lemmas'], inplace=True)
tweets.reset_index(drop=True, inplace=True)

# Read in file with topic probabilities per tweet
doc_topics = pd.read_csv('../../topic_modeling_objects/topics_per_doc_LDA.csv',
                         header=0)

# Extract the date and dominant topic of each tweet to a new df
tweets = pd.concat([tweets['date'], doc_topics['dominant_topic']], axis=1)

# Count the number of tweets posted per topic per month
topics_per_month = tweets.groupby([tweets.date.dt.year, tweets.date.dt.month, tweets.dominant_topic]).size().to_frame('counts')
topics_per_month.index.rename(['year', 'month'], level=[0, 1], inplace=True)

topics_per_month = topics_per_month.unstack(level=2, fill_value=0)

# Collapse the index back into a single date
topics_per_month.columns = topics_per_month.columns.droplevel(0)
topics_per_month.reset_index(inplace=True)
topics_per_month['date'] = pd.to_datetime(dict(year=topics_per_month.year,
                                               month=topics_per_month.month,
                                               day=[1] * len(topics_per_month)))
topics_per_month.drop(['year', 'month'], axis=1, inplace=True)
topics_per_month.columns = topics_per_month.columns.values.astype('str')


def monthly_topics(tweets_df, out_file='../../visuals/monthly_topics.html'):
    """
    Function to generate a stacked bar plot for the count of tweets by month and topic

    :param tweets_df: dataframe with date index and columns containing counts of tweets per topic
    :param out_file: path to file to save plot
    :return: bokeh plot
    """
    topics = [str(i) for i in range(15)]

    palette = ['#E53935', '#0288D1', '#8E24AA', '#00796B', '#689F38',
               '#D81B60', '#5E35B1', '#AFB42B', '#FBC02D', '#90A4AE',
               '#F57C00', '#1976D2', '#3949AB', '#0097A7', '#8D6E63']

    src = ColumnDataSource(tweets_df.to_dict('list'))

    output_file(out_file)
    p = figure(width=1200, height=800,
               x_axis_type='datetime',
               x_axis_label='Date',
               y_axis_label='Number of tweets',
               title='Tweets per month')

    renderers = p.vbar_stack(topics, x='date', color=palette,
                             width=3.6e8 * 5,
                             legend=["Topic " + x for x in topics],
                             source=src)

    legend = p.legend[0]
    p.legend[0].plot = None
    p.add_layout(legend, 'right')

    for r in renderers:
        topic = r.name
        hover = HoverTool(tooltips=[('Date', '@date{%B-%Y}'),
                                    ('Topic {}'.format(topic), '@$name')],
                          formatters={'date': 'datetime'},
                          mode='mouse',
                          renderers=[r])
        p.add_tools(hover)

    p.xgrid.grid_line_color = None

    return p


# Convert counts of tweets per topic to proportions
monthly_props = topics_per_month.drop('date', axis=1).apply(lambda x: 100. * x / x.sum(), axis=1)
monthly_props['date'] = topics_per_month['date']

# Plot the stacked bar plot with counts and the stacked bar plot with proportions
topics_plot = monthly_topics(topics_per_month)
show(topics_plot)
topics_prop_plot = monthly_topics(monthly_props, out_file='../../visuals/monthly_topics_prop.html')
show(topics_prop_plot)

topics_per_month.set_index('date', inplace=True)
monthly_props.set_index('date', inplace=True)

# TODO: Hovertools not working for stacked area plot


def stacked_area(tweets_df, out_file='../../visuals/topic_stacked_area.html'):
    """
    Function to generate a stacked area plot for the count of tweets by month and topic

    :param tweets_df: dataframe with date index and columns containing counts of tweets per topic
    :param out_file: path to file to save plot
    :return: bokeh plot
    """
    topics = [str(i) for i in range(15)]

    def stacked(df):
        df_top = df.cumsum(axis=1)
        df_bottom = df_top.shift(axis=1).fillna({'y0': 0})[::-1]

        return pd.concat([df_bottom, df_top], ignore_index=True)

    areas = stacked(tweets_df)
    colors = ['#E53935', '#0288D1', '#8E24AA', '#00796B', '#689F38',
              '#D81B60', '#5E35B1', '#AFB42B', '#FBC02D', '#90A4AE',
              '#F57C00', '#1976D2', '#3949AB', '#0097A7', '#8D6E63']

    x2 = np.hstack((tweets_df.index[::-1], tweets_df.index))

    output_file(out_file)
    p = figure(width=1800, height=600,
               x_axis_type='datetime',
               x_axis_label='Date',
               y_axis_label='Number of tweets',
               title='Tweets per month')
    p.grid.minor_grid_line_color = '#eeeeee'

    renderers = []
    for topic, (c, color) in enumerate(zip(areas, colors)):
        renderers.append(p.patch(x2, areas[c],
                         color=color, alpha=1, line_color=None,
                         legend='Topic ' + str(topic)))

    # renderers = p.patches([x2] * areas.shape[1], [areas[c].values for c in areas],
    #                       color=colors, alpha=0.8, line_color=None, legend=['Topic ' + x for x in topics])

    legend = p.legend[0]
    p.legend[0].plot = None
    p.add_layout(legend, 'right')

    for r in renderers:
        topic = r.name
        hover = HoverTool(tooltips=[('Date', '@date{%B-%Y}'),
                                    ('Topic {}'.format(topic), '@$name')],
                          formatters={'date': 'datetime'},
                          mode='mouse',
                          renderers=[r])
        p.add_tools(hover)

    return p


# Plot the stacked area plot with counts and the stacked area plot with proportions
area_plot = stacked_area(topics_per_month)
show(area_plot)

area_prop_plot = stacked_area(monthly_props, out_file='../../visuals/topic_prop_stacked_area.html')
show(area_prop_plot)
