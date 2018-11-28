#!/usr/bin/env python3

import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource, Legend

tweets = pd.read_csv('../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'])
tweets.dropna(subset=['lemmas'], inplace=True)
tweets.reset_index(drop=True, inplace=True)

doc_topics = pd.read_csv('./topic_modeling_objects/topics_per_doc_LDA.csv',
                         header=0)

tweets['topic'] = doc_topics['dominant_topic']

topics_per_month = tweets.groupby([tweets.date.dt.year, tweets.date.dt.month, tweets.topic]).size().to_frame('counts')
topics_per_month.index.rename(['year', 'month'], level=[0, 1], inplace=True)

topics_per_month = topics_per_month.unstack(level=2, fill_value=0)

# # Collapse index back into a single date
topics_per_month.columns = topics_per_month.columns.droplevel(0)
topics_per_month.reset_index(inplace=True)
topics_per_month['date'] = pd.to_datetime(dict(year=topics_per_month.year,
                                               month=topics_per_month.month,
                                               day=[1] * len(topics_per_month)))
topics_per_month.drop(['year', 'month'], axis=1, inplace=True)
topics_per_month.columns = topics_per_month.columns.values.astype('str')


def monthly_topics(tweets_df, out_file='../visuals/monthly_topics.html'):
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


topics_plot = monthly_topics(topics_per_month)
show(topics_plot)

monthly_props = topics_per_month.drop('date', axis=1).apply(lambda x: 100. * x / x.sum(), axis=1)
monthly_props['date'] = topics_per_month['date']

topics_prop_plot = monthly_topics(monthly_props, out_file='../visuals/monthly_topics_prop.html')
show(topics_prop_plot)
