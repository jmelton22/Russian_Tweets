#!/usr/bin/env python3

import joblib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

# Read in file with tweets, drop rows with NaN
tweets = pd.read_csv('../../../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'])
tweets.dropna(subset=['lemmas'], inplace=True)
tweets.reset_index(drop=True, inplace=True)

# Load count vectorizer
with open('../../../topic_modeling_objects/sklearn_vect.joblib', 'rb') as f:
    cv = joblib.load(f)

# Load term frequency matrix
with open('../../../topic_modeling_objects/sklearn_CV.joblib', 'rb') as f:
    tf = joblib.load(f)

# Load feature names
with open('../../../topic_modeling_objects/sklearn_feature_names.joblib', 'rb') as f:
    tf_names = joblib.load(f)

# Load fitted LDA model
with open('../../../topic_modeling_objects/sklearn_LDA_model.joblib', 'rb') as f:
    lda_model = joblib.load(f)

# Generate topic to document matrix (W)
lda_W = lda_model.transform(tf)

# Construct k-means clusters model
clusters = KMeans(n_clusters=15,
                  random_state=100).fit_predict(lda_W)

colors = ['#E53935', '#0288D1', '#8E24AA', '#00796B', '#689F38',
          '#D81B60', '#5E35B1', '#AFB42B', '#FBC02D', '#90A4AE',
          '#F57C00', '#1976D2', '#3949AB', '#0097A7', '#8D6E63']

# Build 1D Singular Value Decomposition (SVD) model
svd_model = TruncatedSVD(n_components=1)
lda_output_svd = svd_model.fit_transform(lda_W)

x = lda_output_svd[:, 0]
print('Component\'s weights:\n', np.round(svd_model.components_, 2))
print('Percent variance explained:\n', np.round(svd_model.explained_variance_ratio_, 2))

# Plot 1D SVD clustering of topics
fig1 = plt.figure(figsize=(12, 12))
for topic, color in zip(np.unique(clusters), colors):
    i = np.where(clusters == topic)
    plt.plot(x[i], c=color, label=topic, alpha=0.2)

plt.ylabel('Component 1')
plt.title('Segregation of Topic Clusters 1D')
leg = plt.legend(title='Topic', loc='best', ncol=3)

for lh in leg.legendHandles:
    lh.set_alpha(1)

fig1.savefig('../../../visuals/LDA_topic_clusters_1D.png')

# Build 2D Singular Value Decomposition (SVD) model
svd_model = TruncatedSVD(n_components=2)
lda_output_svd = svd_model.fit_transform(lda_W)

x = lda_output_svd[:, 0]
y = lda_output_svd[:, 1]

print('Component\'s weights:\n', np.round(svd_model.components_, 2))
print('Percent variance explained:\n', np.round(svd_model.explained_variance_ratio_, 2))

# Plot 2D SVD clustering of topics
fig2 = plt.figure(figsize=(12, 12))
for topic, color in zip(np.unique(clusters), colors):
    i = np.where(clusters == topic)
    plt.scatter(x[i], y[i],
                c=color, label=topic, alpha=0.2)

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Segregation of Topic Clusters 2D')
leg = plt.legend(title='Topic', loc='best', ncol=3)

for lh in leg.legendHandles:
    lh.set_alpha(1)

fig2.savefig('../../../visuals/LDA_topic_clusters_2D.png')
plt.show()
