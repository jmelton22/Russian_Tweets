#!/usr/bin/env python3

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

tweets = pd.read_csv('../tweets/tweets_clean.csv',
                     header=0,
                     parse_dates=['date'])
tweets.dropna(subset=['lemmas'], inplace=True)

# Load vectorizer
with open('../scripts/topic_modeling_objects/sklearn_vect.joblib', 'rb') as f:
    cv = joblib.load(f)

# Load term frequency matrix
with open('../scripts/topic_modeling_objects/sklearn_CV.joblib', 'rb') as f:
    tf = joblib.load(f)

# Load feature names
with open('../scripts/topic_modeling_objects/sklearn_feature_names.joblib', 'rb') as f:
    tf_names = joblib.load(f)

# Load fitted LDA model
with open('../scripts/topic_modeling_objects/sklearn_LDA_model.joblib', 'rb') as f:
    lda_model = joblib.load(f)

lda_W = lda_model.transform(tf)

# Construct k-means clusters
clusters = KMeans(n_clusters=15,
                  random_state=100).fit_predict(lda_W)
colors = ['#E53935', '#D81B60', '#8E24AA', '#5E35B1', '#3949AB',
          '#1976D2', '#0288D1', '#0097A7', '#00796B', '#8D6E63',
          '#689F38', '#AFB42B', '#FBC02D', '#90A4AE', '#F57C00']

# Build Singluar Value Decomposition (SVD) model
svd_model = TruncatedSVD(n_components=2)
lda_output_svd = svd_model.fit_transform(lda_W)

x = lda_output_svd[:, 0]
y = lda_output_svd[:, 1]

print('Component\'s weights:\n', np.round(svd_model.components_, 2))
print('Percent variance explained:\n', np.round(svd_model.explained_variance_ratio_, 2))

fig = plt.figure(figsize=(12, 12))
for topic, color in zip(np.unique(clusters), colors):
    i = np.where(clusters == topic)
    plt.scatter(x[i], y[i], c=color, label=topic, alpha=0.2)

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Segregation of Topic Clusters')
leg = plt.legend(title='Topic', loc='best', ncol=3)

for lh in leg.legendHandles:
    lh.set_alpha(1)

fig.savefig('../visuals/LDA_topic_clusters.png')
plt.show()
