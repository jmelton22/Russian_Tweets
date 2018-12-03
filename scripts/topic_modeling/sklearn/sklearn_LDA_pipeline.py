#!/usr/bin/env python3

import pandas as pd
from pprint import pprint
from time import time

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load in file containing tweets, drop rows with NaN
tweets = pd.read_csv('../../../tweets/tweets_clean.csv', header=0)
tweets.dropna(subset=['lemmas'], inplace=True)
tweets.reset_index(drop=True, inplace=True)

# Extract lemmatized tweets as list
tweet_docs = tweets.lemmas.tolist()

# Create a Pipeline for testing various parameters of the count vectorizer and LDA model
pipeline = Pipeline([('vect', CountVectorizer(max_df=0.95,
                                              min_df=100)),
                     ('lda', LatentDirichletAllocation(max_iter=10,
                                                       learning_method='online'))])
params = {'lda__learning_decay': (0.5, 0.7, 0.9),
          'lda__n_components': (2, 5, 10, 15, 20, 25, 30, 35),
          # 'vect__max_features': (None, 50000, 100000),
          # 'vect__max_df': (0.5, 0.75, 0.95),
          # 'vect__min_df': (0, 10, 100)
          }

# For parallel processing (on Windows), branching point has to be within a "main" function
if __name__ == '__main__':
    # Perform grid search cross-validation on tweet data to test and score models with passed parameters
    grid_search = GridSearchCV(pipeline, params,
                               cv=5, n_jobs=-1, verbose=2)
    print('Performing grid search')
    print('Pipeline:', [name for name, _ in pipeline.steps])
    print('Parameters:')
    pprint(params)
    t0 = time()
    print('Start time: {}'.format(t0))
    grid_search.fit(tweet_docs)
    print('Completed in {:.3f}s'.format(time() - t0))
    print()

    print('Best Log Likelihood: {:.3f}'.format(grid_search.best_score_))
    print('Best Parameters set:')
    best_params = grid_search.best_params_
    for k, v in best_params.items():
        print('{}: {}'.format(k, v))

    # Save the results of cross validation to a csv file
    pd.DataFrame(grid_search.cv_results_).to_csv('../../../results_csv/cv_results_df.csv', index=False)
