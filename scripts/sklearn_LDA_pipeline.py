#!/usr/bin/env python3

import pandas as pd
from pprint import pprint
from time import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

tweet_text = pd.read_csv('../tweets/tweets_text.csv', header=0)
tweet_text.dropna(subset=['lemmas'], inplace=True)

tweet_docs = tweet_text.lemmas.tolist()

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

if __name__ == '__main__':
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

    pd.DataFrame(grid_search.cv_results_).to_csv('./topic_modeling_objects/cv_results_df.csv', index=False)
