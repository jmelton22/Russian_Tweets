#!/usr/bin/env python3

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

# Load in csv file with the results of LDA model cross-validation testing
results = pd.read_csv('../../../results_csv/cv_results_df.csv', header=0)
results['params'] = results.params.apply(ast.literal_eval)

# Sort each model by its score and print the scores and parameters for each model
for _, row in results.sort_values(by='rank_test_score').iterrows():
    print(u'Rank {}, Score {:.3f} \u00B1 {:.3f}'.format(row['rank_test_score'],
                                                        row['mean_test_score'],
                                                        row['std_test_score']))
    for k, v in row['params'].items():
        print('{}: {}'.format(k, v))
    print()

# Extract mean and stdev of test score for each model by the learning decay parameter
num_topics = [2, 5, 10, 15, 20, 25, 30, 35]
scores_5 = results.loc[results['param_lda__learning_decay'] == 0.5, 'mean_test_score']
std_5 = results.loc[results['param_lda__learning_decay'] == 0.5, 'std_test_score']
scores_7 = results.loc[results['param_lda__learning_decay'] == 0.7, 'mean_test_score']
std_7 = results.loc[results['param_lda__learning_decay'] == 0.7, 'std_test_score']
scores_9 = results.loc[results['param_lda__learning_decay'] == 0.9, 'mean_test_score']
std_9 = results.loc[results['param_lda__learning_decay'] == 0.9, 'std_test_score']

# Plot the mean score of models by num topics and by learning decay
# Fill area 1 stdev beyond mean score
fig = plt.figure(figsize=(10, 6))

plt.plot(num_topics, scores_5,
         color='orange', label='0.5', linewidth=2)
plt.fill_between(num_topics,
                 (np.array(scores_5) - np.array(std_5)), (np.array(scores_5) + np.array(std_5)),
                 color='orange', alpha=0.2)
plt.plot(num_topics, scores_7,
         color='cornflowerblue', label='0.7', linewidth=2)
plt.fill_between(num_topics,
                 (np.array(scores_7) - np.array(std_7)), (np.array(scores_7) + np.array(std_7)),
                 color='cornflowerblue', alpha=0.2)
plt.plot(num_topics, scores_9,
         color='forestgreen', label='0.9', linewidth=2)
plt.fill_between(num_topics,
                 (np.array(scores_9) - np.array(std_9)), (np.array(scores_9) + np.array(std_9)),
                 color='forestgreen', alpha=0.2)

plt.xlabel('Number of Topics')
plt.ylabel('Log Likelihood Score')
plt.title('Cross-Validation Scores for LDA Models')
plt.legend(title='Learning Decay', loc='best', frameon=True)

fig.savefig('../../../visuals/cv_scores.png')

plt.show()
