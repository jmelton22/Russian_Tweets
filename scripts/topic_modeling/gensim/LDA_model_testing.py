#!/usr/bin/env python3

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    import imp
warnings.simplefilter('ignore')

from pprint import pprint
import pickle

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import matplotlib.pyplot as plt

# Load lemmatized tweets
with open('../../../topic_modeling_objects/lemmas.pkl', 'rb') as f:
    tweet_lemmas = pickle.load(f)

# Load gensim dictionary
id2word = corpora.Dictionary.load('../../../topic_modeling_objects/dictionary.pkl')

# Load gensim corpus
with open('../../../topic_modeling_objects/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

# readable_corpus = [[(id2word[i], freq) for i, freq in cp] for cp in corpus]


if __name__ == "__main__":

    def score_models(dictionary, corpus, texts, limit, start=2, step=4):
        """
        Compute c_v coherence scores and log perplexity values for LDA models with various numbers of topics

        :param dictionary: Gensim dictionary
        :param corpus: Gensim corpus
        :param texts: List of input texts
        :param limit: Max num of topics
        :return:
            model_list: List of LDA topic models
            coherences: Coherence values for each LDA model by number of topics
            log_perplexities: Log perplexity valuess for each LDA model by number of topics
        """
        model_list, coherence_list, log_perplexity_list = [], [], []
        for num_topics in range(start, limit, step):
            print('Fitting LDA model with {} topics'.format(num_topics))
            model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=num_topics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)
            model_list.append(model)
            log_perplexity_list.append(model.log_perplexity(corpus))
            print('Fitting Coherence model with {} topics'.format(num_topics))
            coherence_model = CoherenceModel(model=model,
                                             texts=texts,
                                             dictionary=dictionary,
                                             coherence='c_v')
            coherence_list.append(coherence_model.get_coherence())
            print()

        return model_list, coherence_list, log_perplexity_list

    models, coherences, log_perplexities = score_models(dictionary=id2word,
                                                        corpus=corpus,
                                                        texts=tweet_lemmas,
                                                        start=5,
                                                        limit=50,
                                                        step=5)

    start, limit, step = 5, 50, 5
    x = range(start, limit, step)

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Num topics')
    ax1.set_ylabel('Coherence', color=color)
    ax1.plot(x, coherences, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('Log Perplexity', color=color)
    ax2.plot(x, log_perplexities, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Coherence and Log Perplexity of LDA Models by Number of Topics')

    fig.savefig('../../../visuals/LDA_model_scores.png')

    plt.show()

    for m, c, p in zip(x, coherences, log_perplexities):
        print('Num topics {}\nCoherence score = {:.4f}\nLog perplexity = {:.4f}'.format(m, c, p))
        print()

    # Select optimal model
    optimal_model = models[coherences.index(max(coherences))]
    model_topics = optimal_model.show_topics()
    pprint(model_topics)
