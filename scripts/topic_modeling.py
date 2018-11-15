#!/usr/bin/env python3

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    import imp
warnings.simplefilter('ignore')

import pandas as pd
from pprint import pprint

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import spacy

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import matplotlib.pyplot as plt


def main():
    tweets = pd.read_csv('../tweets/tweets_clean.csv',
                         header=0,
                         parse_dates=['date'])
    print('Reading in tweets')
    tweets = list(tweets.clean_text.dropna())[:10000]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    print('Tokenizing and removing stop words')
    tweets = [[word for word in word_tokenize(tweet)
               if word not in stop_words]
              for tweet in tweets]
    nlp = spacy.load('en', disable=['parser', 'ner'])

    def lemmatization(texts, allowed_postags=('NOUN', 'ADJ', 'VERB', 'ADV')):
        lemmas = []
        for i, tweet in enumerate(texts):
            if i % 1000 == 0 or i == len(tweets)-1:
                print('Lemmatized {} tweets'.format(i+1))
            lemmas.append([token.lemma_ for token in nlp(' '.join(tweet)) if token.pos_ in allowed_postags])

        return lemmas

    tweet_lemmas = lemmatization(tweets)

    id2word = corpora.Dictionary(tweet_lemmas)
    print('Converting docs to bags of words')
    corpus = [id2word.doc2bow(tweet) for tweet in tweet_lemmas]
    print()

    # readable_corpus = [[(id2word[i], freq) for i, freq in cp] for cp in corpus]

    def score_models(dictionary, corpus, texts, limit, start=2, step=3):
        """
        Compute c_v coherence scores for LDA models with various numbers of topics

        :param dictionary: Gensim dictionary
        :param corpus: Gensim corpus
        :param texts: List of input texts
        :param limit: Max num of topics
        :return:
            model_list: List of LDA topic models
            coherences: Coherence values corresponding to LDA model by number of topics
            log_perplexities: Log perplexity values corresponding to LDA model by number of topics
        """
        model_list, coherences, log_perplexities = [], [], []
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
            log_perplexities.append(model.log_perplexity(corpus))
            print('Fitting Coherence model with {} topics'.format(num_topics))
            coherence_model = CoherenceModel(model=model,
                                             texts=texts,
                                             dictionary=dictionary,
                                             coherence='c_v')
            coherences.append(coherence_model.get_coherence())
            print()

        return model_list, coherences, log_perplexities

    models, coherences, log_perplexities = score_models(dictionary=id2word,
                                                        corpus=corpus,
                                                        texts=tweet_lemmas,
                                                        start=2,
                                                        limit=40,
                                                        step=6)

    limit, start, step = 40, 2, 6
    x = range(start, limit, step)
    plt.plot(x, coherences)
    plt.xlabel('Num Topics')
    plt.ylabel('Coherence Score')
    plt.title('Coherence scores of LDA models by numbers of topics')
    plt.show()

    for m, c, p in zip(x, coherences, log_perplexities):
        print('Num topics {}\nCoherence score = {:.4f}\nLog perplexity = {:.4f}'.format(m, c, p))
        print()

    # Select optimal model
    # optimal_model = models[3]
    # model_topics = optimal_model.show_topics()
    # pprint(model_topics)


if __name__ == "__main__":
    main()
