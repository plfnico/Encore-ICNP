import os
import sys
import time
import datetime  
import numpy as np
import pandas as pd
from typing import List, Tuple
import gensim
import logging
from gensim import corpora
from gensim.models.ldamodel import LdaModel
if '/mnt/ssd1/encore/open-source' not in sys.path: sys.path.insert(0, '/mnt/ssd1/encore/open-source')
from utils.initialization import *
from utils.distribution_utils import *
from utils.eval import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
gensim_logger = logging.getLogger('gensim')
gensim_logger.setLevel(logging.WARNING)


def train_lomas(app, corpus, dictionary, num_topics=32, num_epochs=100):
    logging.info(f'Start training app {app}')
    num_pass = 10 if len(corpus) >= 100 else 20
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=num_pass, eval_every=1)
    for i in range(num_epochs):
        lda_model.update(corpus)
    perplexity = lda_model.log_perplexity(corpus)
    logging.info(f'Finish training app {app} with perplexity {perplexity}')
    return lda_model, dictionary


def save_model(app, lda_model, corpus, dictionary, model_dir):
    logging.info(f'Save model for app {app} to {model_dir}')
    topic_probs = []
    num_topics = lda_model.num_topics
    for i in range(len(corpus)):
        topic_prob = np.zeros(num_topics)
        for key, value in lda_model.get_document_topics(corpus[i], minimum_probability=None):
            topic_prob[key] = value
        topic_probs.append(topic_prob / sum(topic_prob))

    topic_word_probs = []
    for topic in range(num_topics):
        topic_word_prob = np.zeros(n_size * n_interval)
        for key, value in lda_model.show_topic(topic, topn=None):
            topic_word_prob[int(key)] = value
        topic_word_probs.append(topic_word_prob / sum(topic_word_prob))
        
    topic_probs = np.array(topic_probs)
    topic_word_probs = np.array(topic_word_probs)
    os.makedirs(os.path.join(model_dir, 'topic_probs'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'topic_word_probs'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'dictionary'), exist_ok=True)
    np.savetxt(os.path.join(model_dir, 'topic_probs', app + '.txt'), topic_probs, delimiter=',', fmt='%.3e')
    np.savetxt(os.path.join(model_dir, 'topic_word_probs', app + '.txt'), topic_word_probs, delimiter=',', fmt='%.3e')
    dictionary.save(os.path.join(model_dir, 'dictionary', app + '.dict'))


if __name__ == "__main__":
    os.chdir('/mnt/ssd1/encore/open-source')

    size_dir = './data/size/'
    interval_dir = './data/interval/'
    metadata_dir = './data/metadata/'
    size_cdf = pd.read_csv('./data/cdf/size_cdf.csv')
    interval_cdf = pd.read_csv('./data/cdf/interval_cdf.csv')
    n_size = len(size_cdf) - 1
    n_interval = len(interval_cdf) - 1
    files = os.listdir(size_dir)
    model_dir = './checkpoints/lomas/'
    for file in files:
        app = file.strip('.txt')
        data = get_data(size_dir, interval_dir, file, n_interval)
        dictionary = corpora.Dictionary([[str(token) for token in doc] for doc in data])
        corpus = [dictionary.doc2bow([str(token) for token in doc]) for doc in data]
        num_topics = 32 if len(corpus) <= 100 else 64
        lda_model, dictionary = train_lomas(app, corpus, dictionary, num_topics=num_topics, num_epochs=10)
        save_model(app, lda_model, corpus, dictionary, model_dir)
