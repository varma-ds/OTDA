import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils import shuffle

from src.utils import *
from src.OTDA import *
from src.text_processing import *

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import silhouette_score, davies_bouldin_score

dataFolder = Path('../data')
corpus = 'data.csv'
batch_size = 500
num_topics = 8


def generate_batches(texts, batch_size):
    for i in range(0, len(texts), batch_size):
        yield texts[i:i+batch_size]


def getvocabulary(texts):
    tfvectorizer = CountVectorizer(max_df=0.8, min_df=5,
                                   stop_words=get_stop_words(),
                                   tokenizer=tokenizetext)
    tfvectorizer.fit_transform(texts)
    return tfvectorizer, tfvectorizer.vocabulary_


def get_docterm_matrix(texts, vocab, method='tfidf'):

    if method == 'tfidf':
        tfidfvectorizer = TfidfVectorizer(max_df=0.8, min_df=5,
                                          stop_words=get_stop_words(),
                                          tokenizer=tokenizetext,
                                          vocabulary=vocab)
        docterm_tfidf = tfidfvectorizer.fit_transform(texts)
        return docterm_tfidf

    if method == 'tf':
        tfvectorizer = CountVectorizer(max_df=0.8, min_df=5,
                                       stop_words=get_stop_words(),
                                       tokenizer=tokenizetext,
                                       vocabulary=vocab)
        docterm_tf = tfvectorizer.fit_transform(texts)
        return docterm_tf


def otda(X, model, prevS=None):
    pseudo_count = 0.01
    S = np.dot(X.T, X)
    S = S.toarray() + prevS

    memorize = 1

    tc = np.sum(S)

    wc_m = np.sum(S, axis=1)[:, None]
    S = np.asarray(S, dtype=np.float64)
    pmi = S + pseudo_count
    pmi = np.log((tc * pmi / wc_m) / wc_m.T)
    pmi[np.isinf(pmi)] = 0.0
    pmi[pmi < 0.0] = 0.0

    H1 = np.random.rand(X.shape[0], model.n_topic)
    H1 = model.nmf_iter(X.T.toarray(), pmi, H1, memorize)
    W1, W2 = model.get_lowrank_matrix()
    H = H1

    return W1.T, H, prevS


def main():
    data_df = readdata(dataFolder/corpus)
    data_df = shuffle(data_df, random_state=0)
    texts = data_df['text']
    print(len(texts))

    texts = preprocessortext(texts)
    texts = filter_text(texts)

    vectorizer, vocab = getvocabulary(texts)
    print('vocab', len(vocab))

    doctopic = None
    model = OTDA(
        len(vocab), num_topics,
        alpha=0.1,
        beta=0,
        max_iter=100,
        max_err=0.0001,
        fix_seed=True)

    for batch_texts in generate_batches(texts, batch_size):
        docterm_m = get_docterm_matrix(batch_texts, vocab, method='tf')
        print(docterm_m.shape)
        prevS = 0
        W, H, prevS = otda(docterm_m, model, prevS)
        print(W.shape, H.shape)

        if isinstance(doctopic, type(None)) == True:
            doctopic = H
        else:
            doctopic = np.vstack((doctopic, H))

    cluster = np.argmax(doctopic, axis=1)
    nmi = NMI(data_df["Tag1"], cluster, average_method='arithmetic')
    print('NMI', nmi)



if __name__== "__main__":
    main()# -*- coding: utf-8 -*-

