import re
import nltk
import collections
import numpy as np

from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def preprocessortext(docs):
    preprocessed_doc = []
    for doc in docs:
        # Pad punctuation with spaces on both sides
        doc = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", doc)
        doc = doc.lower()

        ## Clean the doc
        doc = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", doc)
        doc = re.sub(r"what's", "what is ", doc)
        doc = re.sub(r"\'s", " ", doc)
        doc = re.sub(r"\'ve", " have ", doc)
        doc = re.sub(r"n't", " not ", doc)
        doc = re.sub(r"i'm", "i am ", doc)
        doc = re.sub(r"\'re", " are ", doc)
        doc = re.sub(r"\'d", " would ", doc)
        doc = re.sub(r"\'ll", " will ", doc)
        doc = re.sub(r",", " ", doc)
        doc = re.sub(r"\.", " ", doc)
        doc = re.sub(r"!", " ", doc)
        doc = re.sub(r"\/", " ", doc)
        doc = re.sub(r"\^", " ", doc)
        doc = re.sub(r"\+", " ", doc)
        doc = re.sub(r"\-", " ", doc)
        doc = re.sub(r"\=", " ", doc)
        doc = re.sub(r"'", " ", doc)

        words = word_tokenize(doc)
        words = [word.lower() for word in words if word.isalpha()]
        doc = ' '.join(words)
        preprocessed_doc.append(doc)

    return preprocessed_doc


def wordtokenize(texts):
    texts = [re.sub(r"'s$", '', text) for text in texts]
    words = [word_tokenize(text) for text in texts]
    return words


def get_stop_words():
    stop_words = list()

    stop_words.extend(
        ['a', 'an', 'the', 'this', 'that', 'they', 'if', 'in', 'into', 'is', 'of', 'on', 'or', 'when', 'what',
         'where'])
    stop_words.extend(['for', 'to', 'do', 'at', 'so', 'who', 'which', 'it', 'and', 'as', 'be', 'was', 'are', 'were'])
    stop_words.extend(
        ['all', 'am', 'any', 'any', 'by', 'bye', 'day', 'go', 'he', 'her', 'him', 'im', 'ive', 'me', 'my'])
    stop_words.extend(['na', 'our', 'she', 'them', 'then', 'there', 'theyre', 'too', 'u', 'up', 'we', 'why', 'yall'])

    return stop_words


def get_pos_tagger():
    from nltk.corpus import brown
    regexp_tagger = nltk.RegexpTagger(
        [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
         (r'(The|the|A|a|An|an)$', 'AT'),  # articles
         (r'.*able$', 'JJ'),  # adjectives
         (r'.*ness$', 'NN'),  # nouns formed from adjectives
         (r'.*ly$', 'RB'),  # adverbs
         (r'.*s$', 'NNS'),  # plural nouns
         (r'.*ing$', 'VBG'),  # gerunds
         (r'.*ed$', 'VBD'),  # past tense verbs
         (r'.*', 'NN')  # nouns (default)
         ])
    brown_train = brown.tagged_sents()
    unigram_tagger = UnigramTagger(brown_train, backoff=regexp_tagger)
    bigram_tagger = BigramTagger(brown_train, backoff=unigram_tagger)
    trigram_tagger = TrigramTagger(brown_train, backoff=bigram_tagger)

    # Override particular words
    main_tagger = nltk.RegexpTagger(
        [(r'(A|a|An|an)$', 'ex_quant'),
         (r'(Every|every|All|all)$', 'univ_quant')
         ], backoff=trigram_tagger)

    return main_tagger


tagger = get_pos_tagger()


def tokenizetext(doc):
    lemmatizer = WordNetLemmatizer()
    tagged_text = tagger.tag(word_tokenize(doc))
    words = [lemmatizer.lemmatize(tag[0], get_wordnet_pos(tag[1])) for tag in tagged_text]

    # ps = PorterStemmer()
    # words = [ps.stem(word) for word in words]
    return words


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def filter_text(texts):
    stop_words = get_stop_words()
    lemmatizer = WordNetLemmatizer()

    words = [word_tokenize(text) for text in texts]
    tagger = get_pos_tagger()
    tagged_texts = [tagger.tag(word) for word in words]

    filtered_texts = [[ lemmatizer.lemmatize(tag[0].lower(), get_wordnet_pos(tag[1]))
                          for tag in taglist if ('NN' in tag[1]) and (tag[0].lower() not in stop_words)
                          and tag[0].isalpha() and len(tag[0]) > 2]
                          for taglist in tagged_texts]

    filtered_texts = [' '.join(wordlist) for wordlist in filtered_texts]

    return filtered_texts# -*- coding: utf-8 -*-

