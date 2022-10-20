import re
from collections import OrderedDict

import numpy as np
import pymorphy2
import spacy
from spacy.lang.ru.stop_words import STOP_WORDS

nlp = spacy.load("ru_core_news_sm")


class SearchKeyWord:

    def __init__(self):
        self.d = 0.85
        self.min_diff = 1e-5
        self.steps = 10
        self.node_weight = None
        self.morph = pymorphy2.MorphAnalyzer()

    def delete_stopwords(self, stopwords):

        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, lower):
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                if '.' not in token.text:
                    if token.pos_ in candidate_pos and token.is_stop is False:
                        if lower is True:
                            selected_words.append(token.text.lower())
                        else:
                            selected_words.append(token.text)
                else:
                    pass
            sentences.append(selected_words)
        return sentences

    def get_vocab(self, sentences):
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                word = re.sub('[^А-яа-я0-9ё.-]+', ' ', word)
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(self, window_size, sentences):
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i + 1, i + window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(self, vocab, token_pairs):
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        g = self.symmetrize(g)

        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm != 0)

        return g_norm

    def get_keywords(self, number=10):
        key_words_s = ''
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            if i < number:
                key_words_s += key + ' - ' + str(value) + '\n'
            if i == number:
                key_words_s += key + '-' + str(value)
            if i > number:
                break
        return key_words_s

    def lem_text(self, text):
        text_new = ""
        for i in range(text):
            sentence_new = ""
            for j in range(text[i]):
                word = self.morph.normal_forms(text[i][j])[0]
                sentence_new += word
        text_new += sentence_new
        return text_new

    def analyze(self, text,
                candidate_pos=['NOUN', 'PROPN'],
                window_size=2, lower=False, stopwords=list()):

        self.delete_stopwords(stopwords)

        doc = nlp(text)

        sentences = self.sentence_segment(doc, candidate_pos, lower)

        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                sentences[i][j] = self.morph.normal_forms(sentences[i][j])[0]
        vocab = self.get_vocab(sentences)
        token_pairs = self.get_token_pairs(window_size, sentences)

        g = self.get_matrix(vocab, token_pairs)

        pr = np.array([1] * len(vocab))

        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1 - self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight


model = SearchKeyWord()
