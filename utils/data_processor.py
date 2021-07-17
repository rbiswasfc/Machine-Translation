import math
from typing import List

import numpy as np
import nltk
import sentencepiece as spm

nltk.download("punkt")


def pad_sents(sents, pad_token):
    """
    Pad list of sentences according to the longest sentence in the batch.
    Notes:
    (1) the paddings should be at the end of each sentence
    (2) sentences shorter than the max length sentence are padded out 
    with the pad_token
    (3) each sentences in the processed batch now has equal length

    :param sents: list of sentences comprising of words
    :type sents: List[List[str]]
    :param pad_token: a token string to represent padding
    :type pad_token: str
    :return: padded list of sentences
    :rtype: List[List[str]]
    """
    max_len = len(max(sents, key=lambda x: len(x)))

    for i, sent in enumerate(sents):
        sent.extend([pad_token] * (max_len - len(sent)))
        sents[i] = sent
    return sents


def read_corpus(file_path, source, vocab_size=2500):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    @param vocab_size (int): number of unique subwords in
        vocabulary when reading and tokenizing
    """
    data = []
    sp = spm.SentencePieceProcessor()
    sp.load("{}.model".format(source))

    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            subword_tokens = sp.encode_as_pieces(line)
            # only append <s> and </s> to the target sentence
            if source == "tgt":
                subword_tokens = ["<s>"] + subword_tokens + ["</s>"]
            data.append(subword_tokens)

    return data


def autograder_read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = nltk.word_tokenize(line)
        # only append <s> and </s> to the target sentence
        if source == "tgt":
            sent = ["<s>"] + sent + ["</s>"]
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size : (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

