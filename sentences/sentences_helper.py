# Copyright 2024 The SSAKG Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from nltk import word_tokenize

from nltk import FreqDist


def compute_statistics(sentences_split) -> (np.ndarray, np.ndarray):
    sentences_data = np.zeros(len(sentences_split), dtype=int)

    for i in range(len(sentences_split)):
        sentences_data[i] = len(sentences_split[i])

    max_sentence_length = np.max(sentences_data)

    sentences_histo = np.zeros(max_sentence_length + 1, dtype=int)

    for sentence in sentences_split:
        sentences_histo[len(sentence)] += 1

    return sentences_data, sentences_histo


def create_word_histo(dictionary: dict[str, int]) -> (np.ndarray, np.ndarray):
    words_histo = np.zeros(len(dictionary), dtype=int)
    words_data = []

    sorted_values = np.flip(np.sort(list(dictionary.values())))

    for i, words_counter in enumerate(sorted_values):
        words_histo[i] = words_counter

        for j in range(words_counter):
            words_data.append(i)

    return np.array(words_data), words_histo


def split_sentences(sentences: list[str]) -> list[str]:
    sentences_split = []
    for sentence in sentences:
        sentences_split.append(word_tokenize(sentence))

    return sentences_split


def len_range_filter(sentence: list[str], sentence_len_range) -> bool:
    if len(sentence) in sentence_len_range:
        return True
    else:
        return False


def sentences_filter(sentences_split, sentence_filter, params) -> list[list[str]]:
    filtered_sentences = []
    for sentence in sentences_split:
        if sentence_filter(sentence, params):
            filtered_sentences.append(sentence)
    return filtered_sentences


def to_uppercase(sentences: list[list[str]]) -> list[list[str]]:
    sentences_upper = []
    for sentence in sentences:
        uppercase_sentence = []
        for word in sentence:
            uppercase_sentence.append(word.upper())
        sentences_upper.append(uppercase_sentence)

    return sentences_upper


def words_dictionary_from_numbers_sequences(sentences: np.ndarray) -> FreqDist:
    f_dist = FreqDist()
    for sentence in sentences:
        for word in sentence:
            str_word = str(word)
            f_dist[str_word] += 1

    return f_dist
