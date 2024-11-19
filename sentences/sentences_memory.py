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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
from nltk import FreqDist

import sentences_helper as sentences_helper
from gutenberg_sentences import read_sentences
from sentences_coder import SentencesCoder
from ssakg.ssakg import SSAKG
from ssakg.utils.sequence_generator import SequenceGenerator
from ssakg_sentences_tester import SSAKGSentencesTester


def create_words_histo(words_dictionary: dict[str, int], draw_text: bool = True):
    words_data, word_histo = sentences_helper.create_word_histo(words_dictionary)

    title = f"Most common words in the text"
    x_label = "Words number"
    y_label = "Number of words in the text"

    fig, ax = plt.subplots()

    sns.histplot(pd.DataFrame({"x": words_data}), x="x", ax=ax, bins="auto")

    plt.grid()

    if draw_text:
        plt.title(
            title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.tight_layout()
    plt.show()


def create_sentences_histo(sentences_split: list[list[str]], draw_text: bool = True):
    sentences_data, _ = sentences_helper.compute_statistics(sentences_split)

    title = f"Statistical distribution of sentence length in text"
    x_label = "Sentence length"
    y_label = "Number of sentences"

    fig, ax = plt.subplots()
    ax.hist(sentences_data, bins='auto')

    plt.grid()

    if draw_text:
        plt.title(
            title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    fig.tight_layout()
    plt.show()


def create_random_sequences(sequence_length: int, number_sequences: int, symbol_min: int, symbol_max: int) -> np.ndarray:
    sequence_generator = SequenceGenerator(sequence_length=sequence_length, sequence_min=symbol_min,
                                           sequence_max=symbol_max)

    sequences = sequence_generator.generate_unique_sequences(number_sequences=number_sequences,
                                                            unique_elements=True)

    return sequences


def create_graph_with_sentences(encoded_sequences: np.ndarray, words_coder: SentencesCoder,
                                remove_diagonals=True,
                                weighted_edges=True) -> (
        SSAKG, np.ndarray):
    sequence_length = np.shape(encoded_sequences)[1]
    graph_dim = len(words_coder.word_codes)

    ssakg = SSAKG(number_of_symbols=graph_dim, sequence_length=sequence_length,
                  remove_diagonals=remove_diagonals, weighted_edges=weighted_edges)

    for i in range(len(encoded_sequences)):
        ssakg.insert_sequence(encoded_sequences[i])

    return ssakg


def crate_encodes_sentences(corpus_dir: str, corpus_path: str, base_sequence_length: int,
                            max_sentences: int, corpus_length_range: range,
                            recreate_corpus, output_length_range: range, reserved_symbols: int,
                            randomize_symbols: bool):
    sentences_split, word_freq_dist = read_sentences(new_corpus_dir=corpus_dir, new_corpus_path=corpus_path,
                                                     corpus_length_range=corpus_length_range,
                                                     recreate_corpus=recreate_corpus,
                                                     output_length_range=output_length_range,
                                                     max_sentences=max_sentences)
    words_coder = SentencesCoder(words_dictionary=word_freq_dist, reserved_symbols_no=reserved_symbols,
                                 randomize_symbols=randomize_symbols)
    original_encoded = words_coder.encode_sentences(sentences=sentences_split)

    expanded_sequences = words_coder.sentences_split_to_array(original_encoded, base_sequence_length)

    return words_coder, expanded_sequences, word_freq_dist


def create_test(expanded_sentences: np.ndarray, words_coder: SentencesCoder,
                context_length: int, draw_density=True):
    ssakg = create_graph_with_sentences(expanded_sentences, words_coder)

    if draw_density:
        ssakg.draw_density(x_label="Words number", y_label="Words number", )

    memory_comparator = SSAKGSentencesTester(ssakg, words_coder.reserved_symbols_no, expanded_sentences)
    memory_comparator.make_test(context_length=context_length)
    memory_comparator.plot_agreement_histogram(draw_text=False)
    print(memory_comparator)


def draw_most_common(words_freq_dist: FreqDist, no_most_common: int, draw_text=True):
    most_common = dict(words_freq_dist.most_common(no_most_common))
    dataframe = pd.DataFrame({"word": most_common.keys(), "count": most_common.values()})
    palette = sns.color_palette("crest", n_colors=len(dataframe))

    fig, ax = plt.subplots()

    bar_plot = sns.barplot(dataframe, x="word", y="count", ax=ax, hue="word", legend=False,
                           palette=palette)

    if draw_text:
        bar_plot.set_title(f"Most common words in text ({no_most_common} words)")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.xticks(rotation=45)
    plt.grid()
    fig.tight_layout()
    plt.show()

    words_freq_dist.plot(no_most_common, title=f"Most common words in text ({no_most_common} words)",
                         cumulative=False)


def example_1(corpus_dir: str, corpus_path: str, sequence_length: int, context_length: int, max_sentences: int,
              use_random_sentences=False, recreate_corpus=False, randomize_symbols=True, draw_text=True,
              draw_density=True):
    print("---------------------Example 1---------------------")
    use_random_sentences = use_random_sentences

    words_coder, expanded_sequences, word_freq_dist = crate_encodes_sentences(corpus_dir=corpus_dir,
                                                                              corpus_path=corpus_path,
                                                                              base_sequence_length=sequence_length,
                                                                              max_sentences=max_sentences,
                                                                              recreate_corpus=recreate_corpus,
                                                                              corpus_length_range=range(
                                                                                  10, 20),
                                                                              output_length_range=range(
                                                                                  15, 16),
                                                                              reserved_symbols=0,
                                                                              randomize_symbols=randomize_symbols)

    print(f"Different words (graph dim): {len(word_freq_dist)}")

    if use_random_sentences:
        encoded_sentences = create_random_sequences(sequence_length=sequence_length,
                                                    number_sequences=max_sentences,
                                                    symbol_min=0, symbol_max=len(words_coder.word_codes))
        word_freq_dist = sentences_helper.words_dictionary_from_numbers_sequences(encoded_sentences)

    no_most_common = 20
    draw_most_common(word_freq_dist, no_most_common, draw_text=draw_text)
    create_words_histo(dict(word_freq_dist), draw_text=draw_text)

    create_test(expanded_sequences, words_coder, context_length, draw_density)


print("---------------------------------------------------")


def example_2(corpus_dir: str, corpus_path: str, base_sequence_length: int, sequence_length_range: range,
              context_length, max_sentences: int, recreate_corpus: bool, output_length_range: range,
              reserved_symbols: int, randomize_symbols: bool, draw_text=True, draw_density=True):
    print("---------------------Example 2---------------------")
    words_coder, expanded_sequences, word_freq_dist = crate_encodes_sentences(corpus_dir=corpus_dir,
                                                                              corpus_path=corpus_path,
                                                                              base_sequence_length=base_sequence_length,
                                                                              max_sentences=max_sentences,
                                                                              recreate_corpus=recreate_corpus,
                                                                              corpus_length_range=sequence_length_range,
                                                                              output_length_range=output_length_range,
                                                                              reserved_symbols=reserved_symbols,
                                                                              randomize_symbols=randomize_symbols)

    print(f"Different words (graph dim): {len(word_freq_dist)}")

    draw_most_common(word_freq_dist, 20, draw_text=draw_text)
    create_words_histo(dict(word_freq_dist), draw_text=draw_text)
    create_test(expanded_sequences, words_coder, context_length, draw_density)

    print("---------------------------------------------------")


if __name__ == "__main__":
    example_1_test = False
    example_2_test = True

    randomize_examples_symbols = False
    draw_examples_text = False

    draw_examples_density = True

    if example_1_test:
        example_1(corpus_dir="./data", corpus_path="example_15.txt", sequence_length=15, context_length=8,
                  max_sentences=1000,
                  use_random_sentences=False, randomize_symbols=randomize_examples_symbols, draw_text=draw_examples_text,
                  draw_density=draw_examples_density)
    if example_2_test:
        example_2(corpus_dir="./data", corpus_path="example_10-15.txt", base_sequence_length=15, context_length=8,
                  sequence_length_range=range(10, 20),
                  max_sentences=1000, recreate_corpus=False, output_length_range=range(10, 16), reserved_symbols=100,
                  randomize_symbols=randomize_examples_symbols, draw_text=draw_examples_text, draw_density=draw_examples_density)
