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

import nltk
from nltk import FreqDist
from nltk import downloader
from nltk.corpus import gutenberg
from nltk.corpus import stopwords

from nltk.corpus import PlaintextCorpusReader

from sentences_helper import len_range_filter, sentences_filter, to_uppercase

from pyprind import ProgBar
import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def install_nltk_packages():
    # This function install required nltk packages
    nltk_downloader = downloader.Downloader()

    if not nltk_downloader.is_installed("stopwords"):
        nltk.download("stopwords")
    if not nltk_downloader.is_installed("gutenberg"):
        nltk.download("gutenberg")
    if not nltk_downloader.is_installed("punkt"):
        nltk.download("punkt")
    if not nltk_downloader.is_installed("punkt_tab"):
        nltk.download("punkt_tab")


def remove_stopwords(sentences: list[list[str]], stop_words=None, show_progress=True,
                     progbar_title="Removing Stopwords") -> list[list[str]]:
    if stop_words is None:
        stop_words = set(stopwords.words('english'))

    new_sentences = []

    bar = None
    if show_progress:
        bar = ProgBar(len(sentences), stream=1, title=progbar_title)

    for sentence in sentences:
        new_sentence = [word for word in sentence if word.lower() not in stop_words]
        new_sentences.append(new_sentence)

        if show_progress and bar is not None:
            bar.update()

    return new_sentences


def create_sentences(length_range: range, uppercase=False) -> list[list[str]]:
    print("Creating sentences...")
    print("Reading Gutenberg corps...")

    install_nltk_packages()

    split = gutenberg.sents(gutenberg.fileids())

    if uppercase:
        split = to_uppercase(split)

    split = remove_stopwords(split, show_progress=True, progbar_title="Removing stopwords from Gutenberg Corpus")
    print("Filtering sentences...")
    split = sentences_filter(split, len_range_filter, length_range)

    return split


def create_sentences_from_gutenberg_corpus(new_corpus_dir: str, new_corpus_path: str, length_range: range,
                                           uppercase=False) -> (
        list[str], FreqDist):
    path = os.path.join(new_corpus_dir, new_corpus_path)
    split = create_sentences(uppercase=uppercase, length_range=length_range)

    if not os.path.exists(new_corpus_dir):
        os.makedirs(new_corpus_dir)

    with open(path, "w") as file:
        for sentence in split:
            line = ""
            for word in sentence:
                line += word + " "
            file.write(line)

    return split


def read_sentences(new_corpus_dir: str, new_corpus_path: str, corpus_length_range: range,
                   recreate_corpus, output_length_range: range, max_sentences=500, uppercase=False) -> (
        list[list[str]], FreqDist):
    path = os.path.join(new_corpus_dir, new_corpus_path)

    if not os.path.exists(path) or recreate_corpus:
        create_sentences_from_gutenberg_corpus(new_corpus_dir, new_corpus_path, corpus_length_range,
                                               uppercase=uppercase)

    new_corpus = PlaintextCorpusReader(new_corpus_dir, new_corpus_path, encoding="latin1")

    sentences = new_corpus.sents()
    words_dist = FreqDist(new_corpus.words())

    stop_words = dict(words_dist.most_common(5)).keys()
    sentences = remove_stopwords(sentences, stop_words=stop_words, show_progress=True,
                                 progbar_title="Removing most common words form filtered sentences")

    words_dist = FreqDist()

    sentences = sentences_filter(sentences, len_range_filter, output_length_range)
    sentences = sentences[:max_sentences]

    for sentence in sentences:
        words_dist.update(FreqDist(sentence))

    return sentences, words_dist


if __name__ == "__main__":
    sentences_split, word_freq_dist = read_sentences(new_corpus_dir="./data", new_corpus_path="example_15.txt",
                                                     corpus_length_range=range(15, 20),
                                                     recreate_corpus=True, output_length_range=range(15, 16),
                                                     uppercase=False)
    print(f"Different sentences: {len(sentences_split)}")
    print(f"Different words no: {len(word_freq_dist)}")

    most_common = dict(word_freq_dist.most_common(20))
    pd_dataframe = pd.DataFrame({"word": most_common.keys(), "count": most_common.values()})
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.set_theme()
    palette = sns.color_palette("dark:#5A9_r", n_colors=len(pd_dataframe))

    bar_plot = sns.barplot(pd_dataframe, x="word", y="count", ax=ax, hue="word", legend=False,
                           palette=palette)
    plt.xticks(rotation=45)
    plt.show()
