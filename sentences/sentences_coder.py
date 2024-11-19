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
from numpy.dtypes import StrDType


class SentencesCoder:
    def __init__(self, words_dictionary: [str, int], reserved_symbols_no: int = 0, randomize_symbols=False):
        self.no_words = len(words_dictionary)
        self.no_symbols = self.no_words + reserved_symbols_no
        self.word_codes = np.zeros(self.no_symbols, dtype=StrDType)
        self.word_dictionary = {}

        self.reserved_symbols_no = reserved_symbols_no
        symbols = np.arange(self.no_words)

        if randomize_symbols:
            symbols = np.random.permutation(symbols)

        for i, word in enumerate(words_dictionary):
            self.word_dictionary[word] = symbols[i] + reserved_symbols_no
            self.word_codes[symbols[i] + reserved_symbols_no] = word

    def expand_sentence(self, encoded_sentence: list[int], new_length: int,
                        add_symbols_on_end=False) -> np.ndarray | None:
        if new_length < len(encoded_sentence):
            return None

        empty_codes_no = new_length - len(encoded_sentence)

        if empty_codes_no > self.reserved_symbols_no:
            return None

        sentence_array = np.array(encoded_sentence)
        empty_codes_array = np.random.choice(self.reserved_symbols_no, size=empty_codes_no, replace=False)

        for code in empty_codes_array:
            random_position = np.random.randint(len(encoded_sentence))
            if add_symbols_on_end:
                sentence_array = np.append(sentence_array, code)
            else:
                sentence_array = np.insert(sentence_array, random_position, code)

        return sentence_array

    def sentences_split_to_array(self, sentences_split: list[list[int]], sentence_length: int) -> np.ndarray:
        sentences_array = np.empty((0, sentence_length), dtype=np.int16)

        for sentence in sentences_split:
            if len(sentence) == sentence_length:
                sentences_array = np.append(sentences_array, np.expand_dims(np.array(sentence), axis=0), axis=0)
            elif len(sentence) < sentence_length:
                expanded_sentence = self.expand_sentence(sentence, sentence_length)
                sentences_array = np.append(sentences_array, np.expand_dims(expanded_sentence, axis=0), axis=0)

        return sentences_array

    def encode_sentences(self, sentences: list[list[str]]) -> list[list[int]]:
        encoded_sentences = []
        for sentence in sentences:
            words = []
            for word in sentence:
                words.append(self.word_dictionary[word])

            encoded_sentences.append(words)
        return encoded_sentences

    def decode_sentences(self, encoded_sentences: list[list[int]]) -> list[list[str]]:
        decoded_sentences = []
        for encoded_sentence in encoded_sentences:
            words = []
            for i in encoded_sentence:
                if i >= self.reserved_symbols_no:
                    words.append(str(self.word_codes[i]))

            decoded_sentences.append(words)
        return decoded_sentences


if __name__ == '__main__':
    dictionary = {}
    for number, a in enumerate("abcdefgh"):
        dictionary[a] = number

    print(dictionary)
    sentences_coder = SentencesCoder(dictionary, reserved_symbols_no=10, randomize_symbols=True)
    sentence_to_test = ["a", "b", "c", "d", "e", "f", "g"]
    print(sentence_to_test)
    encoded = sentences_coder.encode_sentences([sentence_to_test])
    print(encoded)
    expanded = sentences_coder.expand_sentence(encoded[0], 10)
    print(expanded)
    decoded = sentences_coder.decode_sentences([expanded])
    print(decoded)
