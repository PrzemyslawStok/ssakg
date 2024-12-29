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
import re

import warnings


class SequenceTranslator:
    def __init__(self, no_unique_symbols: int, sequence_length: int, dtype=np.int16):
        self.base_sequence_length = sequence_length
        self.no_unique_symbols = no_unique_symbols
        self.dtype = dtype
        self.sequence_symbols_dictionary = np.zeros((self.no_unique_symbols, self.base_sequence_length),
                                                    dtype=self.dtype)
        self.sequence_tuples_dictionary = np.zeros((0, 2), dtype=self.dtype)

        self.sequence_symbols_counter = np.zeros([self.no_unique_symbols], dtype=self.dtype)
        self.max_sequence = self.__max_sequence_number()
        self.multiple_symbol_pattern = re.compile(r"^\s*(\d+)_(\d+)\s*$")

    def __max_sequence_number(self):
        return np.max(self.sequence_symbols_dictionary)

    def __symbol_number_from_tuple(self, symbol_tuple):
        # This function returns encoded number of multiple symbol.
        if symbol_tuple[1] == 1:
            return symbol_tuple[0]
        else:
            address_value = self.sequence_symbols_dictionary.item(symbol_tuple)
            return address_value + self.no_unique_symbols

    def __translate_sequence_tuples(self, sequence_tuple: np.ndarray) -> np.ndarray:
        # This function translates sequences with repeated elements into sequences with unique elements.

        new_sequence = np.zeros(len(sequence_tuple), dtype=self.dtype)
        for index, enumerated_sequence_tuple in enumerate(sequence_tuple):
            sequence_address = (enumerated_sequence_tuple[0], enumerated_sequence_tuple[1] - 1)
            address_value = self.sequence_symbols_dictionary.item(sequence_address)
            if address_value == 0:
                new_sequence[index] = enumerated_sequence_tuple[0]
            else:
                new_sequence[index] = address_value + self.no_unique_symbols - 1

        return new_sequence

    def __sequence_to_enumerated_tuples(self, sequence: np.ndarray) -> np.ndarray:
        current_sequence_symbols_counter = np.copy(self.sequence_symbols_counter)

        # The first item of tuple stores symbol,
        # the second item stores number of occurrences of the symbol in the sequence.
        sequence_tuples = np.zeros([len(sequence), 2], dtype=self.dtype)

        for i in range(len(sequence)):
            current_sequence_symbols_counter[sequence[i]] += 1
            sequence_tuples[i] = (sequence[i], current_sequence_symbols_counter[sequence[i]])

        return sequence_tuples

    def translate_sequence(self, sequence: np.ndarray, add_to_dictionary=True) -> (np.ndarray, int):
        rows_to_add = 0
        sequence_tuples = self.__sequence_to_enumerated_tuples(sequence)

        for enumerated_sequence_item in sequence_tuples:
            if enumerated_sequence_item[1] > 1:
                sequence_address = (enumerated_sequence_item[0], enumerated_sequence_item[1] - 1)
                if self.sequence_symbols_dictionary.item(sequence_address) == 0:
                    if add_to_dictionary:
                        # The sequence contains doubled elements, not exists in sequences dictionary.
                        # This section adds doubled elements to the dictionary.
                        self.max_sequence += 1
                        self.sequence_symbols_dictionary[sequence_address] = self.max_sequence
                        self.sequence_tuples_dictionary = np.append(self.sequence_tuples_dictionary,
                                                                    np.array([sequence_address]), axis=0)
                        rows_to_add += 1
                    else:
                        # This sequence contains doubled elements with are not exist in sequences dictionary.
                        # This kind of sequence is not possible to translate. Returning None
                        return None, 0

        new_sequence_tuple = self.__translate_sequence_tuples(sequence_tuples)

        return new_sequence_tuple, rows_to_add

    def decode_sentence(self, sequence: np.ndarray) -> np.ndarray:
        current_enumerated_sequence = np.zeros(np.shape(sequence), dtype=self.dtype)

        for (i, symbol) in enumerate(sequence):
            if symbol < len(self.sequence_symbols_dictionary):
                current_enumerated_sequence[i] = symbol
            else:
                symbol_index = symbol - len(self.sequence_symbols_dictionary)
                if symbol_index >= len(self.sequence_symbols_dictionary):
                    current_enumerated_sequence[i] = None
                else:
                    current_enumerated_sequence[i] = self.sequence_tuples_dictionary[symbol_index][0]

        return current_enumerated_sequence

    def __check_symbol_value_in_range(self, symbol: int) -> bool:
        if symbol > self.no_unique_symbols or symbol < 0:
            return False
        else:
            return True

    def __symbol_to_tuple(self, symbol: str) -> tuple[int, int] | None:
        result = re.match(self.multiple_symbol_pattern, symbol)
        if result is None or len(result.groups()) != 2:
            return None
        return int(result.group(1)), int(result.group(2))

    def sequence_from_placeholders(self, sequence: list[str]) -> np.ndarray:
        decoded_sequence = []
        for symbol in sequence:
            if type(symbol) == str:
                tuple_symbol = self.__symbol_to_tuple(symbol)
                if self.__check_symbol_value_in_range(tuple_symbol[0]):
                    symbol_number = self.__symbol_number_from_tuple(tuple_symbol)
                    decoded_sequence.append(symbol_number)
                else:
                    warnings.warn(f"Symbol {symbol} not in range, will be omitted.")
            else:
                if self.__check_symbol_value_in_range(int(symbol)):
                    decoded_sequence.append(symbol)
                else:
                    warnings.warn(f"Symbol {symbol} not in range, will be omitted.")

        return np.array(decoded_sequence)

    def __str__(self):
        return str(self.sequence_symbols_dictionary)
