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
import warnings

from ssakg.anakg import ANAKG
from ssakg.ordering_algorithms import OrderingAlgorithm, WeightedEdgesNodeOrderingAlgorithm

import ssakg_extension as ssakg_ext


class SSAKG(ANAKG):
    def __init__(self, number_of_symbols: int = 10, sequence_length: int = 5, dtype=None, graphs_to_drawing=False,
                 remove_diagonals=True, weighted_edges=True, bits_graph=False, bits_tests=False):
        super().__init__(number_of_symbols, sequence_length, graphs_to_drawing, remove_diagonals,
                         weighted_edges, bits_graph, dtype)

        self.bits_tests = bits_tests
        self.new_sequences_added = False

    def eval_non_zero_elements_loops(self, graph: np.ndarray, context=None) -> np.ndarray:
        graph_rows_no = len(graph)
        context_length = len(context)

        context_array = np.zeros((context_length, graph_rows_no), dtype=np.int16)
        row_prod = np.ones(graph_rows_no, dtype=np.int16)
        pyton_elements = "["
        for i in range(graph_rows_no):
            for j in range(context_length):
                context_array[j, i] = graph[context[j], i] + graph[i, context[j]]

                if i == context[j]:
                    context_array[j, i] += 1

                if context_array[j, i] != 0:
                    pyton_elements += " " + str(context_array[j, i])

                row_prod[i] *= context_array[j, i] != 0
                if row_prod[i] == 0:
                    break
        pyton_elements += " ]"
        print(f"elements: {pyton_elements}")
        non_zeros = np.where(row_prod != 0)[0]

        return non_zeros

    def get_unsorted_elements(self, context: np.ndarray, context_is_translated=False) -> (
            np.ndarray, np.ndarray):

        if context_is_translated:
            translated_context = context
            unique_context = np.unique(context)
            if len(unique_context) != len(context):
                warnings.warn(
                    "List of symbols contains duplicate elements. Please use underscored symbols instead e.g. \"5_2\".")
                return None, None
        else:
            translated_context = self.translate_context(context)

        if translated_context is None:
            return None, None

        if not self.bits_tests:
            unsorted_elements = ssakg_ext.get_unsorted_elements(self.graph, translated_context.astype(
                dtype=np.uint32))
            return unsorted_elements, unsorted_elements
        else:
            sorted_sequence, unsorted_sequence = ssakg_ext.get_sorted_elements_bits(self.graph,
                                                                                    context.astype(np.uintp))
            # if np.random.randint(0, 100) < 5:
            #     unsorted_sequence[0] = 0

            return sorted_sequence, unsorted_sequence

    def __get_sequence(self, context: np.ndarray, decode_sequence=True, context_is_translated=False,
                       ordering_alg=WeightedEdgesNodeOrderingAlgorithm()) -> np.ndarray | list:

        unsorted_elements, _ = self.get_unsorted_elements(context, context_is_translated)

        if unsorted_elements is None:
            return None
        sorted_elements, _ = self.order_sequence(unsorted_elements, ordering_alg=ordering_alg,
                                                 use_only_first_path=True)

        if decode_sequence:
            return self.decode_sequence(sorted_elements)

        return sorted_elements

    def get_sequence(self, context):
        context = self.sequence_from_placeholders(context)
        return self.__get_sequence(context, decode_sequence=True, context_is_translated=True)

    def insert_sequence(self, sequence: np.ndarray) -> np.ndarray | None:
        self.new_sequences_added = True
        return super().insert_sequence(sequence)

    def insert(self, sequences: np.ndarray):
        self.new_sequences_added = True
        return super().insert(sequences)

    @staticmethod
    def context_from_sequence(context_length: int, sequence: np.ndarray) -> np.ndarray:
        indexes = np.random.permutation(len(sequence))
        indexes = indexes[:context_length]

        context_array = sequence[indexes]
        return context_array

    @staticmethod
    def compare_sorted_sequences(sequence_1: np.ndarray, sequence_2: np.ndarray) -> (bool, np.ndarray):
        agreement = np.zeros(len(sequence_1), dtype=np.int8)

        if sequence_1 is None or sequence_2 is None:
            return False, agreement

        if len(sequence_1) != len(sequence_2):
            return False, agreement

        for i in range(len(sequence_1)):
            if sequence_1[i] == sequence_2[i]:
                agreement[i] = 1

        return np.array_equal(sequence_1, sequence_2), agreement

    @staticmethod
    def compare_sets_of_elements(sequence_1: np.ndarray, sequence_2: np.ndarray) -> bool:
        if len(sequence_1) != len(sequence_2):
            return False
        else:
            sequence_1 = np.sort(sequence_1)
            sequence_2 = np.sort(sequence_2)

            if not np.array_equal(sequence_1, sequence_2):
                return False

            return True

    def order_sequence(self, sequence: np.ndarray, ordering_alg: OrderingAlgorithm | None = None,
                       use_only_first_path=False) -> (
            np.ndarray | None, int):
        sequence_matrix = self.get_subgraph_from_sequence(sequence)
        if sequence_matrix is None:
            return None, 0

        orderings = ordering_alg(sequence_array=sequence_matrix, use_only_first_path=use_only_first_path)

        new_ordering = orderings[0]
        sorted_sequence = sequence[new_ordering]

        return sorted_sequence, len(orderings)
