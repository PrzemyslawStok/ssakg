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


class OrderingAlgorithm:
    def __init__(self, dtype=np.int16):
        self.dtype = dtype

    def __call__(self, sequence_array: np.ndarray, use_only_first_path=False):
        all_paths = self.__ordering_algorithm(sequence_array, [], [], max_index=len(sequence_array) + 1,
                                              use_only_first_path=use_only_first_path)
        new_paths = []
        for path in all_paths:
            new_paths.append(OrderingAlgorithm.ordering_to_matrix_indices(path))

        return new_paths

    def __str__(self):
        pass

    def _ordering_function(self, array: np.ndarray, max_index: int) -> np.ndarray:
        pass

    @staticmethod
    def find_indexes_with_min_zeros_count(array: np.ndarray):
        number_of_zeros_in_row = np.sum(array == 0, axis=1)
        min_zeros = np.min(number_of_zeros_in_row)

        indices_with_min_zeros_values = np.where(number_of_zeros_in_row == min_zeros)[0]
        return indices_with_min_zeros_values

    @staticmethod
    def ordering_to_matrix_indices(path: np.ndarray, dtype=np.int16) -> np.ndarray:
        new_path = np.zeros(len(path), dtype=dtype)
        matrix_indices = np.arange(0, len(path), dtype=int)

        for i in range(len(path)):
            new_path[i] = matrix_indices[path[i]]
            matrix_indices = np.delete(matrix_indices, path[i])

        return new_path

    def __ordering_algorithm(self, array: np.array, path: list, all_paths: list,
                             max_index=None, use_only_first_path=False):
        if len(array) == 0:
            all_paths.append(path)
            return all_paths

        # 1) The program search for line numbers, witch contains desired conditions

        indexes = self._ordering_function(array, max_index=max_index)

        # 2) If we find desired rows we can reduce matrix

        if len(indexes) == 0:
            all_paths.append(path)

        if use_only_first_path:
            indexes = [indexes[0]]

        for index in indexes:
            reduced_array = np.delete(array, index, 0)
            reduced_array = np.delete(reduced_array, index, 1)

            new_path = path.copy()
            new_path.append(index)
            self.__ordering_algorithm(reduced_array, new_path, all_paths,
                                      max_index=max_index, use_only_first_path=use_only_first_path)

            if len(array) == max_index:
                break

        return all_paths


class NodeOrderingAlgorithm(OrderingAlgorithm):
    def _ordering_function(self, array: np.ndarray, max_index: int):
        return OrderingAlgorithm.find_indexes_with_min_zeros_count(array)

    def __str__(self):
        return "Node Ordering"


class SimpleSort(NodeOrderingAlgorithm):
    def __call__(self, sequence_array: np.ndarray, use_only_first_path=False):
        return self.__sort(sequence_array)

    def __sort(self, sequence_array: np.ndarray) -> list:
        only_zeros = np.where(sequence_array == 0, 1, 0)
        number_zeros_in_columns = np.sum(only_zeros, axis=1)

        sorted_indexes = [np.argsort(number_zeros_in_columns)]

        return sorted_indexes

    def __str__(self):
        return "Simple Sort"


class SimpleSortWeighted(NodeOrderingAlgorithm):
    def __call__(self, sequence_array: np.ndarray, use_only_first_path=False):
        return self.__sort(sequence_array)

    def __sort(self, sequence_array: np.ndarray) -> list:
        weight_row_sum = SimpleSortWeighted.create_wight_array_row_sum(sequence_array)

        only_zeros = np.where(sequence_array == 0, 1, 0)
        number_zeros_in_columns = np.sum(only_zeros, axis=1)

        important_weights = np.rec.fromarrays([number_zeros_in_columns, weight_row_sum])
        comparator = important_weights.astype(dtype=[('x', int), ('y', int)])

        sorted_indexes = [np.argsort(comparator, order=['x', 'y'])]

        return sorted_indexes

    @staticmethod
    def create_wight_array_row_sum(sequence_matrix: np.ndarray) -> np.ndarray:
        weight_array = sequence_matrix.copy()
        weight_array = EnhancedNodeOrderingAlgorithm.create_wight_array(weight_array)

        return np.sum(weight_array, axis=1)

    def __str__(self):
        return "Simple Sort Weighted"


class EnhancedNodeOrderingAlgorithm(OrderingAlgorithm):
    @staticmethod
    def create_wight_array(sequence_matrix: np.ndarray) -> np.ndarray:
        weight_array = sequence_matrix.copy().astype(dtype=float)

        with np.errstate(divide='ignore'):
            weight_array = np.where(weight_array == 0, weight_array, 1 / weight_array)

        return weight_array

    def _ordering_function(self, array: np.ndarray, max_index: int):
        indices_with_no_zeros = OrderingAlgorithm.find_indexes_with_min_zeros_count(array)

        if len(indices_with_no_zeros) == 0:
            return np.array([])

        weight_matrix = EnhancedNodeOrderingAlgorithm.create_wight_array(array)
        row_no_zeros_weight_sum = np.sum(weight_matrix[indices_with_no_zeros], axis=1)
        no_zeros_weight_sum = np.vstack((indices_with_no_zeros, row_no_zeros_weight_sum))

        min_indexes = np.where(no_zeros_weight_sum[1] == np.min(no_zeros_weight_sum[1]))[0]
        weight_min_indexed_array = no_zeros_weight_sum[:, min_indexes]
        weight_min_indexed_row = weight_min_indexed_array[0]

        return weight_min_indexed_row.astype(int)

    def __str__(self):
        return "Enhanced Node Ordering"


class WeightedEdgesNodeOrderingAlgorithm(OrderingAlgorithm):
    def _ordering_function(self, array: np.ndarray, max_index: int):
        indices_with_no_zeros = OrderingAlgorithm.find_indexes_with_min_zeros_count(array)

        if len(indices_with_no_zeros) == 0:
            return np.array([])

        correct_number = max_index - len(array)

        # !? this code can contain error

        max_index_array = indices_with_no_zeros

        if len(indices_with_no_zeros) > 1:
            max_correct_numbers = 0
            current_number_list = []

            for index in indices_with_no_zeros:
                current_correct_numbers = len(np.where(array[index] == correct_number)[0])

                if current_correct_numbers == max_correct_numbers:
                    current_number_list.append(index)

                if current_correct_numbers > max_correct_numbers:
                    current_number_list = [index]

                    max_correct_numbers = current_correct_numbers

            max_index_array = np.array(current_number_list).astype(int)

        return max_index_array

    def __str__(self):
        return "Weighted Edges Ordering"


class BitsNodeOrderingAlgorithm(OrderingAlgorithm):
    def _ordering_function(self, array: np.ndarray, max_index: int):
        indices_with_no_zeros = OrderingAlgorithm.find_indexes_with_min_zeros_count(array)

        if len(indices_with_no_zeros) == 0:
            return np.array([])

        bit_no = max_index - len(array)
        correct_number = 2 ** (bit_no - 1)

        max_index_array = indices_with_no_zeros

        if len(indices_with_no_zeros) > 1:
            max_correct_numbers = 0
            current_number_list = []

            for index in indices_with_no_zeros:
                current_correct_numbers = len(np.where(array[index] == correct_number)[0])

                if current_correct_numbers == max_correct_numbers:
                    current_number_list.append(index)

                if current_correct_numbers > max_correct_numbers:
                    current_number_list = [index]

                    max_correct_numbers = current_correct_numbers

            max_index_array = np.array(current_number_list).astype(int)

        return max_index_array

    def __str__(self):
        return "Bits Node Ordering"
