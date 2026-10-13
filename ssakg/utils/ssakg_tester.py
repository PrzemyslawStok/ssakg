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

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import sys
import time

from pyprind import ProgBar

from ssakg.ordering_algorithms import SimpleSort, NodeOrderingAlgorithm, EnhancedNodeOrderingAlgorithm, \
    WeightedEdgesNodeOrderingAlgorithm, OrderingAlgorithm, BitBasedSort, BitBasedSortDummy
from ssakg.ssakg import SSAKG


class SSAKG_Tester:
    def __init__(self, ssakg: SSAKG, sequences: np.ndarray, algorithms_list: list[OrderingAlgorithm] = None):
        self.ssakg = ssakg
        self.sequences = sequences
        self.sequence_length = len(sequences[0])
        self.algorithms_test = {}
        self.context_length = None
        self.title = ""
        self.x_label = "Number of correct elements in sequence"
        self.y_label = "Number of sequences"
        self.algorithms_list = algorithms_list
        # SSAKG 2.0 extension has built in sort method (called bits sort)
        self.bit_based = ssakg.bit_based
        self.unsorted_elements_test = [0, 0]
        self.unsorted_percentage = 0
        self.test_time = 0
        if algorithms_list is None:
            self.algorithms_list = [SimpleSort(), NodeOrderingAlgorithm(),
                                    EnhancedNodeOrderingAlgorithm(),
                                    WeightedEdgesNodeOrderingAlgorithm()]
        if self.bit_based:
            self.algorithms_list = [BitBasedSort()]

    def make_test(self, context_length, show_progress=False, ordering_test=True) -> float:
        """Evaluate memory retrieval performance.

        Parameters
        ----------
        context_length : int
            Length of the context used during testing.
        show_progress : bool, default=False
            Whether to display a progress bar during testing.
        ordering_test : bool, default=True
            The correct order of a sequence is determined by various algorithms, which are tested in this process
        Returns
        -------
        float
            Test performance percentage for unordered sequences.
        """
        self.clear()
        self.context_length = context_length

        if context_length > self.sequence_length:
            warnings.warn("Context is longer than sequence length, unable to make test")

        if show_progress:
            bar = ProgBar(len(self.sequences), stream=sys.stdout, title="ssakg test progress", bar_char='█')
        else:
            bar = None

        start_time = time.perf_counter()

        # The context may contain recurring elements. Symbols assigned to such elements are assigned dynamically.
        # To obtain a context with recurring elements,
        # the simplest way is to translate the entire sequence and choose the desired context.

        for i in range(len(self.sequences)):
            context = self._create_context(self.sequences[i], context_length)
            # Now we have the correct context directly. We can read unsorted elements from ssakg.
            # Context is currently translated
            sequence_elements = self.ssakg.get_sequence_elements(context, context_is_translated=True)

            if SSAKG.compare_sets_of_elements(self._sequence_to_compare_elements(self.sequences[i]),
                                              self._sequence_with_unsorted_elements(sequence_elements)):
                self.unsorted_elements_test[0] += 1

                if ordering_test:
                    self.__ordering_test(self._sequence_to_ordering_test(self.sequences[i]), sequence_elements)
            else:
                self.unsorted_elements_test[1] += 1

            if bar is not None:
                bar.update()

        self.unsorted_percentage = (self.unsorted_elements_test[0] /
                                    (self.unsorted_elements_test[0] + self.unsorted_elements_test[1]) * 100)

        self.test_time = time.perf_counter() - start_time

        return self.unsorted_percentage

    def plot_agreement_histogram(self, draw_text: bool = True):
        """Plot the agreement histogram for sequence ordering tests.

        Parameters
        ----------
        draw_text : bool, default=True
            Whether to display text labels on the histogram.
        """
        self.title = f"\n Comparison of algorithms \n Sequence length {self.sequence_length}, context length {self.context_length}"

        dataframe = self.__create_agreements_dataframe()

        fig, ax = plot.subplots()

        bar_plot = sns.barplot(dataframe, x="Correctly_reproduced_elements", y="Correct_no", ax=ax, hue="algorithm",
                               legend=True,
                               palette="bright", width=1)

        bar_plot.legend().set_title("")

        plot.grid()

        if draw_text:
            plot.title(self.title)
            plot.xlabel(self.x_label)
            plot.ylabel(self.y_label)
        else:
            plot.xlabel("")
            plot.ylabel("")

        plot.show()

    def __add_algorithm(self, name: str):
        histogram = np.zeros(self.sequence_length + 1, dtype=np.int16)
        corrections = 0
        all_values = 0
        self.algorithms_test[name] = [histogram, corrections, all_values]

    def __ordering_test(self, translated_sequence, context_sequence):
        use_only_first_path = True

        for algorithm in self.algorithms_list:
            if self.bit_based:
                # The bit-based algorithm have built-in sort function.
                sorted_sequence, _ = self.ssakg.order_sequence(context_sequence, algorithm,
                                                               use_only_first_path)
            else:
                sorted_sequence, _ = self.ssakg.order_sequence(context_sequence, algorithm, use_only_first_path)

            result, agreement = SSAKG.compare_sorted_sequences(translated_sequence, sorted_sequence)
            self.__add_values(agreement, str(algorithm))

    def __add_value(self, algorithm_name: str, agreement: np.ndarray):
        no_result = np.sum(agreement)
        histogram = self.algorithms_test[algorithm_name][0]

        histogram[no_result] += 1

        if no_result >= self.sequence_length:
            self.algorithms_test[algorithm_name][1] += 1

        self.algorithms_test[algorithm_name][2] += 1

    def __add_values(self, agreement: np.ndarray, name: str):
        if name in self.algorithms_test:
            self.__add_value(name, agreement)
        else:
            self.__add_algorithm(name)
            self.__add_value(name, agreement)

    def clear(self):
        self.algorithms_test = {}
        self.unsorted_elements_test = [0, 0]
        self.unsorted_percentage = 0

    def _translate_sequence(self, sequence: np.ndarray) -> np.ndarray:
        return self.ssakg.translate_sequence(sequence)

    def _create_context(self, sequence: np.ndarray, context_length: int) -> np.ndarray:
        translated_sequence = self._translate_sequence(sequence)
        context = SSAKG.context_from_sequence(context_length, translated_sequence)

        return context

    def _sequence_to_compare_elements(self, sequence: np.ndarray) -> np.ndarray:
        return self._translate_sequence(sequence)

    def _sequence_to_ordering_test(self, sequence: np.ndarray) -> np.ndarray:
        return self._translate_sequence(sequence)

    def _sequence_with_unsorted_elements(self, sequence: np.ndarray) -> np.ndarray:
        return sequence

    def __create_agreements_dataframe(self) -> pd.DataFrame:
        correctly_reproduced_elements = []
        correct_no = []
        algorithms_name = []

        for algorithm_name, values_set in self.algorithms_test.items():
            for i, value in enumerate(values_set[0]):
                correctly_reproduced_elements.append(i)
                correct_no.append(value)
                algorithms_name.append(algorithm_name)

        dataframe = pd.DataFrame(
            {"Correctly_reproduced_elements": correctly_reproduced_elements, "Correct_no": correct_no,
             "algorithm": algorithms_name})

        return dataframe

    def __create_dataframe(self):
        data = {}
        headers = []

        data["no sequences"] = []
        data["correct"] = []
        data["incorrect"] = []
        data["correct sort percentage"] = []

        for algorithm_name, values_set in self.algorithms_test.items():

            headers.append(algorithm_name)
            correct_sort = values_set[1]
            all_values = values_set[2]

            correction_sort_percentage = 0
            incorrect = 0

            if correct_sort > 0:
                correction_sort_percentage = correct_sort / all_values * 100
                incorrect = all_values - correct_sort

            data["no sequences"].append(all_values)
            data["correct"].append(correct_sort)
            data["incorrect"].append(incorrect)
            data["correct sort percentage"].append(correction_sort_percentage)

        df = pd.DataFrame(data,
                          index=headers)

        return df

    def get_agreement_percentage(self):
        agreements = []
        for _, values_set in self.algorithms_test.items():
            correct_sort = values_set[1]
            all_values = values_set[2]
            if all_values > 0:
                agreements.append(correct_sort / all_values * 100)
        return agreements

    def get_sorted_percentage(self):
        if len(self.get_agreement_percentage()) == 0:
            return 0
        return max(self.get_agreement_percentage()) / 100 * self.unsorted_percentage

    def get_unsorted_percentage(self):
        return self.unsorted_percentage

    def get_test_time(self):
        return self.test_time

    def print_dataframe(self):
        df = self.__create_dataframe()
        print(df.to_string(formatters={"correct sort percentage": "{:2,.2f}%".format}))

    def __str__(self):
        algorithm_info = ""
        graph_matrix = self.ssakg.get_graph()
        algorithm_info += f"ssakg dimension: {len(graph_matrix)}\n"
        algorithm_info += f"sequence length: {self.sequence_length}\n"
        algorithm_info += f"context length: {self.context_length}\n"
        algorithm_info += f"unordered sequences restored: {self.unsorted_percentage:.2f}%\n"
        algorithm_info += f"ordered sequences restored: {self.get_sorted_percentage():.2f}%\n"
        algorithm_info += f"elapsed time: {self.test_time:.6f}s\n"
        df = self.__create_dataframe()
        algorithm_info += df.to_string(formatters={"correct sort percentage": "{:2,.2f}%".format})

        return algorithm_info
