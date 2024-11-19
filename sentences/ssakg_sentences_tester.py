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

from ssakg.ordering_algorithms import OrderingAlgorithm
from ssakg.ssakg import SSAKG
from ssakg.utils.ssakg_tester import SSAKG_Tester

import numpy as np


class SSAKGSentencesTester(SSAKG_Tester):
    def __init__(self, ssakg: SSAKG, reserved_symbols_no: int, sequences: np.ndarray,
                 algorithms_list: list[OrderingAlgorithm] = None):
        super().__init__(ssakg, sequences, algorithms_list)
        self.reserved_symbols_no = reserved_symbols_no

    def remove_reserved_symbols(self, sequence: np.ndarray) -> np.ndarray:
        return sequence[sequence > self.reserved_symbols_no]

    def _translate_sequence(self, sequence: np.ndarray) -> np.ndarray:
        return super()._translate_sequence(sequence)

    def _create_context(self, sequence: np.ndarray, context_length: int) -> np.ndarray:
        translated_sequence = self._translate_sequence(sequence)
        sequence_with_no_reserved_symbols = self.remove_reserved_symbols(translated_sequence)

        context = SSAKG.context_from_sequence(context_length, sequence_with_no_reserved_symbols)
        return context

    def _sequence_to_compare_elements(self, sequence: np.ndarray) -> np.ndarray:
        translated_sequence = self._translate_sequence(sequence)
        return self.remove_reserved_symbols(translated_sequence)

    def _sequence_to_ordering_test(self, sequence: np.ndarray) -> np.ndarray:
        return self._translate_sequence(sequence)

    def _sequence_with_unsorted_elements(self, sequence: np.ndarray) -> np.ndarray:
        return self.remove_reserved_symbols(sequence)
