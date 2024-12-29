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

from ssakg import SSAKG_Tester, SSAKG
from ssakg.ordering_algorithms import OrderingAlgorithm


class SSAKG_miRNA_Tester(SSAKG_Tester):
    def __init__(self, ssakg: SSAKG, sequences: np.ndarray, padding_left, padding_right,
                 algorithms_list: list[OrderingAlgorithm] = None, use_consecutive_context=True):

        self.padding_left = padding_left
        self.padding_right = padding_right
        self.use_consecutive_context = use_consecutive_context
        super().__init__(ssakg, sequences, algorithms_list)

    def _create_context(self, sequence: np.ndarray, context_length: int) -> np.ndarray:
        # For test miRNA sequences we use different context
        translated_sequence = self._translate_sequence(sequence)
        if self.use_consecutive_context:
            padding_window = self.padding_left + self.padding_right
            position = np.random.randint(len(translated_sequence) - context_length - padding_window)
            context = translated_sequence[position + self.padding_left:position + self.padding_right + context_length]
        else:
            context = SSAKG.context_from_sequence(context_length, translated_sequence)

        return context
