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

import os
import numpy as np

class FastaHelper:
    def __init__(self,dirname:str, *filenames:str,recreate_files=False):
        self.dirname = dirname
        self.filenames = filenames
        self.recreate_files = recreate_files
        self.converted_filenames = self.__convert_mirbase_files()
        self.sequences = self.__read_sequences()

    def __convert_mirbase_files(self) -> list[str]:
        current_line = ""
        header_line = "Info;Sequence\n"
        converted_text = header_line

        converted_filenames_list = []

        for filename in self.filenames:
            name, _ = os.path.splitext(os.path.basename(filename))
            extension = ".csv"
            filename_converted = f"{name}_converted{extension}"

            if not os.path.exists(os.path.join(self.dirname, filename_converted)) or self.recreate_files:
                file_path = os.path.join(self.dirname, filename)
                with open(file_path, "r") as file:
                    for line in file:
                        if ">" in line:
                            if len(current_line) > 0:
                                converted_text += current_line + "\n"

                            current_line = line[1:].strip() + ";"
                        else:
                            current_line += line.strip()

                with open(os.path.join(self.dirname, filename_converted), "w") as file:
                    file.write(converted_text)

            converted_filenames_list.append(filename_converted)

        return converted_filenames_list

    def __read_sequences(self) -> np.ndarray | None:
        sequences_list = []

        for filename in self.converted_filenames:
            file_path = os.path.join(self.dirname, filename)
            data = np.genfromtxt(file_path, dtype=None, delimiter=";", encoding="UTF-8", names=True)

            for _, sequence in data:
                sequences_list.append(sequence)

        unique_sequences = np.unique(sequences_list)

        return np.array(unique_sequences)

    def get_filter_sequences(self, sequence_length: int, no_sequences=None) -> np.ndarray:
        filtered = self.sequences[np.char.str_len(self.sequences) == sequence_length]
        return filtered[:no_sequences]
