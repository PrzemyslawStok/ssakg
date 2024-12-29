import numpy as np


class MicroRNAEncoder:
    def __init__(self, symbol_padding_left=3, symbol_padding_right=2):
        self.dictionary = {"A": 0, "C": 1, "G": 2, "U": 3}

        if symbol_padding_left + symbol_padding_right > 6:
            raise ValueError("Padding is too large for the correct encoding. Please, choose smaller padding values.")

        self.padding_left = symbol_padding_left
        self.padding_right = symbol_padding_right

        # Currently one function is available
        self.encode_function = self.__encode_sequence_ngb

        self.max_no_symbols = len(self.dictionary) ** (symbol_padding_left + symbol_padding_right + 1) + 1

    def __symbol_from_neighbours(self, symbol_padding: list[int]) -> int:
        base_of_power = len(self.dictionary)
        max_power = len(symbol_padding)
        value = 0
        for i in range(max_power):
            value += symbol_padding[i] * base_of_power ** (max_power - i - 1)

        return value

    def __encode_sequence_ngb(self, sequence: str) -> np.ndarray:
        encoded_sequence = []

        for i in range(len(sequence)):
            sequence_fragment = []
            sequence_part = ""
            for j in range(i - self.padding_left, i + self.padding_right + 1):
                if 0 < j < len(sequence) - 1:
                    sequence_fragment.append(self.dictionary[sequence[j]])
                    sequence_part += sequence[j]
                else:
                    cyclic_no = np.mod(j, len(sequence))
                    sequence_fragment.append(self.dictionary[sequence[cyclic_no]])
                    sequence_part += sequence[cyclic_no]

            value = self.__symbol_from_neighbours(sequence_fragment)

            encoded_sequence.append(value)

        return np.array(encoded_sequence)

    def __encode_sequences(self, sequences_to_encode: np.ndarray, encode_function: any = None, max_sequence_length=None
                           ) -> np.ndarray:
        sequences_lengths = []

        if encode_function is None:
            encode_function = self.__encode_sequence_ngb

        if max_sequence_length is None:
            for sequence in sequences_to_encode:
                sequences_lengths.append(len(sequence))

            max_sequence_length = max(sequences_lengths)

        new_sequences = []

        for sequence in sequences_to_encode:
            encoded_sequence = encode_function(sequence[:max_sequence_length])
            new_sequences.append(encoded_sequence)

        return np.array(new_sequences)

    def encode_sequences(self, sequences_to_encode: np.ndarray) -> np.ndarray:
        return self.__encode_sequences(sequences_to_encode, self.encode_function)

    def get_max_no_symbols(self):
        return self.max_no_symbols

    def __str__(self):
        return (
            f"Max number of symbols: {self.max_no_symbols}\n"
            f"padding_left={self.padding_left}\n"
            f"padding_right={self.padding_right}\n"
        )
