import numpy as np


class SequenceGenerator:
    def __init__(self, sequence_length, sequence_min: int, sequence_max: int, dtype=np.uint16, seed=None):
        self.dtype = dtype

        self.sequence_length = sequence_length
        self.sequence_min = sequence_min
        self.sequence_max = sequence_max

        self.unique_sequences = np.empty((0, self.sequence_length), dtype=self.dtype)
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed=seed)

    def __generate_new_sequences(self, number_sequences: int = 1, unique_elements=False) -> np.ndarray:
        rng = self.rng

        if unique_elements:
            sequences = np.zeros((number_sequences, self.sequence_length), dtype=self.dtype)
            for i in range(number_sequences):
                sequences[i] = rng.choice(self.sequence_max - self.sequence_min,
                                          size=self.sequence_length, replace=False) + self.sequence_min
        else:
            sequences = rng.integers(self.sequence_min, self.sequence_max,
                                     size=(number_sequences, self.sequence_length), dtype=self.dtype)
        return sequences

    def generate_unique_sequences(self, number_sequences=1, unique_elements=False) -> np.ndarray:
        number_to_add = number_sequences
        number_added = 0
        added_sequences = np.empty((0, self.sequence_length), dtype=self.dtype)

        while number_added < number_sequences:
            sequences = self.__generate_new_sequences(number_sequences=number_to_add, unique_elements=unique_elements)
            unique_sequences = np.unique(sequences, axis=0)
            added_sequences = np.unique(np.append(added_sequences, unique_sequences, axis=0), axis=0)
            number_added = len(added_sequences)
            number_to_add = number_sequences - number_added

        if unique_elements:
            self.unique_sequences = np.append(self.unique_sequences, added_sequences, axis=0)

        return added_sequences

    @staticmethod
    def intersect_2d(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
        n_cols = array_1.shape[1]
        dtype = {'names': ['f{}'.format(i) for i in range(n_cols)],
                 'formats': n_cols * [array_1.dtype]}

        intersection = np.intersect1d(array_1.view(dtype), array_2.view(dtype))
        intersection = intersection.view(array_1.dtype).reshape(-1, n_cols)

        return intersection

    @staticmethod
    def setdiff_2d(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
        n_cols = array_1.shape[1]
        dtype = {'names': ['f{}'.format(i) for i in range(n_cols)],
                 'formats': n_cols * [array_1.dtype]}

        difference = np.setdiff1d(array_1.view(dtype), array_2.view(dtype))
        difference = difference.view(array_1.dtype).reshape(-1, n_cols)

        return difference

    def add_unique_sequences(self, number_sequences=1) -> (np.ndarray, np.ndarray):
        number_to_add = number_sequences
        number_added = 0

        init_unique_sequences = len(self.unique_sequences)
        all_sequences = np.copy(self.unique_sequences)

        unique_added = np.empty((0, self.sequence_length), dtype=self.dtype)

        while number_added < number_sequences:
            sequences = self.generate_unique_sequences(number_sequences=number_to_add, unique_elements=True)

            intersection = self.intersect_2d(all_sequences, sequences)

            unique_to_add = self.setdiff_2d(sequences, intersection)
            unique_added = np.append(unique_added, unique_to_add, axis=0)

            unique_sequences, indexes = np.unique(np.append(all_sequences, sequences, axis=0), axis=0,
                                                  return_index=True)

            all_sequences = unique_sequences.copy()

            number_added = len(all_sequences) - init_unique_sequences
            number_to_add = number_sequences - number_added

        self.unique_sequences = all_sequences

        return unique_added, self.unique_sequences
