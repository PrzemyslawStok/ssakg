import numpy as np


class SubgraphPatterns:
    @staticmethod
    def create_upper_triangular(dim: int, remove_diagonals=False, weighted_edges=True, dtype=np.int16) -> np.array:

        array = np.ones([dim, dim], dtype=dtype)

        if weighted_edges:
            mul = np.arange(1, dim + 1, dtype=dtype)
        else:
            mul = np.ones(dim, dtype=dtype)

        upper_triangular_matrix = (np.tril(array) * mul).transpose()

        if remove_diagonals:
            upper_triangular_matrix = upper_triangular_matrix - np.eye(dim, dtype=dtype) * mul

        return upper_triangular_matrix

    @staticmethod
    def create_square(dim: int, remove_diagonals=False, weighted_edges=False, dtype=np.int16) -> np.array:
        if weighted_edges:
            mul = np.arange(1, dim + 1, dtype=dtype)
        else:
            mul = np.ones(dim, dtype=dtype)

        square_matrix = (np.ones([dim, dim], dtype=dtype) * mul).transpose()

        if remove_diagonals:
            square_matrix = square_matrix - np.eye(dim, dtype=dtype) * mul

        return square_matrix

    @staticmethod
    def set_bit(numbers_to_set: np.ndarray, bit_no: int):
        numbers_to_set = np.bitwise_or(numbers_to_set, 1 << bit_no)
        return numbers_to_set

    @staticmethod
    def bits_upper_triangular(dim: int, dtype=np.int16) -> np.array:
        bits_mul = np.zeros([dim, dim], dtype=dtype)
        for no_bit in range(dim):
            bits_mul[no_bit] = SubgraphPatterns.set_bit(bits_mul[no_bit], no_bit)

        array = np.ones([dim, dim], dtype=dtype)
        upper_triangular_matrix = (np.tril(array) - np.eye(dim, dtype=dtype)).transpose()

        return upper_triangular_matrix * bits_mul
