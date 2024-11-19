import numpy as np

from ssakg import SSAKG

if __name__ == "__main__":
    ssakg = SSAKG(number_of_symbols=10, sequence_length=3, graphs_to_drawing=True)
    ssakg.show()
    ssakg.insert_sequence(np.array([5, 2, 3]))
    ssakg.show()
    ssakg.insert_sequence(np.array([1, 2, 3]))
    ssakg.show()
    ssakg.insert(np.array([[1, 2, 3], [2, 4, 5]]))
    ssakg.show()