import numpy as np
from ssakg.ordering_algorithms import WeightedEdgesNodeOrderingAlgorithm

from ssakg import SSAKG, SequenceGenerator, SSAKG_Tester

def create_ssakg_test(bit_based=False, number_of_symbols=1000, number_of_sequences=1000, sequence_length=15,
                      context_length=7):
    ssakg = SSAKG(number_of_symbols=number_of_symbols, sequence_length=sequence_length, bit_based=bit_based)
    sequence_generator = SequenceGenerator(sequence_length=sequence_length, sequence_min=0,
                                           sequence_max=number_of_symbols)

    sequences = sequence_generator.generate_unique_sequences(number_of_sequences, unique_elements=False)
    ssakg.insert(sequences)

    ssakg_tester = SSAKG_Tester(ssakg, sequences, algorithms_list=[WeightedEdgesNodeOrderingAlgorithm()])
    ssakg_tester.make_test(context_length=context_length, show_progress=True)

    ssakg_tester.plot_agreement_histogram(draw_text=True)
    print(ssakg_tester)
    print(ssakg)


def simple_read(bit_based: bool):
    ssakg = SSAKG(number_of_symbols=20, sequence_length=5, graphs_to_drawing=False, bit_based=bit_based)
    ssakg.insert(np.array([[5, 1, 3, 4, 7], [1, 2, 11, 8, 5], [5, 1, 11, 2, 15]]))

    read_sequence = ssakg.get_sequence([3, 1, 5])
    print(read_sequence)


if __name__ == "__main__":
    SPEED_TEST = True
    SIMPLE_READ_TEST = False

    BIT_BASED = True
    if SPEED_TEST:
        create_ssakg_test(bit_based=BIT_BASED, number_of_symbols=1000, number_of_sequences=5_000, sequence_length=15,
                          context_length=6)
    if SIMPLE_READ_TEST:
        simple_read(bit_based=BIT_BASED)
