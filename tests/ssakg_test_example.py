import numpy as np

from ssakg import SSAKG, SequenceGenerator, SSAKG_Tester
from tests.SSAKG_Tester_context import SSAKG_Tester_context


def create_ssakg_test(number_of_symbols=1000, number_of_sequences=1000, sequence_length=15, context_length=7):
    ssakg = SSAKG(number_of_symbols=number_of_symbols, sequence_length=sequence_length)
    sequence_generator = SequenceGenerator(sequence_length=sequence_length, sequence_min=0,
                                           sequence_max=number_of_symbols)

    sequences = sequence_generator.generate_unique_sequences(number_of_sequences, unique_elements=False)
    ssakg.insert(sequences)

    ssakg_tester = SSAKG_Tester(ssakg, sequences)
    ssakg_tester.make_test(context_length=context_length, show_progress=True)

    ssakg_tester.plot_agreement_histogram(draw_text=True)
    print(ssakg_tester)
    print(ssakg)


def crate_ssakg_similarity_test(number_of_symbols=1000, number_of_sequences=1000, sequence_length=15, context_length=7):
    mayor_length = 10
    minor_length = sequence_length - mayor_length

    ssakg = SSAKG(number_of_symbols=number_of_symbols, sequence_length=sequence_length)

    sequence_generator_mayor = SequenceGenerator(sequence_length=mayor_length, sequence_min=0,
                                                 sequence_max=number_of_symbols)

    sequence_generator_minor = SequenceGenerator(sequence_length=minor_length, sequence_min=0,
                                                 sequence_max=number_of_symbols)

    mayor_sequences = sequence_generator_mayor.generate_unique_sequences(number_of_sequences, unique_elements=True)
    minor_sequences = sequence_generator_minor.generate_unique_sequences(number_of_sequences, unique_elements=True)

    minor_sequences = np.tile(minor_sequences[0], (number_of_sequences, 1))

    sequences = np.hstack((mayor_sequences, minor_sequences))
    ssakg.insert(sequences)

    ssakg_tester = SSAKG_Tester_context(ssakg, sequences, context_part_length=mayor_length)
    ssakg_tester.make_test(context_length=context_length, show_progress=True)

    print(ssakg_tester)
    print(ssakg)

    read_sequence = ssakg.get_sequence(sequences[0])
    # print(sequences[0])
    # print(read_sequence)

def simple_read():
    ssakg = SSAKG(number_of_symbols=20, sequence_length=5, graphs_to_drawing=True)
    ssakg.insert(np.array([[5, 1, 3, 4, 7],[1, 2, 11, 8, 5],[5, 1, 11, 2, 15]]))

    read_sequence = ssakg.get_sequence([3, 1, 5])
    print(read_sequence)

if __name__ == "__main__":
    speed_test = False
    similarity_test = False
    simple_read_test = True
    if speed_test:
        create_ssakg_test(number_of_symbols=1000, number_of_sequences=100, sequence_length=15, context_length=6)
    if similarity_test:
        crate_ssakg_similarity_test(number_of_symbols=1000, number_of_sequences=100, sequence_length=15,
                                    context_length=6)
    if simple_read_test:
        simple_read()
