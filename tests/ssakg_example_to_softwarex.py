from ssakg.ssakg import SSAKG
from ssakg.utils.sequence_generator import SequenceGenerator
from ssakg.utils.ssakg_tester import SSAKG_Tester

if __name__ == '__main__':
    number_of_symbols = 2000
    sequence_length = 15
    number_of_sequences = 5000

    bit_based = True

    ssakg = SSAKG(number_of_symbols=number_of_symbols, sequence_length=sequence_length, bit_based=bit_based)

    sequence_generator = SequenceGenerator(sequence_length=sequence_length, sequence_min=0, sequence_max=number_of_symbols)
    sequences = sequence_generator.generate_unique_sequences(number_of_sequences, unique_elements=False)

    ssakg.insert(sequences)

    ssakg_tester = SSAKG_Tester(ssakg, sequences)
    ssakg_tester.make_test(context_length=5, show_progress=True)
    print(ssakg_tester)