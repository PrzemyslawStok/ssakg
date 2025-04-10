from ssakg import SSAKG, SequenceGenerator, SSAKG_Tester


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

if __name__ == "__main__":
    create_ssakg_test(number_of_symbols=1000, number_of_sequences=1000, sequence_length=15, context_length=6)