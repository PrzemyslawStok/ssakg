import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from ssakg.ordering_algorithms import WeightedEdgesNodeOrderingAlgorithm

from ssakg import SSAKG, SSAKG_Tester, SequenceGenerator


def table_symbols_sequences(symbols_list: list[int], number_of_sequences_list: list[int], context_length,
                            sequence_length=15, unique_elements=False,
                            show_progress=False) -> (pd.DataFrame, str):
    dataframe_columns = [f"{number_of_sequences}" for number_of_sequences in number_of_sequences_list]
    dataframe_index_0 = [f"{value}" for symbols in symbols_list
                         for value in
                         (f"{symbols}", f"{symbols}", f"{symbols}", f"{symbols}", f"{symbols}", f"{symbols}")]

    dataframe_formater = {"unsorted": "{:.2%}", "sorted": "{:.2%}", "elapsed": ".2s", "bits unsorted": "{:.2%}",
                          "bits sorted": "{:.2%}", "bits elapsed": ".2s"}

    elements_list = dataframe_formater.keys()

    dataframe_index_1 = [f"{value}" for _ in symbols_list
                         for value in elements_list]

    index = pd.MultiIndex.from_arrays([dataframe_index_0, dataframe_index_1], names=["no symbols", "algorithm"])

    row_height = len(elements_list)
    data_table = np.zeros((row_height * len(symbols_list), len(number_of_sequences_list)), dtype=float)

    for i, symbols in enumerate(symbols_list):
        for j, no_sequences in enumerate(number_of_sequences_list):
            ssakg = SSAKG(number_of_symbols=symbols, sequence_length=sequence_length)
            ssakg_bits = SSAKG(number_of_symbols=symbols, sequence_length=sequence_length, bit_based=True)

            sequence_generator = SequenceGenerator(sequence_length=sequence_length, sequence_min=0,
                                                   sequence_max=symbols)
            sequences = sequence_generator.generate_unique_sequences(no_sequences, unique_elements=unique_elements)

            ssakg.insert(sequences)
            ssakg_bits.insert(sequences)

            ssakg_tester = SSAKG_Tester(ssakg, sequences, algorithms_list=[WeightedEdgesNodeOrderingAlgorithm()])
            ssakg_tester_bits = SSAKG_Tester(ssakg_bits, sequences)

            ssakg_tester.make_test(context_length=context_length,
                                   show_progress=show_progress)

            ssakg_tester_bits.make_test(context_length=context_length,
                                        show_progress=show_progress)

            unsorted_percentage = ssakg_tester.get_unsorted_percentage()
            sorted_percentage = ssakg_tester.get_sorted_percentage()
            elapsed_time = ssakg_tester.get_test_time()
            # print(f"elapsed time: {elapsed_time}")
            sorted_percentage_bits = ssakg_tester_bits.get_sorted_percentage()
            unsorted_percentage_bits = ssakg_tester_bits.get_unsorted_percentage()
            elapsed_time_bits = ssakg_tester_bits.get_test_time()
            # print(f"bits elapsed time: {elapsed_time_bits}")

            data_table[row_height * i, j] = 100 - unsorted_percentage
            data_table[row_height * i + 1, j] = 100 - sorted_percentage
            data_table[row_height * i + 2, j] = elapsed_time
            data_table[row_height * i + 3, j] = 100 - sorted_percentage_bits
            data_table[row_height * i + 4, j] = 100 - unsorted_percentage_bits
            data_table[row_height * i + 5, j] = elapsed_time_bits

    caption = f"Scene recognition error in % for various dataset size for sequence length {sequence_length}, context length {context_length}."
    dataframe = pd.DataFrame(data_table, index=index, columns=dataframe_columns)

    return dataframe, caption, dataframe_formater


def table_various_context(no_symbols: int, number_of_sequences_list: list[int], context_list: list[int],
                          sequence_length=15, unique_elements=False,
                          show_progress=False) -> (pd.DataFrame, str):
    dataframe_columns = [f"{length}" for length in context_list]
    dataframe_index_0 = [f"{value}" for sequences_no in number_of_sequences_list
                         for value in (f"{sequences_no}", f"{sequences_no}")]
    dataframe_index_1 = [f"{value}" for sequences_no in number_of_sequences_list
                         for value in (f"standard", f"bits")]

    index = pd.MultiIndex.from_arrays([dataframe_index_0, dataframe_index_1], names=["stored sequences", "algorithm"])

    data_table = np.zeros((2 * len(number_of_sequences_list), len(context_list)), dtype=float)

    for i, no_sequences in enumerate(number_of_sequences_list):
        ssakg = SSAKG(number_of_symbols=no_symbols, sequence_length=sequence_length)
        ssakg_bits = SSAKG(number_of_symbols=no_symbols, sequence_length=sequence_length, bit_based=True)

        sequence_generator = SequenceGenerator(sequence_length=sequence_length, sequence_min=0,
                                               sequence_max=no_symbols)
        sequences = sequence_generator.generate_unique_sequences(no_sequences, unique_elements=unique_elements)

        ssakg.insert(sequences)
        ssakg_bits.insert(sequences)

        ssakg_tester = SSAKG_Tester(ssakg, sequences)
        ssakg_tester_bits = SSAKG_Tester(ssakg_bits, sequences)

        for j, length in enumerate(context_list):
            data_table[2 * i, j] = 10
            data_table[2 * i + 1, j] = 20

        for j, length in enumerate(context_list):
            data_table[2 * i, j] = 100 - ssakg_tester.make_test(context_length=length,
                                                                show_progress=show_progress)
            data_table[2 * i + 1, j] = 100 - ssakg_tester_bits.make_test(context_length=length,
                                                                         show_progress=show_progress)

    caption = f"Scene recognition error in \\% for various dataset and context sizes for sequence length {sequence_length}, number of symbols {no_symbols}."
    return pd.DataFrame(data_table, index=index, columns=dataframe_columns), caption


def table_of_sequences_test(unique_elements=False, show_progress=False) -> (pd.DataFrame, str):
    no_symbols_list = [1000, 2000]
    context = 6
    sequence_length = 15
    # number_of_sequences_list = [500, 1000, 1500, 2000, 2500, 3000]
    # number_of_sequences_list = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 11000, 13000, 15000]  # , 20000, 25000]
    number_of_sequences_list = [500]

    table2_dataframe, caption, dataframe_formater = table_symbols_sequences(symbols_list=no_symbols_list,
                                                                            number_of_sequences_list=number_of_sequences_list,
                                                                            context_length=context,
                                                                            sequence_length=sequence_length,
                                                                            unique_elements=unique_elements,
                                                                            show_progress=show_progress)

    print(caption)
    print(table2_dataframe)
    return table2_dataframe, caption


def table_of_context_test(unique_elements=False):
    no_symbols = 2000
    context_list = [3, 4, 5, 6, 7]
    sequence_length = 15

    number_of_sequences_list = [1000, 2000, 3000, 4000, 5000]
    # number_of_sequences_list = [1000, 2000]

    table2_dataframe, caption = table_various_context(no_symbols=no_symbols,
                                                      number_of_sequences_list=number_of_sequences_list,
                                                      context_list=context_list,
                                                      sequence_length=sequence_length, unique_elements=unique_elements,
                                                      show_progress=True)

    print(table2_dataframe)

    return table2_dataframe, caption


if __name__ == '__main__':
    table1, caption1 = table_of_sequences_test(unique_elements=True)

    # table2, caption2 = table_of_context_test(unique_elements=True)
