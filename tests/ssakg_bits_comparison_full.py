import numpy as np
import pandas as pd
from ssakg.ordering_algorithms import WeightedEdgesNodeOrderingAlgorithm

from ssakg import SSAKG, SSAKG_Tester, SequenceGenerator

def table_symbols_sequences_full(symbols_list: list[int], number_of_sequences_list: list[int], context_length,
                            sequence_length=15, unique_elements=False,
                            show_progress=False) -> (pd.DataFrame, str):
    dataframe_columns = [f"{number_of_sequences}" for number_of_sequences in number_of_sequences_list]
    dataframe_index_0 = [f"{value}" for symbols in symbols_list
                         for value in
                         (f"{symbols}", f"{symbols}", f"{symbols}", f"{symbols}", f"{symbols}", f"{symbols}")]

    elements_list = ["unsorted", "sorted", "elapsed", "bits unsorted",
                     "bits sorted", "bits elapsed"]

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

    caption = f"Scene recognition error for various dataset size for sequence length {sequence_length}, context length {context_length}."
    dataframe = pd.DataFrame(data_table, index=index, columns=dataframe_columns)
    formated_table = dataframe.style.format("{:.2g}%").format("{:.2g}s",
                                                              subset=(pd.IndexSlice[:, ["elapsed", "bits elapsed"]],
                                                                      dataframe.columns)).set_caption(caption)

    return dataframe, caption, formated_table

def table_of_sequences_test(unique_elements=False, show_progress=False) -> (pd.DataFrame, str):
    no_symbols_list = [1000, 2000]
    context = 6
    sequence_length = 15
    # number_of_sequences_list = [500, 1000, 1500, 2000, 2500, 3000]
    # number_of_sequences_list = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 11000, 13000, 15000]  # , 20000, 25000]
    number_of_sequences_list = [500]

    table2_dataframe, caption, dataframe_formater = table_symbols_sequences_full(symbols_list=no_symbols_list,
                                                                            number_of_sequences_list=number_of_sequences_list,
                                                                            context_length=context,
                                                                            sequence_length=sequence_length,
                                                                            unique_elements=unique_elements,
                                                                            show_progress=show_progress)

    print(caption)
    print(table2_dataframe)
    return table2_dataframe, caption

if __name__ == '__main__':
    table1, caption1 = table_of_sequences_test(unique_elements=True)