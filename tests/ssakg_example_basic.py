import numpy as np

from ssakg import SSAKG


if __name__ == "__main__":
    BIT_BASED = False

    images_output_dir = "../images"

    ssakg = SSAKG(number_of_symbols=15, sequence_length=3, graphs_to_drawing=True,bit_based=BIT_BASED)

    ssakg.insert_sequence(np.array([1, 2, 3]))

    # ssakg.show()
    ssakg.save_fig(title=None, output_dir=images_output_dir, output_file="multi_graph_0.png")

    ssakg.insert_sequence(np.array([3, 5, 5]))

    # ssakg.show()
    ssakg.save_fig(title=None, output_dir=images_output_dir, output_file="multi_graph_1.png")