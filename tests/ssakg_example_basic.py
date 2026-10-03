import numpy as np

from ssakg import SSAKG


if __name__ == "__main__":
    bit_based = False

    images_output_dir = "../images"

    ssakg = SSAKG(number_of_symbols=10, sequence_length=3, graphs_to_drawing=True)

    ssakg.insert_sequence(np.array([1, 2, 3]))

    # ssakg.show()
    ssakg.save_fig(title=None, output_dir=images_output_dir, output_file="softwarex_multi_graph_0.png")

    ssakg.insert_sequence(np.array([3, 5, 5]))

    # ssakg.show()
    ssakg.save_fig(title=None, output_dir=images_output_dir, output_file="softwarex_multi_graph_1.png")