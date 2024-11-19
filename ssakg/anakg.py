# Copyright 2024 The SSAKG Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np

import warnings

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plot

from ssakg.anakg_view import GraphView, create_dir
from ssakg.sequence_translator import SequenceTranslator

from tabulate import tabulate


class ANAKG:
    def __init__(self, graph_dim: int = 10, subgraph_dim: int = 5, dtype=np.int16, graphs_to_drawing=False,
                 remove_diagonals=True, weighted_edges=True):

        # The parameter graphs_to_drawing is only for draw colorfully graph to examples.
        # Do not use it for other purposes.

        self.graph_dim = graph_dim
        self.subgraph_dim = subgraph_dim
        self.dtype = dtype

        self.sequence_translator = SequenceTranslator(sequence_matrix_dim=graph_dim, dtype=self.dtype)

        self.subgraph_pattern = self.create_upper_triangular(self.graph_dim, remove_diagonals=remove_diagonals,
                                                                     weighted_edges=weighted_edges, dtype=self.dtype)

        self.graph = self.crate_graph()
        self.sequences = []

        self.graphs_to_drawing = graphs_to_drawing
        self.graph_matrices = []

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

    def set_subgraph_pattern(self, subgraph_pattern: np.array):
        self.subgraph_pattern = subgraph_pattern

    def crate_graph(self):
        dim = self.graph_dim

        return np.zeros([dim, dim], dtype=self.dtype)

    def add_subgraph(self, i: int, j: int):
        subgraph = self.subgraph_pattern.copy()

        temp_graph = np.zeros(self.graph.shape, dtype=self.dtype)
        temp_graph[i:i + self.subgraph_dim, j:j + self.subgraph_dim] = subgraph
        if self.graphs_to_drawing:
            self.graph_matrices.append(temp_graph)

        self.graph = self.graph + temp_graph

    def add_random_subgraph(self):
        rand_min = 0
        rand_max = len(self.graph) - self.subgraph_dim + 1
        i = np.random.randint(rand_min, rand_max)
        j = np.random.randint(rand_min, rand_max)

        self.add_subgraph(i, j)

    def add_dim_to_graph(self, dim_to_add: int):
        self.graph_dim += dim_to_add
        self.graph = np.pad(self.graph, (0, dim_to_add), 'constant')

        for i in range(len(self.graph_matrices)):
            self.graph_matrices[i] = np.pad(self.graph_matrices[i], (0, dim_to_add), 'constant')

    def sequence_to_subgraph(self, sequence: np.ndarray) -> np.ndarray:
        temp_graph = np.zeros(self.graph.shape, dtype=self.dtype)

        for i in range(self.subgraph_dim):
            for j in range(self.subgraph_dim):
                temp_graph[sequence[i], sequence[j]] = self.subgraph_pattern[i, j]

        return temp_graph

    def insert_sequence(self, sequence: np.ndarray) -> np.ndarray | None:
        if len(sequence) != self.subgraph_dim:
            warnings.warn(f"Sequence has incorrect length {len(sequence)}. Subgraph not inserted.")
            return None

        new_sequence, dims_to_add = self.sequence_translator.translate_sequence(sequence)

        if dims_to_add > 0:
            self.add_dim_to_graph(dims_to_add)

        self.sequences.append(new_sequence)
        temp_graph = self.sequence_to_subgraph(new_sequence)

        if self.graphs_to_drawing:
            self.graph_matrices.append(temp_graph)

        self.graph = self.graph + temp_graph

        return new_sequence

    def insert(self, sequences: np.ndarray):
        if np.ndim(sequences) == 1:
            return self.insert_sequence(sequences)
        else:
            for sequence in sequences:
                self.insert_sequence(sequence)

    def get_subgraph_from_sequence(self, sequence: np.ndarray) -> np.ndarray | None:
        if len(sequence) > self.subgraph_dim:
            warn = f"Sequence has incorrect length {len(sequence)}. Subgraph not created."
            warnings.warn(warn)
            return None

        temp_graph = np.zeros([len(sequence), len(sequence)], dtype=self.dtype)

        for i in range(len(sequence)):
            for j in range(len(sequence)):
                temp_graph[i, j] = self.graph[sequence[i], sequence[j]]

        return temp_graph

    def __str__(self):
        graph_size = self.graph.nbytes
        sequences_no = len(self.graph_matrices)
        sequences_size = 0
        for sequence in self.graph_matrices:
            sequences_size += sequence.nbytes

        density = self.density()

        return (f"graph_size: {graph_size} B\ngraph density: {density:.2f}\nstored sequences: "
                f"{sequences_no}\nstored sequences size: {sequences_size}B\n")

    def get_subgraph(self):
        return self.subgraph_pattern

    def get_graph(self):
        return self.graph

    def create_graph_view(self, to_print=False):
        graph_view = GraphView()
        graph_view.graph_from_matrix(np.zeros(self.graph.shape), color="gray", to_print=to_print)

        last_matrix = np.zeros(self.graph.shape, dtype=self.dtype)

        for graph_matrix in self.graph_matrices:
            last_matrix += graph_matrix

        graph_view.graph_from_matrix(last_matrix, add_empty_nodes=False, color="orange", to_print=to_print)

        return graph_view

    def density(self) -> float:
        n = self.graph_dim
        non_zeros = np.nonzero(self.graph)
        m = len(non_zeros[0])

        return m / n / (n - 1)

    def show(self, to_print=False):
        if not self.graphs_to_drawing:
            warnings.warn("Sequences are not stored. Graph view is empty.")
            return

        graph_view = self.create_graph_view(to_print=to_print)
        graph_view.draw(title=None, to_print=to_print)

    def save_fig(self, title: (str, str), output_dir: str = None, output_file: str = None, to_print=True):
        if not self.graphs_to_drawing:
            warnings.warn("Sequences are not stored. Graph view is empty.")

        graph_view = self.create_graph_view(to_print=to_print)
        graph_view.draw(show=False, output_dir=output_dir, output_file=output_file, title=title, to_print=to_print)

    def save_matrix(self, output_dir: str = None, output_file: str = None):
        if not self.graphs_to_drawing:
            warnings.warn("Sequences are not stored. Matrix is empty.")

        if output_file is not None and output_dir is not None:
            create_dir(output_dir)
            file = open(os.path.join(output_dir, output_file), "w")
            file.write(tabulate(self.graph, tablefmt="latex", floatfmt=".0f"))
            file.close()

    def translate_sequence(self, sequence: np.ndarray):
        return self.sequence_translator.translate_sequence(sequence, add_to_dictionary=False)[0]

    def decode_sequence(self, sequence: np.ndarray):
        return self.sequence_translator.decode_sentence(sequence)

    def check_context_correctness(self, context: np.ndarray):
        pass

    def sequence_from_placeholders(self, sequence: list[str]) -> np.ndarray:
        return self.sequence_translator.sequence_from_placeholders(sequence)

    def translate_context(self, context: np.ndarray):
        return self.sequence_translator.translate_sequence(context, add_to_dictionary=False)[0]

    def clear(self):
        self.graph = self.crate_graph()
        self.sequences = []
        self.sequence_translator = SequenceTranslator(sequence_matrix_dim=self.graph_dim, dtype=self.dtype)
        self.graph_matrices = []

    def draw_density(self, title=None, x_label=None, y_label=None, circle_size: int = 2):
        dim = self.graph_dim

        graph_1d = self.graph.reshape(dim * dim)

        non_zeros = np.nonzero(graph_1d)[0]
        x = np.mod(non_zeros, dim)
        y = np.floor_divide(non_zeros, dim)
        non_zeros_values = graph_1d[non_zeros]

        dataframe = pd.DataFrame({"x": x, "y": y, "z": non_zeros_values})

        palette = sns.color_palette("dark:#5A9_r", as_cmap=True)
        joint_grid = sns.JointGrid(data=dataframe, x="x", y="y", hue="z", palette=palette)
        joint_grid.plot_joint(sns.scatterplot, s=circle_size)
        joint_grid.plot_marginals(sns.histplot, kde=False, bins=100)
        joint_grid.ax_joint.legend().remove()

        if title is not None:
            joint_grid.fig.suptitle("ssakg - local density")

        if x_label is not None:
            joint_grid.ax_joint.set_xlabel(x_label)
        else:
            joint_grid.ax_joint.set_xlabel("")

        if y_label is not None:
            joint_grid.ax_joint.set_ylabel(y_label)
        else:
            joint_grid.ax_joint.set_ylabel("")

        joint_grid.fig.tight_layout()

        plot.show()