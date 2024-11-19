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

import networkx as nx
import matplotlib.pyplot as plot
import numpy as np

import os


def create_dir(dir_name: str):
    if dir_name is None or dir_name == "":
        return

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


class GraphView:
    def __init__(self):
        self.sub_graphs = []

        self.graph_to_draw = nx.MultiDiGraph()

    def draw(self, title: (str, str) = None, show=True, output_dir: str = None, output_file: str = None,
             to_print=False):
        plot.close()

        if to_print:
            plot.figure(figsize=(10, 10), dpi=300)
        else:
            plot.figure(figsize=(5, 5), dpi=100)

        graph = self.graph_to_draw.copy()

        for (subgraph, color) in self.sub_graphs:
            graph = nx.compose(graph, subgraph)

        positions = nx.circular_layout(graph)

        for (subgraph, color) in self.sub_graphs:
            sg = graph.subgraph(subgraph.nodes())

            if to_print:
                nx.draw_networkx(sg, node_size=1000, font_size=20, with_labels=True,
                                 pos=positions, node_color=color, width=5.5, connectionstyle="arc3, rad = 0.1")
            else:
                nx.draw_networkx(sg, with_labels=True,
                                 pos=positions, node_color=color, connectionstyle="arc3, rad = 0.1")

            if title is not None:
                if title[0] is not None:
                    plot.title(title[0], fontsize=20)
                if title[1] is not None:
                    plot.xlabel(title[1], fontsize=20)

        if show:
            plot.show()

        if output_file is not None and output_dir is not None:
            create_dir(output_dir)
            plot.box(False)
            plot.savefig(os.path.join(output_dir, output_file))

    def graph_from_matrix(self, matrix: np.array, add_empty_nodes=True, color: str = "yellow", to_print=True):
        graph = self.graph_to_draw.copy()
        it = np.nditer(matrix, flags=["multi_index"])
        if to_print:
            renumber_nodes = 1
        else:
            renumber_nodes = 0

        for edges in it:
            if edges > 0:
                for i in range(edges):
                    graph.add_edge(it.multi_index[0] + renumber_nodes, it.multi_index[1] + renumber_nodes)
            else:
                if add_empty_nodes:
                    graph.add_node(it.multi_index[0] + renumber_nodes)

        self.sub_graphs.append((graph, color))

    def graph_from_edges(self, edges: list[list[int]], color: str = "yellow"):
        graph = self.graph_to_draw.copy()

        for edge in edges:
            graph.add_edge(edge[0], edge[1])

        self.sub_graphs.append((graph, color))

    def clear(self):
        self.sub_graphs = []
