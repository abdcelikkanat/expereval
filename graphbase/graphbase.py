import os
import networkx as nx
import matplotlib.pyplot as plt
from os.path import basename, splitext


class GraphBase:
    def __init__(self):
        self.g = None
        self.number_of_groundtruth_communities = None
        self.graph_name = "unknown_graphname"

    def read_graph(self, graph_path, extension=None):

        if extension is None:
            ext = os.path.splitext(graph_path)[-1].lower()
        else:
            ext = extension

        if ext == ".gml":
            self.g = nx.read_gml(graph_path)

        elif ext == ".edgelist":
            self.g = nx.read_edgelist(graph_path)

        else:
            raise ValueError("Invalid graph file format!")

        self.set_graph_name(graph_name=splitext(basename(graph_path))[0])

    def set_graph_name(self, graph_name):
        self.graph_name = graph_name

    def get_graph_name(self):
        return self.graph_name

    def set_graph(self, nxg):
        self.g = nxg

    def get_graph(self):
        return self.g

    def set_number_of_groundtruth_communities(self, value):
        self.number_of_groundtruth_communities = value

    def get_number_of_groundtruth_communities(self):

        if self.number_of_groundtruth_communities is None:
            return self.number_of_groundtruth_communities

        # It is assumed that the labels of clusters starts from 0 to K-1
        max_community_label = -1
        communities = nx.get_node_attributes(self.g, "community")
        for node in self.g.nodes():
            c_max = max(list(communities[node]))
            if c_max > max_community_label:
                max_community_label = c_max

        return max_community_label + 1

    def plot_graph(self, node2comm):

        k = len(set(node2comm.values()))

        colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w']

        nodelist = [[] for _ in range(k)]
        for node in node2comm:
            nodelist[int(node2comm[node])].append(node)

        pos = nx.spring_layout(self.g)

        for i in range(k):
            nx.draw_networkx_nodes(self.g, pos,
                                   nodelist=nodelist[i],
                                   node_color=colors[i],
                                   node_size=100,
                                   alpha=0.8)

        nx.draw_networkx_edges(self.g, pos, width=1.0, alpha=0.5)

        plt.show()
