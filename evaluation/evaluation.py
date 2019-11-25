import scipy.io as sio
from scipy.sparse import csr_matrix
import networkx as nx
import numpy as np
import os


class Evaluation:
    def __init__(self):
        pass

    def read_embedding_file(self, embedding_file_path):
        # Read the the embeddings

        node2embedding = {}

        with open(embedding_file_path, 'r') as f:
            # Read the first line
            N, dim = (int(v) for v in f.readline().strip().split())
            # Read embeddings
            for line in f.readlines():
                tokens = line.strip().split()
                if self.classification_method == "svm":
                    node2embedding[tokens[0]] = [int(value) for value in tokens[1:]]
                    print(node2embedding[tokens[0]])
                else:
                    node2embedding[tokens[0]] = [float(value) for value in tokens[1:]]

        return node2embedding

    def get_node2community(self, nxg):

        node2community = nx.get_node_attributes(nxg, name='community')

        for node in nxg.nodes():

            comm = node2community[node]
            if type(comm) == int:
                node2community[node] = [comm]

        return node2community

    def detect_number_of_communities(self, nxg):

        # It is assumed that the labels of communities starts from 0 to K-1
        max_community_label = -1
        communities = nx.get_node_attributes(nxg, "community")
        for node in nxg.nodes():
            comm_list = communities[node]
            if type(comm_list) is int:
                comm_list = [comm_list]

            c_max = max(comm_list)
            if c_max > max_community_label:
                max_community_label = c_max
        return max_community_label + 1

    def _get_file_extension(self, file_path):
        ext = os.path.splitext(file_path)[1][1:].strip().lower()
        return ext

    def _get_networkx_graph(self, graph_path, directed=True, params={}):
        ext = self._get_file_extension(graph_path)
        if ext == 'gml':

            return nx.read_gml(graph_path)

        elif ext == 'mat':

            # Set mat file parameters
            mat_network_name = params['network_name'] if 'network_name' in params else "network"
            mat_cluster_name = params['group_name'] if 'group_name' in params else "group"

            # Read the mat file
            mat_dict = sio.loadmat(graph_path)

            adj_matrix = csr_matrix(mat_dict[mat_network_name])
            community_matrix = csr_matrix(mat_dict[mat_cluster_name])

            N = adj_matrix.shape[1]
            K = community_matrix.shape[1]
            E = adj_matrix.count_nonzero()

            # Print graphs statistics
            #print("Number of nodes: {}".format(N))
            #print("Number of edges: {}".format(E))
            #print("Number of clusters: {}".format(K))

            g = nx.DiGraph()

            cx = adj_matrix.tocoo()
            for i, j, val in zip(cx.row, cx.col, cx.data):
                if val > 0:
                    g.add_edge(str(i), str(j))

            assert g.number_of_nodes() == N, "There is an error, the number of nodes mismatched, {} != {}".format(N,
                                                                                                                  g.number_of_nodes())
            assert g.number_of_edges() == E, "There is an error, the number of edges mismatched, {} != {}".format(E,
                                                                                                                  g.number_of_edges())
            # Convert it into an undirected graph
            if directed is False:
                g = g.to_undirected()

            # Read communities
            values = {node: [] for node in g.nodes()}

            cx = community_matrix.tocoo()
            for i, k, val in zip(cx.row, cx.col, cx.data):
                if val > 0:
                    values[str(i)].append(str(k))

            nx.set_node_attributes(g, name="community", values=values)

            return g

        else:

            raise ValueError("Unknown file type!")

