import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, model_selection, pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import preprocessing
import pickle

from graphbase.graphbase import *
from scipy.spatial import distance

_operators = ["hadamard", "average", "l1", "l2", "hamming", "cosine"]


class LinkPrediction(GraphBase):

    def __init__(self):
        GraphBase.__init__(self)

    def split_into_training_test_sets(self, test_set_ratio):
        # Split the graph into two disjoint sets while keeping the training graph connected

        test_set_size = int(test_set_ratio * self.g.number_of_edges())

        # Generate the positive test edges
        test_pos_samples = []
        residual_g = self.g.copy()
        num_of_ccs = nx.number_connected_components(residual_g)
        if num_of_ccs != 1:
            raise ValueError("The graph contains more than one connected component!")

        num_of_pos_test_samples = 0

        edges = list(residual_g.edges())
        perm = np.arange(len(edges))
        np.random.shuffle(perm)
        edges = [edges[inx] for inx in perm]

        for i in range(len(edges)):
            # Remove the chosen edge
            chosen_edge = edges[i]
            residual_g.remove_edge(chosen_edge[0], chosen_edge[1])

            if chosen_edge[1] in nx.connected._plain_bfs(residual_g, chosen_edge[0]):
                num_of_pos_test_samples += 1
                test_pos_samples.append(chosen_edge)
            else:
                residual_g.add_edge(chosen_edge[0], chosen_edge[1])

            if num_of_pos_test_samples == test_set_size:
                break

        if num_of_pos_test_samples != test_set_size:
            raise ValueError("Enough positive edge samples could not be found!")

        # Generate the negative samples
        non_edges = list(nx.non_edges(self.g))
        #perm = np.arange(len(non_edges))
        #np.random.shuffle(perm)
        #non_edges = [non_edges[inx] for inx in perm]
        np.random.shuffle(non_edges)
        #chosen_non_edge_inx = np.random.choice(len(non_edges), size=test_set_size*2, replace=False)

        #train_neg_samples = [non_edges[perm[p]] for p in chosen_non_edge_inx[:test_set_size]]
        #test_neg_samples = [non_edges[perm[p]] for p in chosen_non_edge_inx[test_set_size:]]
        train_neg_samples = non_edges[:test_set_size]
        test_neg_samples = non_edges[test_set_size:test_set_size*2]

        train_pos_samples = list(residual_g.edges())
        return residual_g, train_pos_samples, train_neg_samples, test_pos_samples, test_neg_samples

    def extract_feature_vectors_from_embeddings(self, edges, embeddings, binary_operator):

        features = []
        for i in range(len(edges)):
            edge = edges[i]
            vec1 = np.asarray(embeddings[edge[0]])
            vec2 = np.asarray(embeddings[edge[1]])

            value = 0
            if binary_operator == "hadamard":
                value = [vec1[i]*vec2[i] for i in range(len(vec1))]
            if binary_operator == "average":
                value = 0.5 * (vec1 + vec2)
            if binary_operator == "l1":
                value = abs(vec1 - vec2)
            if binary_operator == "l2":
                value = abs(vec1 - vec2)**2
            if binary_operator == "hamming":
                value = 1.0 - distance.hamming(vec1, vec2)
            if binary_operator == "cosine":
                value = distance.cosine(vec1, vec2)

            features.append(value)


        features = np.asarray(features)
        # Reshape the feature vector if it is 1d vector
        if binary_operator in ["hamming", "cosine"]:
            features = features.reshape(-1, 1)

        return features

    def split_network(self, test_set_ratio, target_folder):

        # Divide the data into training and test sets
        residual_g, train_pos, train_neg, test_pos, test_neg = self.split_into_training_test_sets(test_set_ratio)

        # Prepare the positive and negative samples for training set
        train_samples = train_pos + train_neg
        train_labels = [1 for _ in train_pos] + [0 for _ in train_neg]
        # Prepare the positive and negative samples for test set
        test_samples = test_pos + test_neg
        test_labels = [1 for _ in test_pos] + [0 for _ in test_neg]

        # Check if the target folder exists or not
        if not os.path.isdir(target_folder):
            os.mkdir(target_folder)

        # Save the residual network
        residual_g_save_path = os.path.join(target_folder, self.get_graph_name() + "_residual.edgelist")
        nx.write_edgelist(residual_g, residual_g_save_path, data=False)

        # Save positive and negative samples for training and test sets
        save_file_path = os.path.join(target_folder, self.get_graph_name() + "_samples.pkl")
        with open(save_file_path, 'wb') as f:
            pickle.dump({'training': {'edges':train_samples, 'labels': train_labels },
                         'testing': {'edges':test_samples, 'labels': test_labels}}, f, pickle.HIGHEST_PROTOCOL)

    def read_samples(self, file_path):

        with open(file_path, 'rb') as f:
            temp = pickle.load(f)
            #residual_g = temp['residual_g']
            train_samples, train_labels = temp['training']['edges'], temp['training']['labels']
            test_samples, test_labels = temp['testing']['edges'], temp['testing']['labels']

            return train_samples, train_labels, test_samples, test_labels

    def predict(self, embedding_file_path, train_samples, train_labels, test_samples, test_labels):

        embeddings = {}
        with open(embedding_file_path, 'r') as fin:
            # skip the first line
            num_of_nodes, dim = fin.readline().strip().split()
            # read the embeddings
            for line in fin.readlines():
                tokens = line.strip().split()
                embeddings[tokens[0]] = [float(v) for v in tokens[1:]]

        scores = {op: {'training': [], 'testing': []} for op in _operators}

        for op in _operators:

            train_features = self.extract_feature_vectors_from_embeddings(edges=train_samples,
                                                                          embeddings=embeddings,
                                                                          binary_operator=op)

            test_features = self.extract_feature_vectors_from_embeddings(edges=test_samples,
                                                                         embeddings=embeddings,
                                                                         binary_operator=op)

            clf = LogisticRegression()
            clf.fit(train_features, train_labels)

            train_preds = clf.predict_proba(train_features)[:, 1]
            test_preds = clf.predict_proba(test_features)[:, 1]

            train_roc = roc_auc_score(y_true=train_labels, y_score=train_preds)
            test_roc = roc_auc_score(y_true=test_labels, y_score=test_preds)

            scores[op]['training'].append(train_roc)
            scores[op]['testing'].append(test_roc)

        return scores
