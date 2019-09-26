import numpy as np
from collections import OrderedDict
from evaluation.evaluation import Evaluation
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
from sklearn.svm import SVC
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist


class NodeClassification(Evaluation):
    """
    Multi-label node classification
    """

    _score_types = ['micro', 'macro']

    def __init__(self, embedding_file, graph_path, params={}, classification_method=None):
        Evaluation.__init__(self)

        self._embedding_file = embedding_file
        self._graph_path = graph_path
        self._directed = params['directed'] if 'directed' in params else False

        self.results = None

        self.classification_method = classification_method

    def evaluate(self, number_of_shuffles, training_ratios):
        g = self._get_networkx_graph(self._graph_path, directed=self._directed, params={})
        node2embedding = self.read_embedding_file(embedding_file_path=self._embedding_file)
        node2community = self.get_node2community(g)

        N = g.number_of_nodes()
        K = self.detect_number_of_communities(g)

        nodelist = [node for node in g.nodes()]
        x = np.asarray([node2embedding[node] for node in nodelist])
        label_matrix = [[1 if k in node2community[node] else 0 for k in range(K)] for node in nodelist]
        label_matrix = csr_matrix(label_matrix)

        results = {}

        for score_t in self._score_types:
            results[score_t] = OrderedDict()
            for ratio in training_ratios:
                results[score_t].update({ratio: []})

        for train_ratio in training_ratios:

            for _ in range(number_of_shuffles):
                # Shuffle the data
                shuffled_features, shuffled_labels = shuffle(x, label_matrix)

                # Get the training size
                train_size = int(train_ratio * N)
                # Divide the data into the training and test sets
                train_features = shuffled_features[0:train_size, :]
                train_labels = shuffled_labels[0:train_size]

                test_features = shuffled_features[train_size:, :]
                test_labels = shuffled_labels[train_size:]


                # Train the classifier
                if self.classification_method == "logistic":
                    ovr = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
                elif self.classification_method == "svm-precomputed":
                    ovr = OneVsRestClassifier(SVC(kernel="rbf", cache_size=4096, probability=True))

                elif self.classification_method == "svm-hamming":
                    ovr = OneVsRestClassifier(SVC(kernel="precomputed", cache_size=4096, probability=True))
                    _train_features = train_features.copy()
                    _test_features = test_features.copy()

                    train_features = cdist(_train_features, _train_features, 'hamming')
                    test_features = cdist(_test_features, _train_features, 'hamming')

                else:
                    raise ValueError("Invalid classification method name: {}".format(self.classification_method))


                ovr.fit(train_features, train_labels)

                # Find the predictions, each node can have multiple labels
                test_prob = np.asarray(ovr.predict_proba(test_features))
                y_pred = []
                for i in range(test_labels.shape[0]):
                    k = test_labels[i].getnnz()  # The number of labels to be predicted
                    pred = test_prob[i, :].argsort()[-k:]
                    y_pred.append(pred)

                # Find the true labels
                y_true = [[] for _ in range(test_labels.shape[0])]
                co = test_labels.tocoo()
                for i, j in zip(co.row, co.col):
                    y_true[i].append(j)

                mlb = MultiLabelBinarizer(range(K))
                for score_t in self._score_types:
                    score = f1_score(y_true=mlb.fit_transform(y_true),
                                     y_pred=mlb.fit_transform(y_pred),
                                     average=score_t)

                    results[score_t][train_ratio].append(score)

        self.results = results

    def get_output_text(self, shuffle_std=False, detailed=False):
      
        num_of_shuffles = len(list(list(self.results.values())[0].values())[0])
        train_ratios = [r for r in list(self.results.values())[0]]
        percentage_title = " ".join("{0:.0f}%".format(100*r) for r in list(self.results.values())[0])

        output = ""
        for score_type in self._score_types:
            if detailed is True:
                for shuffle_num in range(1, num_of_shuffles+1):
                    output += "{} score, shuffle #{}\n".format(score_type, shuffle_num)
                    output += percentage_title + "\n"
                    for ratio in train_ratios:
                        output += "{0:.5f} ".format(self.results[score_type][ratio][shuffle_num-1])
                    output += "\n"

            output += "{} score, mean of {} shuffles\n".format(score_type, num_of_shuffles)
            output += percentage_title + "\n"
            for ratio in train_ratios:
                output += "{0:.5f} ".format(np.mean(self.results[score_type][ratio]))
            output += "\n"

            if shuffle_std is True:
                output += "{} score, std of {} shuffles\n".format(score_type, num_of_shuffles)
                output += percentage_title + "\n"
                for ratio in train_ratios:
                    output += "{0:.5f} ".format(np.std(self.results[score_type][ratio]))
                output += "\n"

        return output

    def print_results(self, shuffle_std, detailed=False):
        output = self.get_output_text(shuffle_std=shuffle_std, detailed=detailed)
        print(output)

    def save_results(self, output_file, shuffle_std, detailed=False, file_format="txt"):

        with open(output_file, 'w') as f:
            if file_format == "txt":
                output = self.get_output_text(shuffle_std=shuffle_std, detailed=detailed)
                f.write(output)
            if file_format == "npy":
                np.save(f, self.results)




