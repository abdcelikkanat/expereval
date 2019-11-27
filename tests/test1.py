import os
from link_prediction.link_prediction import *
import unittest


class ClassificationTest(unittest.TestCase):

    def test_temp(self):
        pass


class LinkPredictionTest(unittest.TestCase):

    graph_name = "karate"
    target_folder = "temp"
    graph_path = "./{}.gml".format(graph_name)
    embedding_file = os.path.join(target_folder, "./{}_residual.embedding".format(graph_name))
    samples_file_path = os.path.join(target_folder, "{}_samples.pkl".format(graph_name))
    scores_file = os.path.join(target_folder, "{}_scores.pkl".format(graph_name))
    set = 'testing'
    test_set_ratio = 0.5

    lp = LinkPrediction()

    def test_split(self):

        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)

        self.lp.read_graph(graph_path=self.graph_path)
        print("The network has been read!")
        self.lp.split_network(test_set_ratio=self.test_set_ratio, target_folder=self.target_folder)
        print("The network has been partitioned!")

    def test_predict(self):

        samples_file_path = self.samples_file_path
        embedding_file = self.embedding_file

        train_samples, train_labels, test_samples, test_labels = self.lp.read_samples(samples_file_path)
        scores = self.lp.predict(embedding_file_path=embedding_file,
                                 train_samples=train_samples, train_labels=train_labels,
                                 test_samples=test_samples, test_labels=test_labels)

        with open(self.scores_file, "wb") as fp:
            pickle.dump(scores, fp)

    def test_read(self):

        pf = pickle.load(open(self.scores_file, 'rb'))
        total_scores = pf

        ''''''
        for metric in total_scores:
            print(total_scores[metric])
            #print("{}: {}".format(metric, total_scores[metric][self.set][0]))


if __name__ == '__main__':
    unittest.main()