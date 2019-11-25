import os, sys
from classification.multi_label import NodeClassification
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, RawTextHelpFormatter
from link_prediction.link_prediction import *

def parse_arguments():
    parser = ArgumentParser(description="Examples: \n For classification \n  " +
                                        "\tpython run.py classification --graph graph_path.gml " +
                                        "--emb file.embedding --output_file output_path.txt \n" +
                                        "For edge prediction,\n Firstly, prepare the training and test sets.\n" +
                                        "\t python run.py link_prediction split --graph graph_path.gml " +
                                        "--test_set_ratio 0.5 --split_folder folder_path\n" +
                                        "Compute the scores\n" +
                                        "\t python run.py link_prediction predict --emb file.embedding " +
                                        "--sample_file samples_path.pkl --output_file scores.pkl" +
                                        "\t Finally, read the scores:\n" +
                                        "\t Example: python run.py link_prediction read scores.pkl\n",
                            formatter_class=RawTextHelpFormatter)

    subparsers = parser.add_subparsers(help='the name of the evaluation method',
                                       dest="method")

    classification_parser = subparsers.add_parser('classification')
    classification_parser.add_argument('--graph', type=str, required=True,
                                       help='Path of the graph, .gml or .mat files')
    classification_parser.add_argument('--emb', type=str, required=True,
                                       help='Embedding file path')
    classification_parser.add_argument('--training_ratio', type=str, default='large', required=False,
                                       choices=['small', 'large', 'all'],
                                       help='the ratios of the training set')
    classification_parser.add_argument('--output_file', type=str, required=False,
                                       help='The path of the output file')
    classification_parser.add_argument('--file_format', type=str, default="txt", required=False,
                                       help='Output file format', choices=["txt", "npy"])
    classification_parser.add_argument('--num_of_shuffles', type=int, default=10, required=False,
                                       help='The number of shuffles')
    classification_parser.add_argument('--shuffle_std', type=lambda x: (str(x).lower() in ['true','1', 'yes']),
                                       default=False, required=False, help='Compute the standart deviation of shuffles')
    classification_parser.add_argument('--directed', type=lambda x: (str(x).lower() in ['true','1', 'yes']),
                                       default=False, required=False, help='The given graph is directed or undirected')
    classification_parser.add_argument('--detailed', type=lambda x: (str(x).lower() in ['true','1', 'yes']),
                                       default=False, required=False, help='Indicates the format of the output')
    classification_parser.add_argument('--classification_method', type=str, required=False, default="logistic",
                                       help='Path of the graph, .gml or .mat files')

    link_parser = subparsers.add_parser('link_prediction')

    link_subparsers = link_parser.add_subparsers(help='the name of the evaluation method',
                                                 dest="link_option")

    link_split_parser = link_subparsers.add_parser('split')
    link_split_parser.add_argument('--graph', type=str, required=True,
                                   help='Path of the graph, in .gml format')
    link_split_parser.add_argument('--test_set_ratio', type=float, default=0.5, required=True,
                                   help='Testing set ratio between 0 and 1')
    link_split_parser.add_argument('--split_folder', type=str, default=None, required=False,
                                   help='folder path for the residual and pickle files')

    link_predict_parser = link_subparsers.add_parser('predict')
    link_predict_parser.add_argument('--emb', type=str, required=True,
                                     help='Embedding file path')
    link_predict_parser.add_argument('--sample_file', type=str, required=True,
                                     help='pickle samples file path')
    link_predict_parser.add_argument('--output_file', type=str, default=None, required=False,
                                     help='output pickle file path')

    link_predict_parser = link_subparsers.add_parser('read')
    link_predict_parser.add_argument('--input_file', type=str, required=True,
                                     help='The input file in pickle file format')
    link_predict_parser.add_argument('--set', type=str, default='testing', required=False, choices=['training', 'testing'],
                                     help='Show the scores for testing or training set.')

    return parser.parse_args()



def process(args):

    params = {}

    if args.method == "classification":

        training_ratios = []
        if args.training_ratio == 'small':
            training_ratios = np.arange(0.01, 0.1, 0.01).tolist()
        elif args.training_ratio == 'large':
            training_ratios = np.arange(0.1, 1, 0.1).tolist()
        elif args.training_ratio == 'all':
            training_ratios = np.arange(0.01, 0.1, 0.01).tolist() + np.arange(0.1, 1, 0.1).tolist()

        params['directed'] = args.directed

        nc = NodeClassification(embedding_file=args.emb, graph_path=args.graph, params=params, classification_method=args.classification_method)
        nc.evaluate(number_of_shuffles=args.num_of_shuffles, training_ratios=training_ratios)

        if args.output_file is None:
            nc.print_results(detailed=args.detailed, shuffle_std=args.shuffle_std)
        else:
            nc.save_results(output_file=args.output_file, shuffle_std=args.shuffle_std,
                            detailed=args.detailed, file_format=args.file_format)

    elif args.method == "link_prediction":

        lp = LinkPrediction()

        if args.link_option == "split":

            test_set_ratio = args.test_set_ratio
            graph_path = args.graph
            target_folder = args.split_folder
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            lp.read_graph(graph_path=graph_path)
            print("The network has been read!")
            lp.split_network(test_set_ratio=test_set_ratio, target_folder=target_folder)
            print("The network has been partitioned!")

        elif args.link_option == 'predict':

            samples_file_path = args.sample_file
            embedding_file = args.emb
            #graph_path = args.graph

            #ep.read_graph(graph_path=graph_path)
            train_samples, train_labels, test_samples, test_labels = lp.read_samples(samples_file_path)
            scores = lp.predict(embedding_file_path=embedding_file,
                                train_samples=train_samples, train_labels=train_labels,
                                test_samples=test_samples, test_labels=test_labels)

            if args.output_file is not None:
                with open(args.output_file, "wb") as fp:
                    pickle.dump(scores, fp)
            else:
                print(scores)

        elif args.link_option == 'read':

            input_file_path = args.input_file
            pf = pickle.load(open(input_file_path, 'rb'))

            total_scores = pf #pf[0]

            for metric in total_scores:
                print("{}: {}".format(metric, total_scores[metric][args.set][0]))


if __name__ == "__main__":
    args = parse_arguments()

    process(args)
