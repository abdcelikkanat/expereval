import os
import sys
import numpy as np
import pickle


# facebook_combined_deepwalk

base_folder = "../edge_prediction" # "./scores"
generic_name = "facebook_combined_gcc" #"p2p-Gnutella08_gcc" #"facebook_combined"
method_name = "deepwalk"

subfolder = "" #generic_name
file_path = os.path.join(base_folder, subfolder, generic_name+"_"+method_name+".scores" )


pf = pickle.load(open(file_path, 'rb'))

total_scores = pf #pf[0]

for metric in total_scores:
	print("{}: {}".format(metric, total_scores[metric]))
