import os
import sys
import numpy as np
import pickle


# facebook_combined_deepwalk

base_folder = "../sil_kesin/results" # "./scores"
generic_name = "p2p-Gnutella04_gcc_residual" #"p2p-Gnutella08_gcc" #"facebook_combined"
#method_name = "expfam_norm_alpha=005"
method_name = "expfam_norm"

#method_name = "expfam_norm"

subfolder = "" #generic_name
#file_path = os.path.join(base_folder, subfolder, generic_name+"_"+method_name+".pkl" )
file_path = sys.argv[1]

pf = pickle.load(open(file_path, 'rb'))

total_scores = pf #pf[0]

for metric in total_scores:
	print("{}: {}".format(metric, total_scores[metric]['test'][0]))
