### Experimental Evaluations

Example runs for performing experiments


**Classification**
```
python run.py classification --graph graph_path.gml --emb file.embedding --output_file output_path.txt
```
**Edge prediction**

Firstly, prepare the training and test sets.
```
python run.py edge_prediction split --graph graph_path.gml --test_set_ratio 0.5 --split_folder folder_path
```
Then, compute the scores
```
python run.py edge_prediction predict --emb file.embedding --sample_file samples_path.pkl --output_file output.pkl
```
