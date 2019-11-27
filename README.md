## Project name: expereval

#### Create a virtual environment with conda

**1.** Create a virtual environment with name 'expereval'
```
conda create -n expereval python=2.7
```

**2.** Activate the virtual environment
```
conda activate expereval
```

**3.** Install the required packages
```
pip install -r requirements.txt
```

#### Classification
```
python run.py classification --graph graph_path.gml --emb file.embedding --output_file output_path.txt
```
#### Link prediction

**1.** Firstly, prepare the training and test sets.
```
python run.py link_prediction split --graph graph_path.gml --test_set_ratio 0.5 --split_folder folder_path
```
**2.** Compute the embeddings of the residual graph located in the specified 'folder_folder' in the previous step

**3.** Then, compute the scores
```
python run.py link_prediction predict --emb file.embedding --sample_file samples_path.pkl --output_file scores_file.pkl
```
**4.** Show the scores.
```
python run.py link_prediction read --input_file scores_file.pkl
```
