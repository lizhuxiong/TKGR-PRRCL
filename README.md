
## Building the Project

### File folders

data: the dataset folder

edge_input: contains the edge files used in preprocessing.

models: contains the main code for the model.

path_data: The result of path collection.

preprocess: contains the preprocessing files.


### Requirements
To run the code, you need the following dependencies at least:

- networkx==2.3 
- numpy==1.21.6 
- pandas==1.3.5 
- scikit_learn==1.2.1 
- scipy==1.4.1
- scipy==1.4.1 
- torch==1.11.0 
- torch_cluster==1.5.9 
- torch_scatter==2.0.9 
- torch_sparse==0.6.12 
- torch_spline_conv==1.2.1 
- torch_geometric==2.0.4 
- tqdm==4.43.0 
- python ==3.8

### preprocessing
The code executes in the following order:
(1) python process_icews.py
(2) python path_data.py
(3) python init_rw.py
(4) g++ gen_merw.cpp -o gen_merw -g -Wall -O2 -mcmodel=medium
(5) ./gen_merw [data_name] [path_num] [path_length]. Such as ./gen_merw ICEWS14 1 8

## Main model

### Files
`learner.py`: The beginning of the model.

`datasets.py`: Used to get datasets

`models.py`: The part of model.

`neibs_info_embedding.py`: The code that gets path embedding information.

`contrastive_learning.py`: The code that gets contrastive learning

### Run the Experiments
python learner.py --dataset ICEWS14 --rank 800  --valid_freq 2 --max_epoch 20 --learning_rate 0.1 --batch_size 50 --n_hidden 160 --num_walks 1 --walk_len 8

## Contact

If you have any problem, feel free to contact with me. Thanks.

E-mail: xionglizhu@email.swu.edu.cn

## Citation

Please consider citing the following paper when using our code for your application.
