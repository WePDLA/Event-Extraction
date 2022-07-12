# Event-Extraction
Event Extraction
## Requirement
mkdir graph_ee_models

download PLM ernie-1.0 into ./graph_ee_models

pip install -r rep.txt (-i http://mirrors.aliyun.com/pypi/simple/)

## How to use your own dataset
#### Step 1
cd ./graph_ee/data/data_preprocess

set source_data_dir in ann2bio.py to your data_dir (data_dir including all *.txt and *.ann)

python ann2bio.py to get trig_data_bio.tsv and arg_data_bio.tsv,
then split trig_data_bio.tsv into train.tsv, dev.tsv and test.tsv(arg_data_bio.tsv is the same)

#### Step 2
cd ./graph_ee/data/data_name

mkdir trigger_set, role_set, 
then move train.tsv, dev.tsv and test.tsv obtained by trig_data_bio.tsv into trigger_set, 
move train.tsv, dev.tsv and test.tsv obtained by arg_data_bio.tsv into role_set 

#### Step 3
cd ./graph_ee/src/conf/data_name

Create trigger_tag.dict, which contains all the tags of tirgger, and create role_tag.dict in the same way. (Example is given in ACE)

Finally, find ./graph_ee/src/sent_sequence_labeling.py, set config_dir and data_dir to './conf/data_name/', '../data/data_name/' respectively.
Then, you can train your own model.


## Train Trigger
python ./graph_ee/src/sent_sequence_labeling.py
## Test Trigger
set ./graph_ee/src/sent_sequence_labeling.py do_predict_mark=True other is False

python ./graph_ee/src/sent_sequence_labeling.py
## Train Arguments
set ./graph_ee/src/sent_sequence_labeling.py do_arg_train_mark=True other is False

python ./graph_ee/src/sent_sequence_labeling.py
## Test Arguments
set ./graph_ee/src/sent_sequence_labeling.py do_arg_predict_mark=True other is False

python ./graph_ee/src/sent_sequence_labeling.py

