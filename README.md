# VetTag

## Introduction

This is the official cleaned repo we used to train, evaluate and interpret for [VetTag paper](https://www.nature.com/articles/s41746-019-0113-1). 

Please feel free to contact `yuhui-zh15@mails.tsinghua.edu.cn` if you have any problem using these scripts. 

## Usage

### Unsupervised Learning

Please create a json file in `/path/to/hypes/` with the following format. 

```json
psvg.json
{
  "data_dir": "/path/to/data/psvg/",
  "encoder_path": "/path/to/data/encoder.json",
  "prefix": "psvg_oneline",
  "label_size": 0
}
```

- data_dir and prefix: save data in `/path/to/data/psvg/psvg_oneline_train.tsv`, `/path/to/data/psvg/psvg_oneline_valid.tsv` and `/path/to/data/psvg/psvg_oneline_test.tsv` for training, validation and test. The file should only contain one line for the whole text.

- encoder_path: save vocabulary in `/path/to/data/encoder.json`. It is a json file with format `{'hello': 0, 'world': 1, ...}`.

- label_size: for unsupervised learning, label size should equal to 0.

Then use the following command to train and save the model in `/path/to/exp/psvg/`.

`python trainer.py --outputdir /path/to/exp/psvg/ --train_emb --corpus psvg --hypes /path/to/hypes/psvg.json --batch_size 5 --bptt_size 600 --model_type transformer`

### Supervised Learning

Please create a json file in `/path/to/hypes/` with the following format. 

```json
csu.json
{
  "data_dir": "/path/to/data/csu/",
  "encoder_path": "/path/to/data/encoder.json",
  "prefix": "csu",
  "label_size": 4577
}
```

- data_dir and prefix: save data in `/path/to/data/csu/csu_train.tsv`, `/path/to/data/csu/csu_valid.tsv` and `/path/to/data/csu/csu_test.tsv` for training, validation and test. The file contains lines of annotated clinical notes with format `text <tab> label_1 <space> label_2 <space> ... <space> label_k` for each line.

- encoder_path: save vocabulary in `/path/to/data/encoder.json` (the same file for unsupervised learning). It is a json file with format `{'hello': 0, 'world': 1, ...}`.

- label_size: for supervised learning, we use 4577 finegrained SNOMED diagnosis codes.

Then use the following command to train and save the model in `/path/to/exp/csu/`.

`python trainer.py --outputdir /path/to/exp/csu/ --corpus csu --hypes /path/to/hypes/csu.json --batch_size 5 --model_type transformer --cut_down_len 600 --train_emb --hierachical --inputdir /path/to/exp/psvg/pretrained_model.pickle`

### External Evaluation

Please create a json file in `/path/to/hypes/` with the following format. 

```json
pp.json
{
  "data_dir": "/path/to/data/pp/",
  "encoder_path": "/path/to/data/encoder.json",
  "prefix": "pp",
  "label_size": 4577
}
```

- data_dir and prefix: save data in `/path/to/data/csu/pp_test.tsv` for test. The file contains lines of annotated clinical notes with format `text <tab> label_1 <space> label_2 <space> ... <space> label_k` for each line.

- encoder_path: save vocabulary in `/path/to/data/encoder.json` (the same file for unsupervised learning). It is a json file with format `{'hello': 0, 'world': 1, ...}`.

- label_size: for supervised learning, we use 4577 finegrained SNOMED diagnosis codes (the same for supervised learning).

Then use the following command to evaluate the model.

`python trainer.py --outputdir /path/to/exp/pp/ --corpus pp --hypes /path/to/hypes/pp.json --batch_size 5 --model_type transformer --cut_down_len 600 --hierachical --inputdir /path/to/exp/psvg/pretrained_model.pickle`

### Statistics and Analysis

Refer to `jupyter/snomed_stat.ipynb`, `jupyter/species_stat.ipynb`, `jupyter/length_label_distribution.ipynb` and `jupyter/analysis.ipynb`

### Hierarchical Training

Two files are required: `parents.json` and `labels.json` (in `data` dir).

- labels.json: the format is [SNOMED_ID_1, SNOMED_ID_2, …, SNOMED_ID_4577], which is all 4577 SNOMED labels we use. 
- parents.json: the format is {SNOMED_ID_i: parent_of_SNOMED_ID_i}, which is all SNOMED labels and their parents in the shortest path from the root node (introduced in the method section).

### Interpretation

Refer to `jupyter/interpret.ipynb` and `jupyter/salient_words.ipynb`

