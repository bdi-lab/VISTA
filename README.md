# VISTA: Visual-Textual Knowledge Graph Representation Learning
This code is the official implementation of the following [paper]():

> Jaejun Lee, Chanyoung Chung, and Joyce Jiyoung Whang, VISTA: Visual-Textual Knowledge Graph Representation Learning, Findings of the Association for Computational Linguistics: EMNLP 2023 (Findings of EMNLP 2023).

All codes are written by Jaejun Lee (jjlee98@kaist.ac.kr). When you use this code or data, please cite our paper.
```bibtex
@inproceedings{vista,
	author={Jaejun Lee and Chanyoung Chung and Joyce Jiyoung Whang},
	title={VISTA: Visual-Textual Knowledge Graph Representation Learning},
	booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
	year={2023},
	pages={},
	doi={}
}
```

## Requirements

We used python 3.8 and PyTorch 1.12.1 with cudatoolkit 11.3.

You can install all requirements with:

```shell
pip install -r requirements.txt
```

## Datasets

Due to the large size of the visual features, we provide the link to download the datasets. Note that we cannot provide the raw images due to some potential license problems. To use the datasets, place the unzipped data folder in the same directory with the codes.

You can download the datasets from https://drive.google.com/file/d/1u4QthmEboMzRarF_HLYfLDOLcOZeH8Gp/view?usp=drive_link

## Reproducing the Reported Results

We provide the checkpoints to produce the results on VTKG-I, VTKG-C, WN18RR++, and FB15K237. If you want to use the checkpoints, place the unzipped checkpoint folder in the same directory with the codes.

You can download the checkpoints from https://drive.google.com/file/d/1EYKrE2yLMgRRfpzR17UgRiRBQMiFOQHc/view?usp=drive_link

The commands to reproduce the results in our paper:

### VTKG-I

```python
bash test_VTKG-I.sh
```

### VTKG-C

```python
bash test_VTKG-C.sh
```

### WN18RR++

```python
bash test_WN18RR++.sh
```

### FB15K237

```python
bash test_FB15K237.sh
```

## Training from Scratch

To train VISTA from scratch, run `train.py` with arguments. Please refer to `train.py` or `test.py` for the examples of the arguments.

The list of arguments of 'train.py':
- `--data`: name of the dataset
- `--lr`: learning rate
- `--dim`: $d$
- `--num_epoch`: total number of training epochs (only used for train.py)
- `--test_epoch`: the epoch to test (only used for test.py)
- `--valid_epoch`: the duration of validation
- `--exp`: experiment name
- `--num_layer_enc_ent`: $L$
- `--num_layer_enc_rel`: $\widehat{L}$
- `--num_layer_dec`: $\widetilde{L}$
- `--num_head`: number of attention heads
- `--hidden_dim`: the hidden dimension of the transformers
- `--dropout`: the dropout rate of the transformers
- `--emb_dropout`: the dropout rate of the embedding matrices
- `--vis_dropout`: the dropout rate of the visual representation vectors
- `--txt_dropout`: the dropout rate of the textual representation vectors
- `--smoothing`: label smoothing ratio
- `--batch_size`: the batch size
- `--decay`: the weight decay
- `--max_img_num`: $k=\hat{k}$
- `--step_size`: the step size of the cosine annealing learning rate scheduler
