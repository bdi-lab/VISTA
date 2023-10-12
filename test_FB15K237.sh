#!/bin/bash

python3 test.py --data FB15K237 --lr 0.001 --dim 256 --test_epoch 750 --valid_epoch 50 --exp best \
                --num_layer_enc_ent 1 --num_layer_enc_rel 1 --num_layer_dec 1 \
                --num_head 64 --hidden_dim 2048 --dropout 0.1 \
                --emb_dropout 0.8 --vis_dropout 0.3 --txt_dropout 0.0 \
                --smoothing 0.0 --batch_size 512 --decay 0.0 --max_img_num 1 --step_size 50