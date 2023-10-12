#!/bin/bash

python3 test.py --data VTKG-I --lr 0.0001 --dim 256 --test_epoch 150 --valid_epoch 50 --exp best \
                --num_layer_enc_ent 4 --num_layer_enc_rel 1 --num_layer_dec 2 \
                --num_head 4 --hidden_dim 768 --dropout 0.01 \
                --emb_dropout 0.7 --vis_dropout 0.4 --txt_dropout 0.1 \
                --smoothing 0.0 --batch_size 128 --decay 0.0 --max_img_num 5 --step_size 50