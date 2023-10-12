from dataset import VTKG
from model import VISTA
from tqdm import tqdm
from utils import calculate_rank, metrics
import numpy as np
import argparse
import torch
import torch.nn as nn
import datetime
import time
import os
import copy
import math
import random
import distutils
import logging

OMP_NUM_THREADS=8
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
torch.cuda.empty_cache()

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_format)
logger.addHandler(stream_handler)


parser = argparse.ArgumentParser()
parser.add_argument('--data', default = "VTKG-C", type = str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--dim', default=256, type=int)
parser.add_argument('--test_epoch', default=150, type=int)
parser.add_argument('--valid_epoch', default=50, type=int)
parser.add_argument('--exp', default='best')
parser.add_argument('--no_write', action='store_true')
parser.add_argument('--num_layer_enc_ent', default=2, type=int)
parser.add_argument('--num_layer_enc_rel', default=1, type=int)
parser.add_argument('--num_layer_dec', default=2, type=int)
parser.add_argument('--num_head', default=4, type=int)
parser.add_argument('--hidden_dim', default = 2048, type = int)
parser.add_argument('--dropout', default = 0.01, type = float)
parser.add_argument('--emb_dropout', default = 0.9, type = float)
parser.add_argument('--vis_dropout', default = 0.4, type = float)
parser.add_argument('--txt_dropout', default = 0.1, type = float)
parser.add_argument('--smoothing', default = 0.0, type = float)
parser.add_argument('--batch_size', default = 512, type = int)
parser.add_argument('--decay', default = 0.0, type = float)
parser.add_argument('--max_img_num', default = 3, type = int)
parser.add_argument('--step_size', default = 50, type = int)
args = parser.parse_args()

file_format = ""

for arg_name in vars(args).keys():
    if arg_name not in ["data", "exp", "no_write", "test_epoch"]:
        file_format+=f"_{vars(args)[arg_name]}"

os.makedirs(f"./logs_test/{args.exp}/{args.data}", exist_ok = True)

file_handler = logging.FileHandler(f"./logs_test/{args.exp}/{args.data}/{file_format}.log")
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

logger.info(f"{os.getpid()}")

KG = VTKG(args.data, logger, max_vis_len = args.max_img_num)

KG_Loader = torch.utils.data.DataLoader(KG, batch_size = args.batch_size, shuffle=True)
model = VISTA(num_ent = KG.num_ent, num_rel = KG.num_rel, ent_vis = KG.ent_vis_matrix, rel_vis = KG.rel_vis_matrix, \
              dim_vis = KG.vis_feat_size, ent_txt = KG.ent_txt_matrix, rel_txt = KG.rel_txt_matrix, dim_txt = KG.txt_feat_size, \
              ent_vis_mask = KG.ent_vis_mask, rel_vis_mask = KG.rel_vis_mask, dim_str = args.dim, num_head = args.num_head, \
              dim_hid = args.hidden_dim, num_layer_enc_ent = args.num_layer_enc_ent, num_layer_enc_rel = args.num_layer_enc_rel, \
              num_layer_dec = args.num_layer_dec, dropout = args.dropout, \
              emb_dropout = args.emb_dropout, vis_dropout = args.vis_dropout, txt_dropout = args.txt_dropout).cuda()



loaded_ckpt = torch.load(f"./ckpt/{args.exp}/{args.data}/{file_format}_{args.test_epoch}.ckpt")
model.load_state_dict(loaded_ckpt['model_state_dict'])


all_ents = torch.arange(KG.num_ent).cuda()
all_rels = torch.arange(KG.num_rel).cuda()


model.eval()
with torch.no_grad():
    test_lp_list_rank = []    
    ent_embs, rel_embs = model()

    for triplet in tqdm(KG.test):
        h,r,t = triplet

        head_score = model.score(ent_embs, rel_embs, torch.tensor([[KG.num_ent + KG.num_rel, r + KG.num_ent, t + KG.num_rel]]).cuda())[0].detach().cpu().numpy()
        head_rank = calculate_rank(head_score, h, KG.filter_dict[(-1, r, t)])
        tail_score = model.score(ent_embs, rel_embs, torch.tensor([[h + KG.num_rel, r + KG.num_ent, KG.num_ent + KG.num_rel]]).cuda())[0].detach().cpu().numpy()
        tail_rank = calculate_rank(tail_score, t, KG.filter_dict[(h, r, -1)])

        test_lp_list_rank.append(head_rank)
        test_lp_list_rank.append(tail_rank)

    test_lp_list_rank = np.array(test_lp_list_rank)
    tmr, tmrr, thit10, thit3, thit1 = metrics(test_lp_list_rank)
    logger.info("Link Prediction on Test Set")
    logger.info(f"MR: {tmr}")
    logger.info(f"MRR: {tmrr}")
    logger.info(f"Hit10: {thit10}")
    logger.info(f"Hit3: {thit3}")
    logger.info(f"Hit1: {thit1}")



