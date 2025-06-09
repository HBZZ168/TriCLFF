from ast import parse
import os
import psutil
import time
import random
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader
import argparse
from train import train
import warnings
warnings.filterwarnings("ignore")
       
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()   

    # preprocess
    parser.add_argument('--dataset', type=str, default="SpatialLIBD")   
    parser.add_argument('--path', type=str, default="/root/miniconda3/envs/TriCLFF/")
    parser.add_argument("--gene_preprocess", choices=("pca", "hvg"), default="hvg") 
    parser.add_argument("--n_gene", choices=(300, 1000, 3000), default=3000)
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--num_workers', type=int, default=15)

    # model
    parser.add_argument('--last_dim', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--p_drop', type=float, default=0.3)
    
    parser.add_argument('--w_g2i', type=float, default=1)
    parser.add_argument('--w_s2i', type=float, default=1)
    parser.add_argument('--w_g2g', type=float, default=0.1)
    parser.add_argument('--w_s2s', type=float, default=0.1)
    parser.add_argument('--w_s2g', type=float, default=0.1)
    parser.add_argument('--w_i2i', type=float, default=0.1)
    parser.add_argument('--w_recon', type=float, default=0.3)
    parser.add_argument('--w_graph_loss', type=float, default=0.5)

    # data augmentation
    parser.add_argument('--prob_mask', type=float, default=0.5)
    parser.add_argument('--pct_mask', type=float, default=0.2)
    parser.add_argument('--prob_noise', type=float, default=0.5)
    parser.add_argument('--pct_noise', type=float, default=0.8)
    parser.add_argument('--sigma_noise', type=float, default=0.5)
    parser.add_argument('--prob_swap', type=float, default=0.5)
    parser.add_argument('--pct_swap', type=float, default=0.1)
    parser.add_argument('--backbone', type=str, default='swin_s')

    # train
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--log_name', type=str, default="log_name")
    parser.add_argument('--name', type=str, default="151672")
    
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--is_load', type=bool, default=False)
    parser.add_argument('--ckpt_path', type=str, default="last.pth")

    args = parser.parse_args()
    print(args)
    train(args, args.name)
