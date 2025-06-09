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

from dataset import Dataset
from model import SpaCLR, TrainerSpaCLR
from utils import get_predicted_results
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(args, name):
    # seed
    seed_torch(1)
    
    # args
    path = args.path
    gene_preprocess = args.gene_preprocess
    n_gene = args.n_gene
    last_dim = args.last_dim
    gene_dims=[n_gene, 2*last_dim]
    image_dims=[n_gene]
    lr = args.lr
    p_drop = args.p_drop
    batch_size = args.batch_size
    dataset = args.dataset
    epochs = args.epochs
    img_size = args.img_size
    device = args.device
    log_name = args.log_name
    num_workers = args.num_workers
    prob_mask = args.prob_mask
    pct_mask = args.pct_mask
    prob_noise = args.prob_noise
    pct_noise = args.pct_noise
    sigma_noise = args.sigma_noise
    prob_swap = args.prob_swap
    pct_swap = args.pct_swap
    backbone = args.backbone
    
    # dataset
    trainset = Dataset(dataset, path, name, gene_preprocess=gene_preprocess, n_genes=n_gene,
                       prob_mask=prob_mask, pct_mask=pct_mask, prob_noise=prob_noise, pct_noise=pct_noise, sigma_noise=sigma_noise,
                       prob_swap=prob_swap, pct_swap=pct_swap, img_size=img_size, train=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = Dataset(dataset, path, name, gene_preprocess=gene_preprocess, n_genes=n_gene,
                       prob_mask=prob_mask, pct_mask=pct_mask, prob_noise=prob_noise, pct_noise=pct_noise, sigma_noise=sigma_noise,
                       prob_swap=prob_swap, pct_swap=pct_swap, img_size=img_size, train=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # network
    network = SpaCLR(gene_dims=gene_dims, image_dims=image_dims, p_drop=p_drop, n_pos=trainset.n_pos, backbone=backbone, projection_dims=[last_dim, last_dim], hidden_dims=[512,last_dim])

    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-4)

    # log
    save_name = f'{name}_{args.w_g2g}_{args.w_i2i}_{args.w_recon}'
    log_dir = os.path.join('log', log_name, save_name)
    
    # train
    trainer = TrainerSpaCLR(args, trainset.n_clusters, network, optimizer, log_dir, device=device)
    if args.is_train:
        if args.is_load:
            trainer.load_model(args.ckpt_path)
        trainer.fit(trainloader, epochs, trainset.graph_data_tensor, testloader, testset.graph_data_tensor, dataset, name, path, trainset.adata)
    else:
        trainer.load_model(args.ckpt_path)
    xg, xg1, xi, label = trainer.valid(testloader, trainset.graph_data_tensor)
    if xi is None:
        z = xg + 0.5*xg1
    else:
        z = xg + 0.5*xg1 + 0.1*xi
        
    if z is not None:
        ari, pred_label=get_predicted_results(dataset, name, path,  z, trainset.adata)
    print("Ari value(z) : ", ari)
    if xg is not None:
        ari, pred_label=get_predicted_results(dataset, name, path,  xg, trainset.adata)
    print("Ari value(xg) : ", ari)
    if xg1 is not None:
        ari, pred_label=get_predicted_results(dataset, name, path,  xg1, trainset.adata)
    print("Ari value(xg1) : ", ari)
    if xi is not None:
        ari, pred_label=get_predicted_results(dataset, name, path,  xi, trainset.adata)
        print("Ari value(xi) : ", ari)
    if xg is not None:
        np.save(f'embeddings/{name}_xg.npy', xg)
    if xg1 is not None:
        np.save(f'embeddings/{name}_xg1.npy', xg1)
    if xi is not None:
        np.save(f'embeddings/{name}_xi.npy', xi)

        
