import os
import numpy as np
import pandas as pd
from copy import deepcopy
import torch
from torch.utils import data
from torchvision import transforms
from torchtoolbox.transform import Cutout
import cv2
import scanpy as sc

import random
from scipy.sparse import csr_matrix

from utils import load_ST_file, adata_preprocess_pca, adata_preprocess_hvg, extract_wash_patches, calculate_adj_matrix
from torch_geometric.data import Data
import sklearn.neighbors as sklearn_neighbors
import scipy.sparse as sp
from sys import getsizeof as getsize
import json


def Batch_Data(adata, num_batch_x, num_batch_y, spatial_key=['array_row', 'array_col'], plot_Stats=False):
    Sp_df = adata.obs.loc[:, spatial_key].copy()
    Sp_df = np.array(Sp_df)
    batch_x_coor = [np.percentile(Sp_df[:, 0], (1/num_batch_x)*x*100) for x in range(num_batch_x+1)]
    batch_y_coor = [np.percentile(Sp_df[:, 1], (1/num_batch_y)*x*100) for x in range(num_batch_y+1)]

    Batch_list = []
    for it_x in range(num_batch_x):
        for it_y in range(num_batch_y):
            min_x = batch_x_coor[it_x]
            max_x = batch_x_coor[it_x+1]
            min_y = batch_y_coor[it_y]
            max_y = batch_y_coor[it_y+1]
            temp_adata = adata.copy()

            temp_adata = temp_adata[temp_adata.obs[spatial_key[0]].map(lambda x: min_x <= x <= max_x)]
            temp_adata = temp_adata[temp_adata.obs[spatial_key[1]].map(lambda y: min_y <= y <= max_y)]
            Batch_list.append(temp_adata)
    if plot_Stats:
        f, ax = plt.subplots(figsize=(1, 3))
        plot_df = pd.DataFrame([x.shape[0] for x in Batch_list], columns=['#spot/batch'])
        sns.boxplot(y='#spot/batch', data=plot_df, ax=ax)
        sns.stripplot(y='#spot/batch', data=plot_df, ax=ax, color='red', size=5)
    return Batch_list


def Transfer_pytorch_Data_label(adata, label, select_X):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(cells.shape[0], cells.shape[0]))
    G = G + sp.eye(G.shape[0])
    edgeList = np.nonzero(G)
    np.array([edgeList[0], edgeList[1]])
    if type(select_X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(select_X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(select_X.todense()))  # .todense()
    return data     #构建的图结构SNN

def Transfer_pytorch_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].apply(lambda x: max(0, x))
    G_df['Cell2'] = G_df['Cell2'].apply(lambda x: max(0, x))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(cells.shape[0], cells.shape[0]))
    G = G + sp.eye(G.shape[0])
    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data

def Transfer_pytorch_Data1(adata): 
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    # 转换 'Cell1' 和 'Cell2' 为数值类型
    G_df['Cell1'] = pd.to_numeric(G_df['Cell1'], errors='coerce')
    G_df['Cell2'] = pd.to_numeric(G_df['Cell2'], errors='coerce')
    # 处理 NaN 值（根据需要选择填充或删除）
    G_df = G_df.dropna(subset=['Cell1', 'Cell2'])
    # 将 Cell1 和 Cell2 中的索引转换为整数 ID
    G_df['Cell1'] = G_df['Cell1'].apply(lambda x: max(0, x) if pd.notna(x) else 0)
    G_df['Cell2'] = G_df['Cell2'].apply(lambda x: max(0, x) if pd.notna(x) else 0)
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(cells.shape[0], cells.shape[0]))
    G = G + sp.eye(G.shape[0])
    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data

def Cal_Spatial_Net(adata, rad_cutoff=150, k_cutoff=150, model='Radius', verbose=True, label=None):
    """\
    Construct the spatial neighbor networks.
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. 
        When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    """
 
    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    
    if label is not None:
        coor = pd.DataFrame(adata.obsm['spatial'])
        coor = coor[label != -1]
        coor.index = adata.obs.index[label != -1]
    else:
        coor = pd.DataFrame(adata.obsm['spatial'])
        coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']


    if model == 'Radius':

        nbrs = sklearn_neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn_neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))
 
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
 
    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net
    
def Cal_Spatial_Net1(adata, coor1, rad_cutoff=150, k_cutoff=150, model='Radius', verbose=True, label=None): 
    assert(model in ['Radius', 'KNN'])
    if label is not None:
        coor = coor1
        coor = coor[label != -1]
        coor.index = coor.index[label != -1]
    else:
        coor = coor1
        coor.index = coor1.index
    coor.columns = ['imagerow', 'imagecol']
    if model == 'Radius':
        nbrs = sklearn_neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    if model == 'KNN':
        nbrs = sklearn_neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    adata.uns['Spatial_Net'] = Spatial_Net
    
class Dataset(data.Dataset):
    def __init__(self, dataset, path, name, gene_preprocess='pca', n_genes=3000,
                 prob_mask=0.5, pct_mask=0.2, prob_noise=0.5, pct_noise=0.8, sigma_noise=0.5,
                 prob_swap=0.5, pct_swap=0.1, img_size=112, train=True):
        super(Dataset, self).__init__()
        self.dataset = dataset
 
        if dataset == "MouseOlfactoryBulb":
            root_path = os.path.join(path, name)
            counts_file = os.path.join(root_path,'RNA_counts.tsv')
            coor_file = os.path.join(root_path, 'position.tsv')
            counts = pd.read_csv(counts_file, sep='\t', index_col=0, nrows=100000)
            coor_df = pd.read_csv(coor_file, sep='\t')
            counts.columns = ['Spot_'+str(x) for x in counts.columns]
            coor_df.index = coor_df['label'].map(lambda x: 'Spot_'+str(x))
            coor_df = coor_df.loc[:, ['x','y']]
            adata = sc.AnnData(counts.T)

            adata.var_names_make_unique()
            coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
            adata.obsm["spatial"] = coor_df.to_numpy().astype(int)
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts", show=False)
            used_barcode = pd.read_csv(os.path.join(root_path, 'used_barcodes.txt'), sep='\t', header=None)
            used_barcode = used_barcode[0]
            adata.obsm['only_index'] = np.array([i for i in range(len(adata))])
            adata = adata[used_barcode,]
            sc.pp.filter_genes(adata, min_cells=50)
            self.label = None
        
        elif dataset == "MouseBrain" or dataset=="BreastCancer":
            adata = load_ST_file(os.path.join(path, name))
            df_meta = pd.read_csv(os.path.join(path, name, 'metadata.tsv'), sep='\t')
            try:
                self.label = pd.Categorical(df_meta['layer_guess']).codes
            except:
                self.label = pd.Categorical(df_meta['ground_truth']).codes
            # image
            full_image = cv2.imread(os.path.join(path, name, './spatial/tissue_hires_image.png'))
            full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
            with open(os.path.join(path, name, './spatial/scalefactors_json.json'), 'r') as fp:
                scale = json.load(fp)['tissue_hires_scalef']
            patches = []
            for x, y in adata.obsm['spatial']:
                x = int(round(x*scale))
                y = int(round(y*scale))
                patches.append(full_image[y-img_size:y+img_size, x-img_size:x+img_size])
            patches = np.array(patches)
            self.image = patches

        elif dataset == "SpatialLIBD":
            adata = load_ST_file(os.path.join(path, name))
            df_meta = pd.read_csv(os.path.join(path, name, 'metadata.tsv'), sep='\t')
            try:
                self.label = pd.Categorical(df_meta['layer_guess']).codes
            except:
                self.label = pd.Categorical(df_meta['ground_truth']).codes
            # image
            full_image = cv2.imread(os.path.join(path, name, f'{name}_full_image.tif'))
            full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
            patches = []
            for x, y in adata.obsm['spatial']:           
                patches.append(full_image[y-img_size:y+img_size, x-img_size:x+img_size])
            patches = np.array(patches)
            self.image = patches

        elif dataset == "Slideseq": 
        
            adata = sc.read("/root/autodl-tmp/ConGI/data/Slideseq/STARmap_mouse_brain_data.h5ad")
            adata.var_names_make_unique()
            sc.pp.filter_genes(adata, min_cells=50)
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            self.label = None

        adata.X = sp.csr_matrix(adata.X)
        try:
            self.n_clusters = self.label.max() + 1
        except:
            self.n_clusters = None
        self.spatial = adata.obsm['spatial']
        self.n_pos = self.spatial.max() + 1
        if gene_preprocess == 'pca':
            self.gene = adata_preprocess_pca(adata, pca_n_comps=n_genes).astype(np.float32)
        elif gene_preprocess == 'hvg':
            self.gene = adata_preprocess_hvg(adata, n_top_genes=n_genes)

        self.train = train
        self.img_train_transform = transforms.Compose([
            Cutout(0.5),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.RandomAffine(45),
            
            transforms.RandomCrop(150),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomApply([transforms.ColorJitter(contrast=1)], p=0.8),
            transforms.RandomApply([transforms.ColorJitter(saturation=0.5)], p=0.8),
            transforms.RandomApply([transforms.ColorJitter(saturation=0.5)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.img_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.gene_train_transform = GeneTransforms(n_genes, 
                                                    prob_mask=0.5, pct_mask=0.2,
                                                    prob_noise=0.5, pct_noise=0.8, sigma_noise=0.5,
                                                    prob_swap=0.5, pct_swap=0.1)
      

        Cal_Spatial_Net(adata)
        self.adata = adata
        if 'highly_variable' in adata.var.columns:
            adata_Vars = adata[:, adata.var['highly_variable']]
        else:
            adata_Vars = adata
        #adata1.X = adata1.X[self.label != -1]
        self.graph_data_tensor = Transfer_pytorch_Data_label(adata_Vars, label=self.label, select_X=self.gene)
        self.gene = self.gene.todense().A
        self.gene = np.squeeze(self.gene)

    def __len__(self):
        return len(self.gene)
    
    def __getitem__(self, idx):
        spatial = torch.from_numpy(self.spatial[idx])
        if self.train:  
               
            xg = self.gene[idx].astype(np.float32)
            xg_u = self.gene_train_transform(deepcopy(xg))
            xg_v = self.gene_train_transform(deepcopy(xg))
            xg = torch.from_numpy(xg)
            xg_u = torch.from_numpy(xg_u)
            xg_v = torch.from_numpy(xg_v)
            if self.dataset != 'MouseOlfactoryBulb':
                xi_u = self.img_train_transform(self.image[idx])
                xi_v = self.img_train_transform(self.image[idx])
                y = self.label[idx]
            else:
                xi_u = torch.ones(1, 1)
                xi_v = torch.ones(1, 1)
                y = torch.ones(1, 1)
            
            return xg, xg_u, xg_v, xi_u, xi_v, spatial, y, idx
        
        else:           
            xg = self.gene[idx]
            xg = torch.from_numpy(xg).float()
            if self.dataset == 'MouseOlfactoryBulb':
                xi = torch.ones(1, 1)        
                y = torch.ones(1, 1)
            else:
                xi = self.img_test_transform(self.image[idx])
                y = self.label[idx]
            return xg, xi, spatial, y, idx

    
class GeneTransforms(torch.nn.Module):
    def __init__(self, n_genes,
                prob_mask, pct_mask,
                prob_noise, pct_noise, sigma_noise,
                prob_swap, pct_swap):
        super(GeneTransforms, self).__init__()
        
        self.n_genes = n_genes
        self.prob_mask = prob_mask
        self.pct_mask = pct_mask
        self.prob_noise = prob_noise
        self.pct_noise = pct_noise
        self.sigma_noise = sigma_noise
        self.prob_swap = prob_swap
        self.pct_swap = pct_swap
        
    def build_mask(self, pct_mask):
        mask = np.concatenate([np.ones(int(self.n_genes * pct_mask), dtype=bool), 
                               np.zeros(self.n_genes - int(self.n_genes * pct_mask), dtype=bool)])
        np.random.shuffle(mask)
        return mask
        
    def forward(self, xg):
        if np.random.uniform(0, 1) < self.prob_mask:
            mask = self.build_mask(self.pct_mask)
            xg[mask] = 0
        
        if np.random.uniform(0, 1) < self.prob_noise:
            mask = self.build_mask(self.pct_noise)
            noise = np.random.normal(0, self.sigma_noise, int(self.n_genes * self.pct_noise))
            xg[mask] += noise
        
        if np.random.uniform(0, 1) < self.prob_swap:
            swap_pairs = np.random.randint(self.n_genes, size=(int(self.n_genes * self.pct_swap / 2), 2))
            xg[swap_pairs[:, 0]], xg[swap_pairs[:, 1]] = xg[swap_pairs[:, 1]], xg[swap_pairs[:, 0]]
            
        return xg

# 数据增强执行Mask, Noise, Swap
class GeneExpressionAugmenter:
    def __init__(self, prob_mask, prob_noise, sigma_noise, prob_swap, pct_swap):
        self.prob_mask = prob_mask
        self.prob_noise = prob_noise
        self.sigma_noise = sigma_noise
        self.prob_swap = prob_swap
        self.pct_swap = pct_swap

    def apply_mask(self, adata):
        num_rows, num_cols = adata.X.shape
        mask = np.random.rand(num_rows, num_cols) < self.prob_mask
        mask = mask.astype(float)
        masked_data = adata.X.multiply(1 - mask)
        adata.X = masked_data
        return adata

    def apply_noise(self, adata):
        num_rows, num_cols = adata.X.shape
        noise = np.random.normal(0, self.sigma_noise, (num_rows, num_cols))
        noisy_data = adata.X + noise
        adata.X = noisy_data
        return adata

    def apply_swap(self, adata):
        num_rows, num_cols = adata.X.shape
        num_swap = int(self.pct_swap * num_rows * num_cols)  # 计算交换的数量
        swap_indices = random.sample(range(num_rows * num_cols), num_swap)
        adata_swapped = adata.X.copy()
        for idx in swap_indices:
            cell_idx = idx // num_cols
            gene_idx = idx % num_cols
            # 随机选择要交换的基因
            swap_with_cell = random.randint(0, num_rows-1)
            swap_with_gene = random.randint(0, num_cols-1)
            adata_swapped[cell_idx, gene_idx], adata_swapped[swap_with_cell, swap_with_gene] = \
                adata_swapped[swap_with_cell, swap_with_gene], adata_swapped[cell_idx, gene_idx]
        adata.X = adata_swapped 
        return adata
    
    def augment(self, adata):
        adata1 = adata.copy()
        adata2 = adata.copy()
        # Mask
        if random.random() < self.prob_mask:
            adata1 = self.apply_mask(adata1)
            adata2 = self.apply_mask(adata2)
        # Noise
        if random.random() < self.prob_noise:
            adata1 = self.apply_noise(adata1)
            adata2 = self.apply_noise(adata2)
        # Swap
        if random.random() < self.prob_swap:
            adata1 = self.apply_swap(adata1)
            adata2 = self.apply_swap(adata2)
        return adata1, adata2

