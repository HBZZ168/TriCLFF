import os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
from sklearn.metrics import adjusted_rand_score

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.02, 2, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            return res

def KMeans_predict(x, n_clusters, model='EEE', random_seed=2020):

    np.random.seed(random_seed)
    import sklearn.cluster as sc
    model = sc.KMeans(n_clusters=n_clusters)
    yhat = model.fit_predict(x)
    return yhat

def eval_mclust_ari(labels, z, n_clusters):
    raw_preds = KMeans_predict(z, n_clusters)
    return  raw_preds
