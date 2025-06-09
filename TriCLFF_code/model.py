import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import scanpy as sc
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, densenet121, swin_s, swin_b
from torchtoolbox.tools import mixup_data, cutmix_data, mixup_criterion

from utils import load_ST_file, adata_preprocess_pca, adata_preprocess_hvg, calculate_adj_matrix, refine, get_predicted_results, extract_wash_patches, calculate_adj_matrix
from metrics import  eval_mclust_ari
from loss import NT_Xent
from loss import DCL
from loss import DCLW
from dataset import Cal_Spatial_Net, Transfer_pytorch_Data, Cal_Spatial_Net1, Transfer_pytorch_Data1, GeneExpressionAugmenter, Transfer_pytorch_Data_label
import torch.nn.functional as F
from gat_conv import GATConv
from torch.optim import lr_scheduler


def LinearBlock(input_dim, output_dim, p_drop):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )

class GATEncoder(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(GATEncoder, self).__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv1_2 = GATConv(num_hidden, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2_h = GATConv(out_dim, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    def forward(self, features, edge_index):

        h1 = F.elu(self.conv1(features, edge_index))
        h1 =  F.elu(self.conv1_2(h1, edge_index, attention=True))
        h2 = self.conv2(h1, edge_index, attention=True)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h_h = self.conv2_h(h2, edge_index, attention=False)
        h4 = self.conv4(h3, edge_index, attention=False)
        return h2, h4, h_h  # F.log_softmax(x, dim=-1)

class SpaCLR(nn.Module):
    def __init__(self, gene_dims, image_dims, p_drop, n_pos, backbone='swin_s', projection_dims=[64, 64], hidden_dims=[512, 30], is_gat=True):
        super(SpaCLR, self).__init__()
        gene_dims.append(projection_dims[0])
        if not is_gat:
            self.gene_encoder = nn.Sequential(OrderedDict([
                (f'gene_block{i+1}', LinearBlock(gene_dims[i], gene_dims[i+1], p_drop)) for i, _ in enumerate(gene_dims[:-1])
            ]))

            self.gene_decoder = nn.Linear(projection_dims[0], gene_dims[0])
        else:
            self.gene_gat = GATEncoder(hidden_dims = [gene_dims[0]] + hidden_dims)
            self.gene_encoder = nn.Sequential(OrderedDict([
                (f'gene_block{i+1}', LinearBlock(gene_dims[i], gene_dims[i+1], p_drop)) for i, _ in enumerate(gene_dims[:-1])
            ]))

            self.gene_decoder = nn.Linear(projection_dims[0], gene_dims[0])
        self.mse_loss = nn.MSELoss()

        if backbone == 'densenet':
            self.image_encoder = densenet121(pretrained=True)
            n_features = self.image_encoder.classifier.in_features
            self.image_encoder.classifier = nn.Identity()
        elif backbone == 'resnet':
            self.image_encoder = resnet50(pretrained=True)
            n_features = self.image_encoder.fc.in_features
            self.image_encoder.fc = nn.Identity()
        elif backbone == 'swin_s':

            self.image_encoder = swin_s(pretrained=True)

            n_features = self.image_encoder.head.in_features
            for name, param in self.image_encoder.named_parameters():
                param.requires_grad = False
            self.image_encoder.head = self.projector = nn.Sequential(
                                        nn.Linear(n_features, n_features),
                                        nn.ReLU(),
                                        nn.Linear(n_features, n_features),
                                      )

        self.x_embedding = nn.Embedding(n_pos, n_features)
        self.y_embedding = nn.Embedding(n_pos, n_features)

        image_dims[0] = n_features
        image_dims.append(projection_dims[0])
        self.image_linear = nn.Sequential(OrderedDict([
            (f'image_block{i+1}', LinearBlock(image_dims[i], image_dims[i+1], p_drop)) for i, _ in enumerate(image_dims[:-1])
        ]))

        self.projector = nn.Sequential(
            nn.Linear(projection_dims[0], projection_dims[0]),
            nn.ReLU(),
            nn.Linear(projection_dims[0], projection_dims[1]),
            nn.ReLU(),
            nn.Linear(projection_dims[0], projection_dims[1]),
            nn.Tanh()
        )
        self.projector2 = nn.Sequential(
            nn.Linear(projection_dims[0], projection_dims[0]),
            nn.ReLU(),
            nn.Linear(projection_dims[0], projection_dims[1]),
            nn.ReLU(),
            nn.Linear(projection_dims[0], projection_dims[1]),
            nn.Tanh()
        )
        self.projector3 = nn.Sequential(
            nn.Linear(projection_dims[0], projection_dims[0]),
            nn.ReLU(),
            nn.Linear(projection_dims[0], projection_dims[1]),
            nn.ReLU(),
            nn.Linear(projection_dims[0], projection_dims[1]),
            nn.Tanh()
        )
    def forward_image(self, xi, spatial):
        xi = self.image_encoder(xi)        
        xi = self.image_linear(xi)
        hi = self.projector2(xi)

        return xi, hi

    def forward_gene(self, xg):
        xg = self.gene_encoder(xg)
        hg = self.projector3(xg)

        return xg, hg
    def forward_gene_graph_old(self, data, idx, specific_data=None):

        if specific_data is not None:
            graph1 = data.x.detach().clone()
            graph1[idx] = specific_data
        else:
            graph1 = data.x
        z, out, hg = self.gene_gat(graph1, data.edge_index)
        loss_graph = F.mse_loss(graph1[idx], out[idx])
        hg = self.projector(hg)
        del graph1
        return z[idx], hg[idx], loss_graph
    def forward_gene_graph(self, data):
        z, out, _ = self.gene_gat(data.x, data.edge_index)
        hg = self.projector(z)
        return z, out, hg

    def forward(self, xg, xi, spatial, is_graph=True, graph=None, idx=None, specific_data=None):
        if is_graph:
            xg, hg, _ = self.forward_gene_graph(graph1, idx, specific_data)
        else:
            xg, hg = self.forward_gene(xg)
        xi, hi = self.forward_image(xi, spatial)

        return xg, xi, hg, hi
    
    def recon_loss(self, zg, xg, graph_data=None, idx=None, specific_data=None):
        if idx is None:
            zg = self.gene_decoder(zg)
        else:
            zg, _, _ = self.forward_gene_graph(graph_data, idx, specific_data)
        return self.mse_loss(zg, xg)


class TrainerSpaCLR:
    def __init__(self, args, n_clusters, network, optimizer, log_dir, device='cuda', is_gat=True):
        self.n_clusters = n_clusters
        self.network = network
        self.optimizer = optimizer
        self.optimizer_graph = torch.optim.Adam(self.network.gene_gat.parameters(), lr=0.001, weight_decay=0.0001)
        self.optimizer_rest = torch.optim.Adam(self.network.gene_gat.parameters(), lr=0.001, weight_decay=0.0001)
        gene_preprocess = 'hvg'
        n_genes=3000
        
        self.train_writer = SummaryWriter(log_dir+'_train')
        self.valid_writer = SummaryWriter(log_dir+'_valid')
        self.device = device
        self.args = args
        self.w_g2i = args.w_g2i       
        self.w_s2i = args.w_s2i       
        self.w_g2g = args.w_g2g       
        self.w_s2s = args.w_s2s       
        self.w_s2g = args.w_s2g       
        self.w_i2i = args.w_i2i       
        self.w_recon = args.w_recon       
        self.w_graph_loss = args.w_graph_loss      

        if args.dataset == "SpatialLIBD":
            adata = load_ST_file(os.path.join(args.path, args.name))
            df_meta = pd.read_csv(os.path.join(args.path, args.name, 'metadata.tsv'), sep='\t')
            try:
                self.label = pd.Categorical(df_meta['layer_guess']).codes
            except:
                self.label = pd.Categorical(df_meta['ground_truth']).codes
                
            self.adj_2d = calculate_adj_matrix(x=adata.obs["array_row"].tolist(), y=adata.obs["array_col"].tolist(), histology=False)
        elif args.dataset == "MouseBrain" or args.dataset=="BreastCancer":
            adata = load_ST_file(os.path.join(args.path, args.name))
            df_meta = pd.read_csv(os.path.join(args.path, args.name, 'metadata.tsv'), sep='\t')
            try:
                self.label = pd.Categorical(df_meta['layer_guess']).codes
            except:
                self.label = pd.Categorical(df_meta['ground_truth']).codes
            # image
            full_image = cv2.imread(os.path.join(args.path, args.name, './spatial/tissue_hires_image.png'))
            full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
            with open(os.path.join(args.path, args.name, './spatial/scalefactors_json.json'), 'r') as fp:
                scale = json.load(fp)['tissue_hires_scalef']
            patches = []
            for x, y in adata.obsm['spatial']:
                x = int(round(x*scale))
                y = int(round(y*scale))
                
                patches.append(full_image[y-img_size:y+img_size, x-img_size:x+img_size])
            patches = np.array(patches)
            self.image = patches
            
        elif args.dataset == "MouseOlfactoryBulb":
            root_path = os.path.join(args.path, args.name)
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
        if gene_preprocess == 'pca':
            self.gene = adata_preprocess_pca(adata, pca_n_comps=n_genes).astype(np.float32)
        elif gene_preprocess == 'hvg':
            self.gene = adata_preprocess_hvg(adata, n_top_genes=n_genes)
        self.sample_id = adata.obs.index.tolist()
        self.is_gat = is_gat

        augmenter = GeneExpressionAugmenter(prob_mask=0.3, prob_noise=0.3, sigma_noise=0.05, prob_swap=0.3, pct_swap=0.1)
        self.adata1, self.adata2 = augmenter.augment(adata)
        self.coor = pd.DataFrame(adata.obsm['spatial'])
        self.coor.columns = ['imagerow', 'imagecol']
        self.spatial=adata.obsm['spatial']
        self.coor.index = adata.obs.index
    def eval_mclust_refined_ari(self, label, z):
        if z.shape[0] < 1000:
            num_nbs = 4
        else:
            num_nbs = 24
        ari = eval_mclust_ari(label, z, self.n_clusters)
        refined_preds = refine(sample_id=self.sample_id, pred=ari, dis=self.adj_2d, num_nbs=num_nbs)
        ari = adjusted_rand_score(label, refined_preds)
        return ari
    
    def train_augmented_graph1(self, trainloader, epoch):
        scheduler = lr_scheduler.ExponentialLR(self.optimizer,gamma = 0.95)
        with tqdm(total=len(trainloader)) as t:

            adata1 = self.adata1
            adata2 = self.adata2
            Cal_Spatial_Net1(adata1, self.coor)
            self.graph_su = Transfer_pytorch_Data_label(adata1, label=self.label, select_X=self.gene)
            Cal_Spatial_Net1(adata2, self.coor)
            self.graph_sv = Transfer_pytorch_Data_label(adata2, label=self.label, select_X=self.gene)
            return self.graph_su, self.graph_sv

    def train(self, trainloader, epoch, train_graph):
        scheduler = lr_scheduler.ExponentialLR(self.optimizer,gamma = 0.95)
        with tqdm(total=len(trainloader)) as t:
            self.network.train()
            train_loss = 0
            train_loss_g2g = 0
            train_loss_g2i = 0
            train_loss_i2i = 0
            train_loss_graph = 0
            train_loss_recon = 0
            train_cnt = 0
            if self.is_gat:
                for param in self.network.gene_gat.parameters():
                    param.requires_grad = True  
                for i in range(100):
                    self.optimizer_graph.zero_grad()
                    zg_all, out_all, hg_all = self.network.forward_gene_graph(train_graph)
                    graph_loss = F.mse_loss(train_graph.x, out_all)
                    if graph_loss > 100:
                        print(train_graph.x)
                        print(out_all)
                    graph_loss.backward()
                    gradient_clipping = 5.0
                    torch.nn.utils.clip_grad_norm_(self.network.gene_gat.parameters(), gradient_clipping)
                    self.optimizer_graph.step()
                for param in self.network.gene_gat.parameters():
                    param.requires_grad = False  
                with torch.no_grad():
                    zg_all, out_all, hg_all = self.network.forward_gene_graph(train_graph)
            for i, batch in enumerate(trainloader):
                i2i_loss= None
                t.set_description(f'Epoch {epoch} train')
                self.optimizer.zero_grad()
                xg, xg_u, xg_v, xi_u, xi_v, spatial, y, idx = batch
                xg = xg.to(self.device)
                xg_u = xg_u.to(self.device)
                xg_v = xg_v.to(self.device)
                if min(xi_u.shape) != 0:
                    xi_u = xi_u.to(self.device)
                    xi_v = xi_v.to(self.device)
                spatial = spatial.to(self.device)
                idx = idx.to(self.device)
                criterion = NT_Xent(xg.shape[0])
                if not (xi_u==1).all():
                    xg, xi_a, xi_b, lam = mixup_data(xg, xi_u)
                if self.is_gat:
                    zg_all, out_all, hg_all = self.network.forward_gene_graph(train_graph)
                    zg = zg_all[idx]
                    hg = hg_all[idx]
                    graph_su, graph_sv = self.train_augmented_graph1(trainloader, epoch+1)
                    graph_su = graph_su.to(self.device)
                    graph_sv = graph_sv.to(self.device)
                    zs_uall, out_u, hs_uall = self.network.forward_gene_graph(graph_su)
                    zs_vall, out_v, hs_vall = self.network.forward_gene_graph(graph_sv)
                    zs_u = zs_uall[idx]
                    hs_u = hs_uall[idx]
                    zg1, hg1 = self.network.forward_gene(xg)

                else:
                    zg, hg = self.network.forward_gene(xg)
                    graph_loss = 0
                if not (xi_u==1).all():
                    zi_u, hi_u = self.network.forward_image(xi_u, spatial)
                    zi_a, hi_a = self.network.forward_image(xi_a, spatial)
                    zi_b, hi_b = self.network.forward_image(xi_b, spatial)
                s2i_loss = mixup_criterion(criterion, hs_u, hi_a, hi_b, lam) * self.w_s2i
                g2i_loss = mixup_criterion(criterion, hg1, hi_a, hi_b, lam) * self.w_g2i
                if not self.is_gat:
                    xg_u, xg_a, xg_b, lam = mixup_data(xg_u, xg_v)
                    zg_u, hg_u = self.network.forward_gene(xg_u)
                    zg_a, hg_a = self.network.forward_gene(xg_a)
                    zg_b, hg_b = self.network.forward_gene(xg_b)
                else:
                    xg_u, xg_a, xg_b, lam = mixup_data(xg_u, xg_v)
                    zg_u, hg_u = self.network.forward_gene(xg_u)
                    zg_a, hg_a = self.network.forward_gene(xg_a)
                    zg_b, hg_b = self.network.forward_gene(xg_b)
                    graph_su_data, graph_sa_data, graph_sb_data, lam = mixup_data(graph_su.x, graph_sv.x)
                    # 创建 AnnData 对象
                    graph_sa_adata = sc.AnnData(X=graph_sa_data.cpu().numpy())  # 将张量转换为 NumPy 数组并赋给 X
                    graph_sa_adata.obsm['spatial'] = self.spatial  # 存储空间坐标到 obsm，形状为 [num_nodes, 2]
                    Cal_Spatial_Net1(graph_sa_adata, self.coor)
                    graph_sa = Transfer_pytorch_Data1(graph_sa_adata)
                    graph_sa = graph_sa.to(self.device)
                    graph_sb_adata = sc.AnnData(X=graph_sb_data.cpu().numpy())  # 将张量转换为 NumPy 数组并赋给 X
                    graph_sb_adata.obsm['spatial'] = self.spatial  # 存储空间坐标到 obsm，形状为 [num_nodes, 2]
                    Cal_Spatial_Net1(graph_sb_adata, self.coor)
                    graph_sb = Transfer_pytorch_Data1(graph_sb_adata)
                    graph_sb = graph_sb.to(self.device)
                    # 如果需要，可以设置 obs（例如，用于存储标签或其他元数据）
                    zs_aall, out_a, hs_aall = self.network.forward_gene_graph(graph_sa)
                    zs_ball, out_b, hs_ball = self.network.forward_gene_graph(graph_sb)
                    zs_a = zs_aall[idx]
                    hs_a = hs_aall[idx]
                    zs_b = zs_ball[idx]
                    hs_b = hs_ball[idx]
                g2g_loss = mixup_criterion(criterion, hg_u, hg_a, hg_b, lam) * self.w_g2g
                s2g_loss = mixup_criterion(criterion, hs_u, hg_a, hg_b, lam) * self.w_s2g
                s2s_loss = mixup_criterion(criterion, hs_u, hs_a, hs_b, lam) * self.w_s2s

                if not (xi_u==1).all():
                    zi_c, hi_c = self.network.forward_image(xi_v, spatial)
                    i2i_loss = criterion(hi_a, hi_c) * self.w_i2i + criterion(hi_a, hi_b) * self.w_i2i +criterion(hi_a, hi_u) * self.w_i2i
                if not self.is_gat:
                    recon_loss = self.network.recon_loss(zg, xg) * self.w_recon
                else:                   
                    recon_loss =  F.mse_loss(train_graph.x[idx], out_all[idx]) * self.w_recon
                if self.is_gat:
                    if not (xi_u==1).all():
                        loss = g2i_loss*0.1 + i2i_loss*0.1 + g2g_loss*0.1 + s2i_loss*0.1 + s2g_loss*0.1 + s2s_loss*0.1 + recon_loss
                    else:
                        loss = g2g_loss*0.1 + s2g_loss*0.1 + s2s_loss*0.1 + recon_loss
                else:
                    if not (xi_u==1).all():
                        loss = g2i_loss + g2g_loss + i2i_loss 
                    else:
                        loss = g2g_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), gradient_clipping)
                self.optimizer.step()
               
                train_cnt += 1
                train_loss += loss.item()
                train_loss_g2g += g2g_loss.item()
                if not (xi_u==1).all():
                    train_loss_g2i += g2i_loss.item()
                    train_loss_i2i += i2i_loss.item()
                if not self.is_gat:
                    train_loss_g2g += g2g_loss.item()
                train_loss_recon += recon_loss.item()
                
                if self.is_gat:
                    if not (xi_u==1).all():
                        t.set_postfix(loss=f'{(train_loss/train_cnt):.3f}', 
                                      g2i_loss=f'{g2i_loss.item():.3f}', 
                                      g2g_loss=f'{g2g_loss.item():.3f}',
                                      i2i_loss=f'{i2i_loss.item():.3f}', 
                                      recon_loss=f'{recon_loss.item():.3f}',
                                      )
                    else:
                        t.set_postfix(loss=f'{(train_loss/train_cnt):.3f}', 
                                      g2g_loss=f'{g2g_loss.item():.3f}',
                                      recon_loss=f'{recon_loss.item():.3f}',
                                      )
                else:
                    if not (xi_u==1).all():
                        t.set_postfix(loss=f'{(train_loss/train_cnt):.3f}', 
                                  g2i_loss=f'{g2i_loss.item():.3f}', 
                                  g2g_loss=f'{g2g_loss.item():.3f}',
                                  i2i_loss=f'{i2i_loss.item():.3f}', 
                                  recon_loss=f'{recon_loss.item():.3f}')
                    else:
                        t.set_postfix(loss=f'{(train_loss/train_cnt):.3f}', 
                                  g2g_loss=f'{g2g_loss.item():.3f}',
                                  recon_loss=f'{recon_loss.item():.3f}')
                t.update(1)

            self.train_writer.add_scalar('loss', (train_loss/train_cnt), epoch)
            
            self.train_writer.add_scalar('loss_g2i', (train_loss_g2i/train_cnt), epoch)
            self.train_writer.add_scalar('loss_i2i', (train_loss_i2i/train_cnt), epoch)
            self.train_writer.add_scalar('loss_g2g', (graph_loss), epoch)
            self.train_writer.add_scalar('loss_recon', (train_loss_recon/train_cnt), epoch)
            self.train_writer.flush()
            scheduler.step()

    def train_image(self, trainloader, epoch):
        with tqdm(total=len(trainloader)) as t:
            self.network.train()
            train_loss_i2i = 0
            train_cnt = 0

            for i, batch in enumerate(trainloader):
                t.set_description(f'Epoch {epoch} train')
                self.optimizer.zero_grad()
                xg, xg_u, xg_v, xi_u, xi_v, spatial, y, idx = batch
                xg = xg.to(self.device)
                xg_u = xg_u.to(self.device)
                xg_v = xg_v.to(self.device)
                xi_u = xi_u.to(self.device)
                xi_v = xi_v.to(self.device)
                spatial = spatial.to(self.device)
                idx = idx.to(self.device)
                criterion = NT_Xent(xg.shape[0])
                xg, xi_a, xi_b, lam = mixup_data(xg, xi_u)
                zi_u, hi_u = self.network.forward_image(xi_u, spatial)
                zi_a, hi_a = self.network.forward_image(xi_a, spatial)
                zi_b, hi_b = self.network.forward_image(xi_b, spatial)
                zi_c, hi_c = self.network.forward_image(xi_v, spatial)
                i2i_loss = criterion(hi_a, hi_c) * self.w_i2i + criterion(hi_a, hi_b) * self.w_i2i +criterion(hi_a, hi_u) * self.w_i2i
                i2i_loss.backward()
                self.optimizer.step()
                train_cnt += 1
                train_loss_i2i += i2i_loss.item()
                t.set_postfix(loss=f'{(train_loss_i2i/train_cnt):.3f}'
                                  )
                t.update(1)
            self.train_writer.add_scalar('loss_i2i_pre', (train_loss_i2i/train_cnt), epoch)
            self.train_writer.flush()

    def train_graph(self, trainloader, epoch, tarin_graph):
        self.network.train()
        scheduler1 = lr_scheduler.ExponentialLR(self.optimizer_graph,gamma = 0.999)
        self.optimizer_graph.zero_grad()
        z, out, _ = self.network.gene_gat(tarin_graph.x, tarin_graph.edge_index)
        loss_graph = F.mse_loss(tarin_graph.x, out)
        loss_graph.backward()
        torch.nn.utils.clip_grad_norm_(self.network.gene_gat.parameters(), 5.)
        self.optimizer_graph.step()
        self.train_writer.add_scalar('loss_graph_pre', loss_graph, epoch)
        self.train_writer.flush()
            

    def valid(self, validloader, test_graph, epoch=0):
        Xg = []
        Xg1 = []
        Xi = []
        Y = []
        test_graph = test_graph.to(self.device)
        with torch.no_grad():
            with tqdm(total=len(validloader)) as t:
                self.network.eval()
                valid_loss = 0
                valid_cnt = 0
                zg_all, out_all, hg_all = self.network.forward_gene_graph(test_graph)
                for i, batch in enumerate(validloader):
                    xg, xi, spatial, y, idx = batch
                    xg = xg.to(self.device)
                    xi = xi.to(self.device)
                    spatial = spatial.to(self.device)
                    idx = idx.to(self.device)
                    if self.is_gat:
                        xg1, hg1 = self.network.forward_gene(xg)
                        xg = zg_all[idx]
                        hg = hg_all[idx]
                        if not (xi==1).all():
                            xi, hi = self.network.forward_image(xi, spatial)
                    else:
                        if not (xi==1).all():
                            xg, xi, hg, hi = self.network(xg, xi, spatial, is_graph=False)
                    if not (xi==1).all():
                        criterion = NT_Xent(xg.shape[0])
                        loss = criterion(hg, hi)
                    if not (xi==1).all():
                        valid_loss += loss.item()
                    else:
                        valid_loss = 0
                    valid_cnt += 1
                    Xg1.append(xg1.detach().cpu().numpy())
                    Xg.append(xg.detach().cpu().numpy())
                    if not (xi==1).all():
                        Xi.append(xi.detach().cpu().numpy())
                    Y.append(y)
                    t.set_postfix(loss=f'{(valid_loss/valid_cnt):.3f}')
                    t.update(1)
        
                Xg = np.vstack(Xg)
                Xg1 = np.vstack(Xg1)
                
                if not (xi==1).all():
                    Xi = np.vstack(Xi)
                Y = np.concatenate(Y, 0)
        if not (xi==1).all():
            return Xg, Xg1, Xi, Y
        else:
            return Xg, Xg1, None, Y

    def fit(self, trainloader, epochs, tarin_graph, testloader, test_graph, dataset, name, path, adata):
        self.network = self.network.to(self.device)
        train_graph = tarin_graph.to(self.device)
        test_graph = test_graph.to(self.device)
        if self.is_gat:
            for epoch in tqdm(range(epochs*0)):
                self.train_graph(trainloader, epoch+1, tarin_graph)
                if epoch % 1000 == 999:
                    xg, _, xi, label = self.valid(testloader, tarin_graph)
                    z = xg
                    ari, pred_label=get_predicted_results(dataset, name, path, z, adata)
                    self.train_writer.add_scalar('Ari value(pre xg)', ari, epoch)
        for epoch in range(epochs):
            self.train(trainloader, epoch+1, train_graph)
            if epoch % 10 == 0:
                self.save_model('last.pth')
                xg, xg1, xi, label = self.valid(testloader, test_graph)
                if xi is None:
                    combined = np.concatenate([xg,0.5*xg1],axis=1)
                    z = xg + 0.5*xg1
                else:    
                    combined = np.concatenate([xg,0.5*xg1,0.1*xi],axis=1)
                    z = xg + 0.5*xg1 + 0.1*xi  
                ari, pred_label=get_predicted_results(dataset, name, path, z, adata)
                self.train_writer.add_scalar('Ari value(z)', ari, epoch)
                print("Ari value(z) : ", ari)
                ari, pred_label=get_predicted_results(dataset, name, path,  xg, adata)
                self.train_writer.add_scalar('Ari value(xg)', ari, epoch)
                print("Ari value(xg) : ", ari)
                ari, pred_label=get_predicted_results(dataset, name, path,  xg1, adata)
                self.train_writer.add_scalar('Ari value(xg1)', ari, epoch)
                print("Ari value(xg1) : ", ari)
                if xi is not None:
                    ari, pred_label=get_predicted_results(dataset, name, path,  xi, adata)
                    self.train_writer.add_scalar('Ari value(xi)', ari, epoch)
                    print("Ari value(xi) : ", ari)
                np.save(f'embeddings/{name}_xg.npy', xg)
                np.save(f'embeddings/{name}_xg1.npy', xg1)
                np.save(f'embeddings/{name}_xi.npy', xi)

    def get_embeddings(self, validloader, save_name):
        xg, xi, _  = self.valid(validloader)
        np.save(os.path.join('preds', f'{save_name}_xg.npy'), xg)
        np.save(os.path.join('preds', f'{save_name}_xi.npy'), xi)

    def encode(self, batch):
        xg, xi, spatial, y, _ = batch
        xg = xg.to(self.device)
        xi = xi.to(self.device)
        spatial = spatial.to(self.device)
        xg, xi, hg, hi = self.network(xg, xi, spatial)
        return xg + 0.1 * xi
    
    def save_model(self, ckpt_path):
        torch.save(self.network.state_dict(), ckpt_path)

    def load_model(self, ckpt_path):
        self.network.load_state_dict(torch.load(ckpt_path))
        self.network = self.network.to(self.device)

