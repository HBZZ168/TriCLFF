# Title
TriCLFF: A Multi-modality feature fusion framework using contrastive learning for spatial domain identification

# Overview
TriCLFF is a contrastive learning framework to learn low-dimensional fusion embeddings of gene expression profiles and histological images. We first utilize three different kinds of encoders (i.e., graph autoencoder, multi-layer perception and Swin Transformer) to extract embeddings for spatial associations, gene expression levels and histological morphological features individually and then integrate them with a weighted sum method. To better capture the fusion features, we use contrastive learning strategy in both single-modality and cross-modality levels, which introducing six contrastive learning losses. Combined with GMM clustering method, our proposed TriCLFF framework can significantly improve the spatial domain identification accuracy and further benefit to other downstream analysis, such as trajectory inference and UMAP visualization. The implement details of TriCLFF are described in the following sections.

![TriCLFF method](TriCLFF.bmp)

# Requirements
To run TriCLFF, please ensure that all the libraries below are successfully installed:
- torch 2.0.0
- CUDA Version 11.7
- scanpy 1.8.1
- mclust

You can set up conda environment for TriCLFF：
- conda env create -n TriCLFF-env python=3.8.10
- conda activate TriCLFF-env
- pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
- pip install torch_scatter==2.1.1 torch_sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.0.0%2Bcu117.html
- pip install torch-geometric==2.3.0
If you need more details on the dependences, look at the requirements.txt file.
Then you can run:
- pip install -r requirements.txt

# Run TRiCLFF on the example data.
Specify specific sample data in the train.py file, you can first run: 
- python train.py
# output
- Then run DLPFC_151672.ipynb or MouseBrain.ipynb.
- The clustering labels will be stored in the dir `output` /dataname_pred.csv. 
# Datasets
- The spatial transcriptomics datasets that support the findings of this study are available here:
- The DLPFC dataset is accessible within the spatialLIBD package (http://spatial.libd.org/spatialLIBD). 
- The MouseBrain dataset and human breast cancer data can be obtained at https://zenodo.org/record/6925603#.YuM5WXZBwuU. 
- The processed Stereo-seq data from mouse olfactory bulb tissue is accessible at https://github.com/JinmiaoChenLab/SEDR_analyses.
# Contact
Feel free to submit an issue or contact us at qhjiang@hit.edu.cn for problems about the tool.
# Citation
Please cite our paper:
```
@article{pangfl,
  title={TriCLFF: A Multi-modality feature fusion framework using contrastive learning for spatial domain identification },
  author={Fenglan Pang1#, Guangfu Xue1#, Wenyi Yang1#, Yideng Cai1, Jinhao Que1, Haoxiu Sun2, Pingping Wang2, Shuaiyu Su1, 
  Xiyun Jin2, Qian Ding1, Zuxiang Wang2, Meng Luo1, Yuexin Yang1, Yi Lin2, Renjie Tan2, Yusong Liu2*, Zhaochun Xu2*, Qinghua Jiang1,2*},
  journal={Briefings in Bioinformatics},
  year={2025}
 publisher={Jiang Qinghua Laboratory, Harbin Institute of Technology}
}
```

