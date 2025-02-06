import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scanpy as sc


# In[7]:
### Find marker genes
def rank_genes(adata, filepath=''):
    sc.tl.rank_genes_groups(adata, groupby='celltype.l1', n_genes=3000, method='wilcoxon')
    # sc.pl.rank_genes_groups(adata, sharey=False, ncols=2)

    ### Get the top ranked genes for each celltype
    genes_df_dict = {}

    for celltype in types:
        genes_df = sc.get.rank_genes_groups_df(adata, group=celltype)
        genes_df_dict[celltype] = genes_df
        # genes_df.to_csv(f'{celltype}_DEG3000.csv', index=False)
    
    # Export the top ranked genes for each celltype
    # os.chdir('/home/evanlee/PBMC_Hao/Level_1_alpha0.01/Level1_DEG')
    if filepath:
        for celltype in genes_df_dict.keys():
            genes_df_dict[celltype].to_csv(f'{filepath}/{celltype}_DEG3000.csv', index=False)
    
    return genes_df_dict

# In[14]:
### Read ADlasso selected features
# os.chdir('/home/evanlee/PBMC_Hao/Level_1_alpha0.01/Level1_result')
def read_AD_features(filepath):
    feature_df_dict = {}
    for celltype in types:
        feature_df = pd.read_csv(f'{filepath}/{celltype}/{celltype}_features.txt', names=['Gene', 'Weight', 'Tendency'], sep='\t')
        feature_df_dict[celltype] = feature_df

    return feature_df_dict


# In[15]:
def read_DEG(filepath):
    DEG_df_dict = {}
    for celltype in types:
        DEG_df = pd.read_csv(f'{filepath}/{celltype}_DEG3000.csv')
        DEG_df_dict[celltype] = DEG_df

    return DEG_df_dict


# In[27]:
### B cells vs. Rest
# subset adata to only the ADlasso features for B cells
def leiden_AD_features(adata, celltype, ad_features_dict, result_path):
    # celltype = 'B'
    celltype_features = ad_features_dict[celltype]['Gene'].tolist()

    # Leiden UMAP with celltype features
    adata_type_features = adata[:, celltype_features]
    print(f'{celltype} features adata shape:', adata_type_features.shape)

    # Delete the 'neighbors' key from adata.uns
    # if 'neighbors' already exists in adata.uns, error would occurs at sc.pp.neighbors
    # if 'neighbors' in adata_type_features.uns:
    #     print('delete neighbor')
    #     del adata_type_features.uns['neighbors']


    # PCA
    sc.tl.pca(adata_type_features, svd_solver='arpack')

    # neighborhood graph
    sc.pp.neighbors(adata_type_features, n_neighbors=15)

    # UMAP
    sc.tl.umap(adata_type_features)

    # Leiden
    sc.tl.leiden(adata_type_features)
    leiden_clus_no = len(np.unique(adata_type_features.obs['leiden']))
    print(f'Number of Leiden clusters: {leiden_clus_no}')
    # Plot Leiden UMAP
    sc.pl.umap(adata_type_features, color='leiden', title=f'Leiden L1 ({celltype} features)', save=f'{result_path}/{celltype}_features_leiden.png')

    return adata_type_features


# In[36]:
### Plot Leiden UMAP with DEGn (celltype B, n=17)
def leiden_DEGn(adata, celltype, DE_genes_dict, n, result_path):
    # n = len(celltype_features)
    DEGn = DE_genes_dict[celltype]['names'][:n].tolist()

    # filtering the DEGenes in data
    adata_DEGn = adata[:, DEGn]  # choose columns, Genes
    print(f'{celltype} DEGn adata shape:', adata_DEGn.shape)

    # Delete the 'neighbors' key from adata.uns
    # if 'neighbors' already exists in adata.uns, error would occurs at sc.pp.neighbors
    if 'neighbors' in adata_DEGn.uns:
        del adata_DEGn.uns['neighbors']

    # PCA
    sc.tl.pca(adata_DEGn, svd_solver='arpack')

    # neighborhood graph
    sc.pp.neighbors(adata_DEGn, n_neighbors=15)

    # UMAP
    sc.tl.umap(adata_DEGn)

    # Leiden
    sc.tl.leiden(adata_DEGn)
    leiden_clus_no = len(np.unique(adata_DEGn.obs['leiden']))
    print(f'Number of Leiden clusters: {leiden_clus_no}')
    # Plot Leiden UMAP
    sc.pl.umap(adata_DEGn, color='leiden', title=f'Leiden L1 ({celltype} DEG{n})', save=f'{result_path}/{celltype}_DEG{n}_leiden.png')

    return adata_DEGn




# In[55]: Main code
# read in data with complete cells
adata = sc.read('/home/evanlee/PBMC_Hao/Hao_PBMC.h5ad')
print(adata.shape)
# adata = sc.read('/Users/evanli/Documents/Research_datasets/PBMC_Hao/Hao_PBMC.h5ad')

types = np.unique(adata.obs['celltype.l1']).tolist()
# types = ['B', 'CD4 T', 'CD8 T', 'DC', 'Mono', 'NK', 'other', 'other T']
print('cell types:', types)

sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.log1p(adata)
print(adata.shape)  # row is cells, column is gene
# (161764, 20568)

# Read ADlasso features
ad_features_dict = read_AD_features('/home/evanlee/PBMC_Hao/Level_1_alpha0.01/Level1_result')
DE_genes_dict = read_DEG('/home/evanlee/PBMC_Hao/Level_1_alpha0.01/Level1_DEG')

print('====================')
print('AD:', ad_features_dict.keys())
print(ad_features_dict['B'])
print('DEG:', DE_genes_dict.keys())
print(DE_genes_dict['B'])


# In[56]:
# Implement for celltypes
result_path = '/home/evanlee/PBMC_Hao/Level_1_alpha0.01/AD_DEG_result'
celltype = 'B'
adata_type_features = leiden_AD_features(adata, celltype, ad_features_dict, result_path)
n = ad_features_dict[celltype].shape[0]
adata_DEGn = leiden_DEGn(adata, celltype, DE_genes_dict, n, result_path)

print('ADlasso adata shape:', adata_type_features.shape)
print('DEGn adata shape:', adata_DEGn.shape)

b = adata_type_features.var_names
d = adata_DEGn.var_names
inter = b.intersection(d)
print('Intersection of ADlasso features and DEGn:', inter)
