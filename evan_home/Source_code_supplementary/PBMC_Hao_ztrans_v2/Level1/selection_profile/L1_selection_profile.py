import os
import sys
# sys.path.append('/home/evanlee/PBMC_Hao')
# sys.path.append('/Users/evanli/Documents/EvanPys/Progress')
# sys.path.append('/Users/evanli/Documents/EvanPys/Progress')
sys.path.append('/home/jovyan/work/GitHub/EvanPys/Progress')
from ADlasso2 import AD2_w_utils_lossdiff as ad

import numpy as np
import pandas as pd
from pathlib import Path
import scanpy as sc
import sklearn
from scipy.sparse import csr_matrix
from sklearn.metrics.cluster import adjusted_rand_score
import copy
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def plot_feature_property(adata, raw_count, celltype):
    # # Read data (use only representative cells)
    # adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/PBMC_Hao/GSE164378_Hao/Harmony_and_ztrans/Hao_L1_repcells_loginv_Harmony_ztrans_afterrep.h5ad')
    # print('Original adata:', adata.shape)  # (32349, 20568)

    # ### Remove the genes whose expression is zero in all cells of this celltype
    # adata_celltype = adata[adata.obs['celltype.l1'] == celltype]
    # print('adata celltype shape:', adata_celltype.shape)

    # # Remove explicit zeros from the sparse matrix
    # adata_celltype.X.eliminate_zeros()

    # # Find the columns that are all zeros
    # all_zeros = np.where(adata_celltype.X.getnnz(axis=0) == 0)[0]

    # # Remove the columns that are all zeros from the anndata object
    # adata = adata[:, ~adata_celltype.var_names.isin(adata_celltype.var_names[all_zeros])]
    # print('adata shape after removing all zero columns for celltype cells:', adata.shape)
    # del adata_celltype, all_zeros

    # L1 celltype as labels
    label = adata.obs['celltype.l1'].tolist()

    ### Read optimal lambda dictionary from json
    path_opt_lambda = '/home/jovyan/work/GitHub/EvanPys/Progress/PBMC_Hao_ztrans_v2/Level1/L1c_opt_lmbd_final.json'
    with open(path_opt_lambda, 'r') as f:
        opt_lambda_dict = json.load(f)
    opt_lmbd = opt_lambda_dict[celltype]
    print('optimal lambda:', opt_lmbd)

    # Build ADlasso model
    # Binary classification of a celltype
    celltype_label = [1 if x == celltype else 0 for x in label]
    # create index for a celltype
    celltype_indices = [idx for idx, label in enumerate(celltype_label) if label == 1]

    # Find marker genes with optimal lambda
    # pvl = ad.get_prevalence(raw_count.X, celltype_indices)
    # print('Fitting with optimal lambda:', opt_lmbd)

    # opt_res = ad.ADlasso2(lmbd=opt_lmbd, loss_tol=1e-6, echo=True, device='cuda')  # cuda
    # opt_res.fit(adata.X, celltype_label, pvl)

    # # Read opt_res from pickle
    with open(f'{celltype}_PreL_model.pkl', 'rb') as f:
        opt_res = pickle.load(f)
    print(f'total selected feature: {np.sum(opt_res.feature_set)}')

    # Get feature property
    prop = ad.featureProperty(raw_count.X, celltype_label, opt_res)
    prop['featureID'] = adata.var_names
    print(prop.head())
    # export feature property df
    # prop.to_csv(f'{celltype}_feature_property.csv', index=False)

    # Plot feature property
    # Filter the data
    positive_data = prop[prop['select'] == 'PreLect_positive']
    negative_data = prop[prop['select'] == 'PreLect_negative']
    other_data = prop[prop['select'] == 'No selected']

    # Plot the other dots with grey color and alpha=0.5
    sns.scatterplot(x="prevalence_1", y="prevalence_0", color='#BCBCBC', alpha=0.5, data=other_data, label='Others')
    # Plot the positive dots with red color and alpha=1
    sns.scatterplot(x="prevalence_1", y="prevalence_0", color='r', alpha=1, data=positive_data, label='PreLect_positive')
    # Plot the negative dots with blue color and alpha=1
    sns.scatterplot(x="prevalence_1", y="prevalence_0", color='b', alpha=1, data=negative_data, label='PreLect_negative')

    # Get the current axes
    ax = plt.gca()

    # Get the handles and labels from the scatterplot
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 2, 0]
    # Set the legend
    plt.legend(handles=[handles[idx] for idx in order], labels=[labels[idx] for idx in order], loc='upper left', fontsize='small')
    plt.xlabel('Target prevalence')
    plt.ylabel('Other prevalence')
    plt.title(f'{celltype}: selection profile')
    plt.show()
    plt.savefig(f'{celltype}_selection_profile.png', dpi=300)
    plt.close('all')
    

    del adata
    return opt_res, prop


types = ['B', 'CD4_T', 'CD8_T', 'DC', 'Mono', 'NK', 'other', 'other_T']

# Read data (use only representative cells)
adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/PBMC_Hao/GSE164378_Hao/Harmony_and_ztrans/Hao_L1_repcells_loginv_Harmony_ztrans_afterrep.h5ad')
print('Original adata:', adata.shape)


raw_count_adata = sc.read_h5ad("/home/jovyan/work/Research_datasets/PBMC_Hao/GSE164378_Hao/Hao_PBMC_GSE164378_raw.h5ad")
raw_count_adata_subset = raw_count_adata[adata.obs_names, adata.var_names]
print('Subsetted raw count adata:', raw_count_adata_subset.shape, type(raw_count_adata_subset.X))

for celltype in types:
    st = time.time()
    # test = 'B'
    print('====================')
    print('Starting job for {}'.format(celltype))
    opt_res, prop = plot_feature_property(adata, raw_count_adata_subset, celltype)
    # prop.to_csv(f'{celltype}_feature_property.csv', index=False)

    # Save opt_res as pickle
    # with open(f'{celltype}_PreL_model.pkl', 'wb') as f:
    #     pickle.dump(opt_res, f)

    et = time.time()
    elapsed = (et-st)/60
    print(f'Elapsed time for {celltype}: {elapsed} minutes')
