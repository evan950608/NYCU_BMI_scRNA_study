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
import multiprocessing as mp


# %% Feature selection with optimal lambda
def pipeline_feature_selection(data, celltype, label, opt_lmbd, output_path=''):
    print('====================')
    print('Starting job for {}'.format(celltype))
    st = time.time()

    # Binary classification of a celltype
    celltype_label = [1 if x == celltype else 0 for x in label]
    # create index for a celltype
    celltype_indices = [idx for idx, label in enumerate(celltype_label) if label == 1]

    # Find marker genes with optimal lambda
    pvl = ad.get_prevalence(data.X, celltype_indices)
    print('Fitting with optimal lambda:', opt_lmbd)
    # Set loss_difference_tol to 1e-6
    opt_res = ad.ADlasso2(lmbd=opt_lmbd, loss_tol=1e-6, echo=True, device='cuda')  # cuda
    opt_res.fit(data.X, celltype_label, pvl)
    
    # Export selection results
    os.chdir(output_path)
    opt_res.writeList(outpath=output_path+f'/{celltype}_features.txt', featureNameList=data.var_names)
    print(f'{celltype} feature list exported.')

    et = time.time()
    elapsed = (et-st)/60
    print('Done')
    print('stop iteration:', opt_res.stop_iter)

    # Output loss history as json
    loss_history = opt_res.loss_history
    with open(f'{celltype}_loss_history.json', 'w') as f:
        json.dump(loss_history, f)
    # Output loss difference history
    loss_diff_history = opt_res.loss_diff_history
    with open(f'{celltype}_loss_diff_history.json', 'w') as f:
        json.dump(loss_diff_history, f)
    
    # Ouput description
    description = f'''Optimal lambda: {opt_lmbd}
    median of selected prevalence: {np.median([pvl[i]  for i, w in enumerate(opt_res.feature_set) if w != 0])}
    minimal loss: {opt_res.loss_}
    minimal weight diff: {opt_res.convergence_}
    total selected feature: {np.sum(opt_res.feature_set)}
    Time elapsed: {elapsed}\n'''
    print('---Selection result for {}'.format(celltype))
    print(description)

    with open(f'{celltype}_description.txt', 'w') as f:
        f.write(description)
    
    # Export loss_history as json
    with open(f'{celltype}_loss_history.json', 'w') as f:
        json.dump(opt_res.loss_history, f)
    
    # Export loss_diff_history as json
    with open(f'{celltype}_loss_diff_history.json', 'w') as f:
        json.dump(opt_res.loss_diff_history, f)


# %% Multi-processing to run feature selection

# Define a function to execute pipeline_feature_selection for a single cell type
def run_pipeline_feature_selection(celltype):
    st = time.time()
    # Read adata
    adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/GSE164378/Hao_PBMC_GSE_level1_rep_cells_onlyLog1p.h5ad')
    print('Original adata:', adata.shape)  # (32349, 20568)

    ### Remove the genes whose expression is zero in all cells of this celltype
    adata_celltype = adata[adata.obs['celltype.l1'] == celltype]
    print('adata celltype shape:', adata_celltype.shape)

    # Remove explicit zeros from the sparse matrix
    adata_celltype.X.eliminate_zeros()

    # Find the columns that are all zeros
    all_zeros = np.where(adata_celltype.X.getnnz(axis=0) == 0)[0]

    # Remove the columns that are all zeros from the anndata object
    adata = adata[:, ~adata_celltype.var_names.isin(adata_celltype.var_names[all_zeros])]
    print('adata shape after removing all zero columns for celltype cells:', adata.shape)
    del adata_celltype, all_zeros

    label = adata.obs['celltype.l1'].tolist()

    # optimal lambda for CD4 T cells (weight difference tuning)
    opt_lmbd = 0.0014677992676220694
    server_fractal_path = '/home/jovyan/work/GitHub/EvanPys/Progress/PBMC_Hao_GSE/L1_by_lossdiff/CD4T_iter_1000'
    pipeline_feature_selection(adata, celltype, label, opt_lmbd, output_path=server_fractal_path)

    et = time.time()
    print(f'Elapsed time for {celltype}: {(et-st)/60:.2f} minutes')


# %% Main code
run_pipeline_feature_selection('CD4_T')



    