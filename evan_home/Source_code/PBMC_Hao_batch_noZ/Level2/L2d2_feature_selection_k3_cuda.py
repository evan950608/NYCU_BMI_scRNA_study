from pathlib import Path

# Get the current working directory as a Path object
current_path = Path.cwd()
home_folder = 'evan_home'

# Traverse up the directory tree until you find the target folder
for parent in [current_path] + list(current_path.parents):
    if parent.name == home_folder:
        home_path = parent
        break
else:
    raise ValueError(f"Folder '{home_folder}' not found in the current working directory.")

print("Home Path:", home_path)
source_code_dir = home_path / 'Source_code'
dataset_dir = home_path / 'Dataset'

import os
import sys
# sys.path.append('/Users/evanli/Documents/EvanPys/Progress')
# sys.path.append('/home/jovyan/work/GitHub/EvanPys/Progress')
sys.path.append(str(source_code_dir))
from ADlasso2 import AD2_w_utils_lossdiff_noZ as ad

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
    
    opt_res = ad.ADlasso2(lmbd=opt_lmbd, loss_tol=1e-6, echo=True, device='cuda')  # cuda
    opt_res.fit(data.X, celltype_label, pvl)
    
    # Export selection results
    os.chdir(output_path)
    opt_res.writeList(outpath=output_path+f'/{celltype}_features.txt', featureNameList=data.var_names)
    print(f'{celltype} feature list exported.')

    et = time.time()
    elapsed = (et-st)/60
    # print(f'Elapsed time for {celltype}: {elapsed} minutes')

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


# %% 

# Define a function to execute pipeline_feature_selection for a single cell type
def run_pipeline_feature_selection(celltype):
    st = time.time()
    # Read adata
    # adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/PBMC_Hao/GSE164378_Hao/Harmony_noZ/Hao_L2_repcells_loginv_Harmony_noZ.h5ad')
    adata = sc.read_h5ad(dataset_dir / 'PBMC_Hao/GSE164378_Hao/Harmony_noZ/Hao_L2_repcells_loginv_Harmony_noZ.h5ad')
    print('Original adata:', adata.shape)
    
    # L1 celltype as labels
    label = adata.obs['celltype.l2'].tolist()

    ### Read optimal lambda dictionary from json
    # path_opt_lambda = '/home/jovyan/work/GitHub/EvanPys/Progress/PBMC_Hao_batch_noZ/Level2/L2c_k3_opt_lmbd.json'
    path_opt_lambda = source_code_dir / 'PBMC_Hao_batch_noZ/Level2/L2c_k3_opt_lmbd.json'
    with open(path_opt_lambda, 'r') as f:
        opt_lambda_dict = json.load(f)
    opt_lmbd = opt_lambda_dict[celltype]
    print('optimal lambda:', opt_lmbd)

    # server_fractal_path = '/home/jovyan/work/GitHub/EvanPys/Progress/PBMC_Hao_batch_noZ/Level2/feature_selection_k3'
    server_fractal_path = source_code_dir / 'PBMC_Hao_batch_noZ/Level2/feature_selection_k3'
    pipeline_feature_selection(adata, celltype, label, opt_lmbd, output_path=server_fractal_path)

    et = time.time()
    print(f'Elapsed time for {celltype}: {(et-st)/60:.2f} minutes')
    del adata


# %% Main code
# types = ['B', 'CD4_T', 'CD8_T', 'DC', 'Mono', 'NK', 'other', 'other_T']
# adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/PBMC_Hao/GSE164378_Hao/Harmony_noZ/Hao_L2_repcells_loginv_Harmony_noZ.h5ad')
adata = sc.read_h5ad(dataset_dir / 'PBMC_Hao/GSE164378_Hao/Harmony_noZ/Hao_L2_repcells_loginv_Harmony_noZ.h5ad')
print('Original adata:', adata.shape)
label = adata.obs['celltype.l2'].tolist()
types = np.unique(label).tolist()
print('all cell types:', types)
print('====================')
del adata

queue = types
print('Queue', queue)
# print('force stop')
# sys.exit()

for celltype in queue:
    run_pipeline_feature_selection(celltype)
