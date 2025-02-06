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
    opt_res.writeList(outpath=output_path+f'/{celltype}_{opt_lmbd}_features.txt', featureNameList=data.var_names)
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

    with open(f'{celltype}_{opt_lmbd}_description.txt', 'w') as f:
        f.write(description)
    
    # Export loss_history as json
    with open(f'{celltype}_{opt_lmbd}_loss_history.json', 'w') as f:
        json.dump(opt_res.loss_history, f)
    
    # Export loss_diff_history as json
    with open(f'{celltype}_{opt_lmbd}_loss_diff_history.json', 'w') as f:
        json.dump(opt_res.loss_diff_history, f)


# %% 

# Define a function to execute pipeline_feature_selection for a single cell type
def run_pipeline_feature_selection_at_lambda(celltype, set_lambda):
    st = time.time()
    # Read adata
    # adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/PBMC_Hao/GSE164378_Hao/Harmony_noZ/Hao_L1_repcells_loginv_Harmony_noZ.h5ad')
    adata = sc.read_h5ad(dataset_dir / 'PBMC_Hao/GSE164378_Hao/Harmony_noZ/Hao_L1_repcells_loginv_Harmony_noZ.h5ad')
    print('Original adata:', adata.shape)
    
    # L1 celltype as labels
    label = adata.obs['celltype.l1'].tolist()

    print('Set lambda:', set_lambda)

    # server_fractal_path = '/home/jovyan/work/GitHub/EvanPys/Progress/PBMC_Hao_batch_noZ/Level1/large_lambda_various/at_various_number/added_features_at_various_no'
    server_fractal_path = source_code_dir / 'PBMC_Hao_batch_noZ/Level1/large_lambda_various/at_various_number/added_features_at_various_no'
    pipeline_feature_selection(adata, celltype, label, set_lambda, output_path=server_fractal_path)

    et = time.time()
    print(f'Elapsed time for {celltype}: {(et-st)/60:.2f} minutes')
    del adata


def get_lambdas_for_celltype(idx, tuning_filename):
    def convert_lists_to_arrays(dictionary):
        for key in dictionary:
            dictionary[key] = np.array(dictionary[key])
        return dictionary

    with open(tuning_filename) as f:
        tuning = json.load(f)
    tuning = convert_lists_to_arrays(tuning)

    queue_lambda = tuning['log_lambda_range'][idx]
    queue_lambda = np.exp(queue_lambda)
    return queue_lambda
  

# %% Main code
# types = ['B', 'CD4_T', 'CD8_T', 'DC', 'Mono', 'NK', 'other', 'other_T']
# adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/PBMC_Hao/GSE164378_Hao/Harmony_noZ/Hao_L1_repcells_loginv_Harmony_noZ.h5ad')
adata = sc.read_h5ad(dataset_dir / 'PBMC_Hao/GSE164378_Hao/Harmony_noZ/Hao_L1_repcells_loginv_Harmony_noZ.h5ad')
print('Original adata:', adata.shape)
label = adata.obs['celltype.l1'].tolist()
types = np.unique(label).tolist()
print('all cell types:', types)
print('====================')
del adata

# queue = types
queue = ['CD4_T', 'DC', 'Mono', 'NK', 'CD8_T', 'B']
print('Queue', queue)
idx_dict = {'CD4_T': [15,16,17,20,21], 
            'DC': [16,19,21], 
            'Mono': [17,19,22], 
            'NK': [20,23], 
            'CD8_T': [21], 
            'B': [19]}
new_idx_dict = {'CD4_T': [9, 11, 19],
                 'DC': [9, 11, 13, 15, 17],
                 'Mono': [9, 11, 13, 21],
                 'NK': [9, 11, 13, 15, 19, 21],
                 'CD8_T': [9, 11, 13, 15, 17, 19],
                 'B': [9, 11, 13, 17, 21]}
print('Indices at different celltype:\n', new_idx_dict)

lambda_dict = {}
for celltype, indices in new_idx_dict.items():
    # tuning_filename = os.path.join('/home/jovyan/work/GitHub/EvanPys/Progress/PBMC_Hao_batch_noZ/Level1/tuning_result_cuda', f'{celltype}_tuning.json')
    tuning_filename = os.path.join(source_code_dir / 'PBMC_Hao_batch_noZ/Level1/tuning_result_cuda', f'{celltype}_tuning.json')
    lambda_dict[celltype] = get_lambdas_for_celltype(indices, tuning_filename)
print('lambda_dict:\n', lambda_dict)
print()
# print('force stop')
# sys.exit()

for celltype in queue:
    lambda_list_celltype = lambda_dict[celltype].tolist()
    for set_lmbd in lambda_list_celltype:
        print('====================')
        print(f'JOB: {celltype}, {set_lmbd}')
        run_pipeline_feature_selection_at_lambda(celltype, set_lambda=set_lmbd)
