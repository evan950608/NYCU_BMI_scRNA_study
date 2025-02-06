import os
import sys
sys.path.append('/home/evanlee/PBMC_Hao')
from ADlasso2 import AD2_w_utils_para as ad

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

# os.chdir('/Users/evanli/Documents/Research_datasets/PBMC_Hao')
# Read representative cells
adata = sc.read('/home/evanlee/PBMC_Hao/Level_1_alpha0.01/Hao_PBMC_level1_rep_cells.h5ad')
print(adata.shape)  # (32349, 20568)

label = adata.obs['celltype.l1'].tolist()
types = np.unique(label).tolist()
print(types)

# %%
def pipeline_ad2(data, celltype, label, alpha, output_path='', tuning_result=''):
    print('====================')
    print('Starting job for {}'.format(celltype))
    # Binary classification of a celltype
    celltype_label = [1 if x == celltype else 0 for x in label]
    # create index for a celltype
    celltype_indices = [idx for idx, label in enumerate(celltype_label) if label == 1]

    if tuning_result:
        os.chdir(tuning_result)  # /home/evanlee/Pancreas_AD2/Pancreas_result
        with open(f'./{celltype}/{celltype}_tuning.json') as f:
            print('Loading tuning result for {}'.format(celltype))
            result_dict = json.load(f)
            for key in result_dict.keys():
                result_dict[key] = np.array(result_dict[key])
    else:
        # a list of lambdas to test
        log_lmbd_range = np.linspace(np.log(1e-4), np.log(1), 25)
        lmbd_range = np.exp(log_lmbd_range)
        # Lambda tuning (can modify learning rate alpha)
        result_dict = ad.lambda_tuning_parallel(data.X, celltype_label, lmbd_range, alpha=alpha, device='cpu', n_jobs=25)

        # Export lambda tuning results as json
        os.chdir(output_path)
        output = dict()
        for key in result_dict.keys():
            output[key] = result_dict[key].tolist()
        with open('{}_tuning.json'.format(celltype), 'w') as f:
            json.dump(output, f)
    
        # Plot lambda tuning results
        Fig = ad.lambda_tuning_viz(result_dict, 'Feature_number', savepath='{}_feature_number.png'.format(celltype))
        Fig = ad.lambda_tuning_viz(result_dict, 'AUC', savepath='{}_AUC.png'.format(celltype))
        Fig = ad.lambda_tuning_viz(result_dict, 'loss_history', savepath='{}_loss_history.png'.format(celltype))
        Fig = ad.lambda_tuning_viz(result_dict, 'error_history', savepath='{}_error_history.png'.format(celltype))

    # Lambda decision k = 2
    os.chdir(output_path)
    opt_lmbd, fig = ad.lambda_decision(result_dict, k=2, savepath='{}_lambda_decision.png'.format(celltype))
    print('Optimal lambda: {}'.format(opt_lmbd))
    with open('{}_opt_lambda.txt'.format(celltype), 'w') as f:
        f.write(str(opt_lmbd) + '\n')

    return



# %%

for celltype in types:
    st = time.time()
    # set learning rate alpha to 0.001
    pipeline_ad2(adata, celltype, label, alpha=0.001, output_path='/home/evanlee/PBMC_Hao/Level_1_alpha0.001/Level1_alpha0.001_result')
    et = time.time()
    print('{} Time elapsed: {} minutes.'.format(celltype, (et-st)/60))

print('***** Finished lambda tuning')
print('====================')
print('***** Starting feature selection')


# %% Feature selection with optimal lambda
def pipeline_feature_selection(data, celltype, label, opt_lmbd, alpha, output_path=''):
    print('====================')
    print('Starting job for {}'.format(celltype))
    # Binary classification of a celltype
    celltype_label = [1 if x == celltype else 0 for x in label]
    # create index for a celltype
    celltype_indices = [idx for idx, label in enumerate(celltype_label) if label == 1]

    # Find marker genes with optimal lambda
    pvl = ad.get_prevalence(data.X, celltype_indices)
    print('Fitting with optimal lambda:', opt_lmbd)
    # Fit ADlasso2 (can modify learning rate alpha)
    opt_res = ad.ADlasso2(lmbd=opt_lmbd, alpha=alpha, echo=True, device='cpu')
    opt_res.fit(data.X, celltype_label, pvl)
    
    # Export selection results
    os.chdir(output_path)
    opt_res.writeList(outpath=output_path+f'/{celltype}_features.txt', featureNameList=data.var_names)
    print(f'{celltype} feature list exported.')

    # Ouput description
    description = f'''Optimal lambda: {opt_lmbd}
    median of selected prevalence: {np.median([pvl[i]  for i, w in enumerate(opt_res.feature_set) if w != 0])}
    total selected feature: {np.sum(opt_res.feature_set)}\n'''
    print('---Selection result for {}'.format(celltype))
    print(description)

    with open(f'{celltype}_description.txt', 'w') as f:
        f.write(description)


# %% Multi-processing
import multiprocessing as mp

# Define a function to execute pipeline_feature_selection for a single cell type
def run_pipeline_feature_selection(celltype):
    # read optimal lambda
    os.chdir('/home/evanlee/PBMC_Hao/Level_1_alpha0.001/Level1_alpha0.001_result')
    with open(f'./{celltype}_opt_lambda.txt', 'r') as f:
        opt_lmbd = float(f.read())
    print(celltype, 'Optimal lambda:', opt_lmbd, '\n')

    # set learning rate alpha to 0.001
    pipeline_feature_selection(adata, celltype, label, opt_lmbd, alpha=0.001, output_path='/home/evanlee/PBMC_Hao/Level_1_alpha0.001/Level1_alpha0.001_result')

# Create a pool of worker processes
with mp.Pool(processes=len(types)) as pool:
    # Map the run_pipeline_feature_selection function to each cell type using the pool of workers
    pool.map(run_pipeline_feature_selection, types)


    