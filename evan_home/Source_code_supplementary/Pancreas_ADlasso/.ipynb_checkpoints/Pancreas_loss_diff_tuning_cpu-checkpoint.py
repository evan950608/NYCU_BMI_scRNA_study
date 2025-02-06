import os
import sys
# sys.path.append('/Users/evanli/Documents/EvanPys/Progress')
# sys.path.append('/home/evanlee/PBMC_Hao')
sys.path.append('/home/jovyan/work/GitHub/EvanPys/Progress')
from ADlasso2 import AD2_w_utils_lossdiff as ad

import numpy as np
import pandas as pd
from pathlib import Path
import scanpy as sc
import scvelo as scv
import sklearn
from scipy.sparse import csr_matrix
from sklearn.metrics.cluster import adjusted_rand_score
import copy
import json
import time

# %%
# Function to convert NumPy types to Python types
def convert_numpy_types(data):
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    else:
        return data

# %%
def pipeline_ad2(data, celltype, label, output_path='', tuning_result=''):
    print('====================')
    print('Starting job for {}'.format(celltype))
    # Binary classification of a celltype
    celltype_label = [1 if x == celltype else 0 for x in label]
    # create index for a celltype
    celltype_indices = [idx for idx, label in enumerate(celltype_label) if label == 1]

    if tuning_result:
        os.chdir(tuning_result)
        with open(f'./{celltype}/{celltype}_tuning.json') as f:
            print('Loading tuning result for {}'.format(celltype))
            result_dict = json.load(f)
            for key in result_dict.keys():
                result_dict[key] = np.array(result_dict[key])
    else:
        # a list of lambdas to test
        # log_lmbd_range = np.linspace(np.log(1e-4), np.log(1), 25)
        log_lmbd_range = np.linspace(np.log(1e-5), np.log(1e-1), 25)
        lmbd_range = np.exp(log_lmbd_range)

        # Lambda tuning
        # result_dict = ad.lambda_tuning_para_ttsplit(data.X, celltype_label, lmbd_range, alpha=alpha, device='cpu', n_jobs=25)
        # result_DF, all_result_dict, loss_history_dict, loss_diff_history_dict = ad.lambda_tuning_cuda(data.X, celltype_label, lmbd_range, device='cuda', loss_tol=1e-6)
        result_dict, loss_history_dict, loss_diff_history_dict = ad.lambda_tuning_para_ttsplit(data.X, celltype_label, lmbd_range, device='cpu', loss_tol=1e-6, n_jobs=25)

        # Export lambda tuning results dict as json
        os.chdir(output_path)
        output = dict()
        for key in result_dict.keys():
            output[key] = result_dict[key].tolist()
        with open('{}_tuning.json'.format(celltype), 'w') as f:
            json.dump(output, f)
    
        
        # Export loss history dict as json
        with open('{}_loss_history.json'.format(celltype), 'w') as f:
            json.dump(loss_history_dict, f)
        # Export loss difference history dict as json
        with open('{}_loss_diff_history.json'.format(celltype), 'w') as f:
            json.dump(loss_diff_history_dict, f)
        
        try:
            # Plot lambda tuning results
            Fig = ad.lambda_tuning_viz(result_dict, 'Feature_number', savepath='{}_feature_number.png'.format(celltype))
            Fig = ad.lambda_tuning_viz(result_dict, 'AUC', savepath='{}_AUC.png'.format(celltype))
            Fig = ad.lambda_tuning_viz(result_dict, 'loss_history', savepath='{}_loss_history.png'.format(celltype))
            Fig = ad.lambda_tuning_viz(result_dict, 'error_history', savepath='{}_error_history.png'.format(celltype))
            Fig = ad.lambda_tuning_viz(result_dict, 'Precision', savepath='{}_Precision.png'.format(celltype))
            Fig = ad.lambda_tuning_viz(result_dict, 'F1 score', savepath='{}_F1.png'.format(celltype))
            Fig = ad.lambda_tuning_viz(result_dict, 'Train_prevalence', savepath='{}_Train_prevalence.png'.format(celltype))
        except:
            print('***** Error in plotting lambda tuning results')

    return


# %%
adata = scv.datasets.pancreas(file_path='/home/jovyan/work/Research_datasets/Pancreas/endocrinogenesis_day15.h5ad')
label = adata.obs['clusters'].tolist()
print('Original adata:', adata.shape)
types = np.unique(label).tolist()
print('all cell types:', types)
print('====================')
del adata

cts = ['Alpha']
print('cts', cts)
print('====================')


# %%
for celltype in cts:
    ### Read again for each iteration
    adata = scv.datasets.pancreas(file_path='/home/jovyan/work/Research_datasets/Pancreas/endocrinogenesis_day15.h5ad')
    print('Original adata:', adata.shape)  # (32349, 20568)

    ### Remove the genes whose expression is zero in all B cells
    adata_celltype = adata[adata.obs['clusters'] == celltype]
    print('adata celltype shape:', adata_celltype.shape)

    # Remove explicit zeros from the sparse matrix
    adata_celltype.X.eliminate_zeros()

    # Find the columns that are all zeros
    all_zeros = np.where(adata_celltype.X.getnnz(axis=0) == 0)[0]

    # Remove the columns that are all zeros from the anndata object
    adata = adata[:, ~adata_celltype.var_names.isin(adata_celltype.var_names[all_zeros])]
    print('adata shape after removing all zero columns for celltype cells:', adata.shape)
    del adata_celltype, all_zeros

    ### Start lambda tuning
    st = time.time()
    # Set output path
    server_path = '/home/jovyan/work/GitHub/EvanPys/Progress/Pancreas_ADlasso'

    # set learning rate alpha to 0.01
    pipeline_ad2(adata, celltype, label, output_path=server_path)
    et = time.time()
    print('{} Time elapsed: {} minutes.'.format(celltype, (et-st)/60))
    del adata

print('***** Finished lambda tuning')
print('====================')


# %%
