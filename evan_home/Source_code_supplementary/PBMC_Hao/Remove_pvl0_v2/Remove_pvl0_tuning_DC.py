import os
import sys
# sys.path.append('/Users/evanli/Documents/EvanPys/Progress')
# sys.path.append('/home/evanlee/PBMC_Hao')
# sys.path.append(r'C:\Users\evanlee\Documents\GitHub\EvanPys\Progress')
# sys.path.append('~/work/Do/Documents/GitHub/EvanPys/Progress')
sys.path.append('/home/jovyan/work/Documents/GitHub/EvanPys/Progress')
from ADlasso2 import AD2_w_utils_precision as ad

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
# adata = sc.read('/home/evanlee/PBMC_Hao/Hao_PBMC_level1_rep_cells.h5ad')
adata = sc.read('/home/jovyan/work/Documents/Research_datasets/Hao_PBMC_level1_rep_cells.h5ad')
print('Original adata:', adata.shape)  # (32349, 20568)


# %% DC cells
# Remove the genes whose expression is zero in all B cells
# TODO: write function into ADlasso2
adata_DC = adata[adata.obs['celltype.l1'] == 'DC']
print('adata DC shape:', adata_DC.shape)  # (2760, 20568)

# Convert the anndata object to a Pandas DataFrame
df = pd.DataFrame(adata_DC.X.toarray(), columns=adata_DC.var_names, index=adata_DC.obs_names)
# Find the columns that are all zeros
all_zeros = np.where(~df.any(axis=0))[0]

# Remove the columns that are all zeros from the anndata object
# 2794 genes are removed, their expression is zero in all representative B cells
adata = adata[:, ~df.columns.isin(df.columns[all_zeros])]
print('adata shape after removing all zero columns for DC cells:', adata.shape)  # (32349, 17774) not (2760, 17774)
del adata_DC, df, all_zeros

label = adata.obs['celltype.l1'].tolist()
types = np.unique(label).tolist()
print('all cell types:', types)

# %%
def pipeline_ad2(data, celltype, label, alpha, output_path='', tuning_result=''):
    print('====================')
    print('Starting job for {}'.format(celltype))
    # Binary classification of a celltype
    celltype_label = [1 if x == celltype else 0 for x in label]
    print(celltype_label[:50])
    print(set(celltype_label))
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
        # Lambda tuning
        # result_dict = ad.lambda_tuning_parallel(data.X, celltype_label, lmbd_range, device='cpu', n_jobs=25)
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
        Fig = ad.lambda_tuning_viz(result_dict, 'Precision', savepath='{}_Precision.png'.format(celltype))
        Fig = ad.lambda_tuning_viz(result_dict, 'F1 score', savepath='{}_F1.png'.format(celltype))

    # Lambda decision k = 2
    os.chdir(output_path)
    opt_lmbd, fig = ad.lambda_decision(result_dict, k=2, savepath='{}_lambda_decision.png'.format(celltype))
    print('Optimal lambda: {}'.format(opt_lmbd))
    with open('{}_opt_lambda.txt'.format(celltype), 'w') as f:
        f.write(str(opt_lmbd) + '\n')

    return


# %%
def run_tuning(adata, cts, label):
    for celltype in cts:
        st = time.time()
        # set learning rate alpha to 0.01
        # local_path = '/Users/evanli/Documents/EvanPys/Progress/PBMC_Hao/Level_1_alpha0.001'
        # server_path = '/home/evanlee/PBMC_Hao/Remove_pvl0/DC'
        server_path = '/home/jovyan/work/Documents/GitHub/EvanPys/Progress/PBMC_Hao/Remove_pvl0_v2/DC'
        pipeline_ad2(adata, celltype, label, alpha=0.01, output_path=server_path)
        et = time.time()
        print('{} Time elapsed: {} minutes.'.format(celltype, (et-st)/60))

    print('***** Finished lambda tuning')
    print('====================')
    print('***** Starting feature selection')


# %%
cts = [types[3]]
print('cts', cts)
print('====================')

# for celltype in cts:
#     st = time.time()
#     pipeline_ad2(adata, celltype, label, output_path='/home/evanlee/PBMC_Hao/AD2_result')
#     et = time.time()
#     print('{} Time elapsed: {} minutes.'.format(celltype, (et-st)/60))

run_tuning(adata, cts, label)