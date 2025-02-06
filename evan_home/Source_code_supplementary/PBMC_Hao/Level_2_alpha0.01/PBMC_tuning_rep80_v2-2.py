import os
os.chdir('/home/evanlee/PBMC_Hao')
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
adata = sc.read('/home/evanlee/PBMC_Hao/Hao_PBMC_rep_cells.h5ad')
print(adata.shape)  # row is cells, column is gene

label = adata.obs['celltype.l2'].tolist()
types = np.unique(label).tolist()

# %%
def pipeline_ad2(data, celltype, label, output_path='', tuning_result=''):
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
        # Lambda tuning
        result_dict = ad.lambda_tuning_parallel(data.X, celltype_label, lmbd_range, device='cpu', n_jobs=25)

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
cts = types[6:12]
print(cts)

for celltype in cts:
    if celltype == 'CD4 CTL':
        print('====================')
        print('skipping CD4 CTL')
        continue
    st = time.time()
    pipeline_ad2(adata, celltype, label, output_path='/home/evanlee/PBMC_Hao/AD2_result')
    et = time.time()
    print('{} Time elapsed: {} minutes.'.format(celltype, (et-st)/60))
    