import os
import sys
# sys.path.append('/Users/evanli/Documents/EvanPys/Progress')
sys.path.append('/home/jovyan/work/GitHub/EvanPys/Progress')
from ADlasso2 import AD2_w_utils_lossdiff as ad

import numpy as np
import pandas as pd
from pathlib import Path
import scanpy as sc
# import scvelo as scv
import sklearn
from scipy.sparse import csr_matrix
from sklearn.metrics.cluster import adjusted_rand_score
import copy
import json
import time

if len(sys.argv) > 1:
    queue_ct = sys.argv[1]
else:
    print("No celltype provided.")
    sys.exit(1)

print('Queue celltype:', queue_ct)

# %%
def pipeline_ad2(X_norm, celltype, label, output_path='', tuning_result=''):
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
        start = np.log(1e-5)
        end = np.log(1e-1)
        num_steps = 25
        step_size = (end - start) / (num_steps - 1)
        # Extend end by 5 steps
        extended_end = end + 5 * step_size
        log_lmbd_range = np.linspace(start, extended_end, num_steps + 5)  # 25 steps from 1e-5 to 1e-1, then extend by 5 steps
        log_lmbd_last5 = log_lmbd_range[-5:]
        print(log_lmbd_last5)
        lmbd_range = np.exp(log_lmbd_last5)
        print(lmbd_range)

        # Lambda tuning
        # result_dict = ad.lambda_tuning_para_ttsplit(data.X, celltype_label, lmbd_range, alpha=alpha, device='cpu', n_jobs=25)
        # result_DF, all_result_dict, loss_history_dict, loss_diff_history_dict = ad.lambda_tuning_cuda(data.X, celltype_label, lmbd_range, device='cuda', loss_tol=1e-6)
        result_dict, loss_history_dict, loss_diff_history_dict = ad.lambda_tuning_para_ttsplit(X_norm, celltype_label, lmbd_range, device='cpu', loss_tol=1e-6, n_jobs=5)

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
adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/PBMC_Hao/GSE164378_Hao/Batch_corrected/Hao_L2_rep_cells_loginv_Harmony.h5ad')
print('Original adata:', adata.shape, type(adata.X))
label = adata.obs['celltype.l2'].tolist()
types = np.unique(label).tolist()
print('all cell types:', types)
print('====================')
del adata

# queue = ['ILC', 'CD4_Proliferating', 'CD8_Proliferating', 'Eryth', 'ASDC']
# print('Queue', queue)
# print('====================')
# ['B', 'CD4_T', 'CD8_T', 'DC', 'Mono', 'NK', 'other', 'other_T']
# ['ASDC', 'B_intermediate', 'B_memory', 'B_naive', 'CD14_Mono', 'CD16_Mono', 'CD4_CTL', 'CD4_Naive', 'CD4_Proliferating', 'CD4_TCM', 'CD4_TEM', 'CD8_Naive', 'CD8_Proliferating', 'CD8_TCM', 'CD8_TEM', 'Doublet', 'Eryth', 'HSPC', 'ILC', 'MAIT', 'NK', 'NK_CD56bright', 'NK_Proliferating', 'Plasmablast', 'Platelet', 'Treg', 'cDC1', 'cDC2', 'dnT', 'gdT', 'pDC']


# print('Force quit')
# sys.exit()

# %%
for celltype in [queue_ct]:
    ### Read again for each iteration
    adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/PBMC_Hao/GSE164378_Hao/Batch_corrected/Hao_L2_rep_cells_loginv_Harmony.h5ad')
    print('Original adata:', adata.shape)

    ### Normalization:
    # already performed in Hao_L1_inv_rep_cells_scaled_Harmony.h5ad

    ### Remove the genes whose expression is zero in all B cells
    adata_celltype = adata[adata.obs['celltype.l2'] == celltype]
    print('adata celltype shape:', adata_celltype.shape)
    print('type adata_celltype.X', type(adata_celltype.X))

    # Remove explicit zeros from the sparse matrix
    adata_celltype.X.eliminate_zeros()

    # Find the columns that are all zeros
    all_zeros = np.where(adata_celltype.X.getnnz(axis=0) == 0)[0]
    # all_zeros = np.where(np.all(adata_celltype.X == 0, axis=0))[0]

    # Remove the columns that are all zeros from the anndata object
    adata = adata[:, ~adata_celltype.var_names.isin(adata_celltype.var_names[all_zeros])]
    print('adata shape after removing all zero columns for celltype cells:', adata.shape)
    del adata_celltype, all_zeros
    
    ad_X_sparse = csr_matrix(adata.X)
    # print('TYPE', type(ad_X_sparse))
    
    ### Start lambda tuning
    st = time.time()
    # Set output path
    server_path = '/home/jovyan/work/GitHub/EvanPys/Progress/PBMC_Hao_batch/Level2_log_inv/'

    # set learning rate alpha to 0.01
    pipeline_ad2(ad_X_sparse, celltype, label, output_path=server_path)
    et = time.time()
    print('{} Time elapsed: {} minutes.'.format(celltype, (et-st)/60))
    del adata


print('***** Finished lambda tuning')
print('====================')


# %%
