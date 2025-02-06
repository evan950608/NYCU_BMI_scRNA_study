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


# %%
def pipeline_ad2(X_norm, X_raw_count, celltype, label, output_path='', tuning_result=''):
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
        # log_lmbd_range = np.linspace(np.log(1e-2), np.log(1e2), 25)
        print(log_lmbd_range)
        lmbd_range = np.exp(log_lmbd_range)
        print(lmbd_range)

        # Lambda tuning
        result_DF, result_dict, loss_history_dict, loss_diff_history_dict = ad.lambda_tuning_cuda(X_norm, X_raw_count, celltype_label, lmbd_range, device='cuda', loss_tol=1e-6)        
        # result_dict, loss_history_dict, loss_diff_history_dict = ad.lambda_tuning_para_ttsplit(X_norm, celltype_label, lmbd_range, device='cpu', loss_tol=1e-6, n_jobs=25)
        print(result_DF.head())
        # print('-----')
        # print(result_dict)
        # print('-----')
        # print(loss_history_dict)
        # print(loss_diff_history_dict)

        # Export dataframe
        os.chdir(output_path)
        print('Exporting resultDF')
        result_DF.to_csv('{}_result_DF.csv'.format(celltype))

        # Export lambda tuning results dict as json
        print('Exporting result Dict')
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
adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/PBMC_Hao/GSE164378_Hao/Harmony_and_ztrans/Hao_L1v2_repcells_loginv_Harmony_ztrans.h5ad')
print('Z-transformed rep_cells adata:', adata.shape, type(adata.X))
label = adata.obs['celltype.l1'].tolist()
types = np.unique(label).tolist()
print('all cell types:', types)
print('====================')

queue = types
print('Queue', queue)
print('====================')
# ['B', 'CD4_T', 'CD8_T', 'DC', 'Mono', 'NK', 'other', 'other_T']
# ['ASDC', 'B_intermediate', 'B_memory', 'B_naive', 'CD14_Mono', 'CD16_Mono', 'CD4_CTL', 'CD4_Naive', 'CD4_Proliferating', 'CD4_TCM', 'CD4_TEM', 'CD8_Naive', 'CD8_Proliferating', 'CD8_TCM', 'CD8_TEM', 'Doublet', 'Eryth', 'HSPC', 'ILC', 'MAIT', 'NK', 'NK_CD56bright', 'NK_Proliferating', 'Plasmablast', 'Platelet', 'Treg', 'cDC1', 'cDC2', 'dnT', 'gdT', 'pDC']

raw_count_adata = sc.read_h5ad("/home/jovyan/work/Research_datasets/PBMC_Hao/GSE164378_Hao/Hao_PBMC_GSE164378_raw.h5ad")
raw_count_adata_subset = raw_count_adata[adata.obs_names, adata.var_names]
print('Subsetted raw count adata:', raw_count_adata_subset.shape, type(raw_count_adata_subset.X))
del adata

# print('Force quit')
# sys.exit()

# %%
print('***** Starting tuning')
for celltype in queue:
    ### Read again for each iteration
    adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/PBMC_Hao/GSE164378_Hao/Harmony_and_ztrans/Hao_L1v2_repcells_loginv_Harmony_ztrans.h5ad')
    print('Z-transformed rep_cells adata:', adata.shape)

    ### Normalization:
    # already performed in representative cell selection

    ### Remove the genes whose expression is zero in all B cells
    # raw_count_adata_subset_celltype = raw_count_adata_subset[raw_count_adata_subset.obs['celltype.l1'] == celltype]  # further subset to the target celltype
    # print('Target celltype shape:', raw_count_adata_subset_celltype.shape)
    # Find the columns that are all zeros
    # all_zeros = np.where(raw_count_adata_subset_celltype.X.getnnz(axis=0) == 0)[0]
    # all_zeros = np.where(np.all(adata_celltype.X == 0, axis=0))[0]
    # print('All zeros cols count:', len(all_zeros))

    # Remove the columns that are all zeros from the Z-transformed Anndata object
    # adata = adata[:, ~adata.var_names.isin(adata.var_names[all_zeros])]
    # print('Adata shape after removing all zero columns for celltype cells:', adata.shape)
    # del raw_count_adata_subset_celltype, all_zeros

    # ad_X_sparse = csr_matrix(adata.X)
    # print('TYPE', type(ad_X_sparse))
    ad_X_dense = np.asarray(adata.X)
    print('TYPE', type(ad_X_dense))
    
    ### Start lambda tuning
    st = time.time()
    # Set output path
    server_path = '/home/jovyan/work/GitHub/EvanPys/Progress/PBMC_Hao_ztrans/Level1_v2'

    # set learning rate alpha to 0.01
    # pipeline_ad2(ad_X_sparse, celltype, label, output_path=server_path)
    pipeline_ad2(ad_X_dense, raw_count_adata_subset.X, celltype, label, output_path=server_path)
    et = time.time()
    print('{} Time elapsed: {} minutes.'.format(celltype, (et-st)/60))
    del adata


print('***** Finished lambda tuning')
print('====================')

