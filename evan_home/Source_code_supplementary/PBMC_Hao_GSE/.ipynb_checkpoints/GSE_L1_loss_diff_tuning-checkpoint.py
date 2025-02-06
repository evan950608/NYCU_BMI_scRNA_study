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
        os.chdir(tuning_result)  # /home/evanlee/Pancreas_AD2/Pancreas_result
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
        result_DF, all_result_dict, loss_history_dict, loss_diff_history_dict = ad.lambda_tuning_cuda(data.X, celltype_label, lmbd_range, device='cuda', loss_tol=1e-6)

        # re-organize all_result_dict
        result_dict_for_viz = dict()
        for outer_key in all_result_dict:
            for inner_key, value in all_result_dict[outer_key].items():
                if inner_key not in result_dict_for_viz:
                    result_dict_for_viz[inner_key] = []
                result_dict_for_viz[inner_key].append(value)

        # Export lambda tuning results DF as csv
        os.chdir(output_path)
        result_DF.to_csv('{}_tuning_dff.csv'.format(celltype), index=False)

        # Export lambda tuning results dict as json
        all_result_dict = convert_numpy_types(all_result_dict)
        with open('{}_tuningg.json'.format(celltype), 'w') as f:
            json.dump(all_result_dict, f)
        
        # Export loss history dict as json
        with open('{}_loss_historyy.json'.format(celltype), 'w') as f:
            json.dump(loss_history_dict, f)
        # Export loss difference history dict as json
        with open('{}_loss_diff_historyy.json'.format(celltype), 'w') as f:
            json.dump(loss_diff_history_dict, f)
        
    
        # Plot lambda tuning results
        Fig = ad.lambda_tuning_viz(result_dict_for_viz, 'Feature_number', savepath='{}_feature_number.png'.format(celltype))
        Fig = ad.lambda_tuning_viz(result_dict_for_viz, 'AUC', savepath='{}_AUC.png'.format(celltype))
        Fig = ad.lambda_tuning_viz(result_dict_for_viz, 'loss_history', savepath='{}_loss_history.png'.format(celltype))
        Fig = ad.lambda_tuning_viz(result_dict_for_viz, 'error_history', savepath='{}_error_history.png'.format(celltype))
        Fig = ad.lambda_tuning_viz(result_dict_for_viz, 'Precision', savepath='{}_Precision.png'.format(celltype))
        Fig = ad.lambda_tuning_viz(result_dict_for_viz, 'F1 score', savepath='{}_F1.png'.format(celltype))
        Fig = ad.lambda_tuning_viz(result_dict_for_viz, 'Train_prevalence', savepath='{}_Train_prevalence.png'.format(celltype))


    ### Only output tuning results for now, leave ignore_section lambda decision for later
    # os.chdir(output_path)
    # opt_lmbd, fig = ad.lambda_decision(result_dict, k=3, savepath='{}_lambda_decision.png'.format(celltype))
    # print('Optimal lambda: {}'.format(opt_lmbd))
    # with open('{}_opt_lambda.txt'.format(celltype), 'w') as f:
    #     f.write(str(opt_lmbd) + '\n')

    return


# %%
# adata = sc.read('/home/evanlee/PBMC_Hao/Hao_PBMC_level1_rep_cells.h5ad')
# adata = sc.read_h5ad('Hao_PBMC_GSE_level1_rep_cells.h5ad')
adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/GSE164378/Hao_PBMC_GSE164378.h5ad')
print('Original adata:', adata.shape)  # (32349, 20568)
label = adata.obs['celltype.l1'].tolist()
types = np.unique(label).tolist()
print('all cell types:', types)
print('====================')

cts = ['B']
print('cts', cts)
print('====================')
# ['B', 'CD4_T', 'CD8_T', 'DC', 'Mono', 'NK', 'other', 'other_T']
# ['ASDC', 'B_intermediate', 'B_memory', 'B_naive', 'CD14_Mono', 'CD16_Mono', 'CD4_CTL', 'CD4_Naive', 'CD4_Proliferating', 'CD4_TCM', 'CD4_TEM', 'CD8_Naive', 'CD8_Proliferating', 'CD8_TCM', 'CD8_TEM', 'Doublet', 'Eryth', 'HSPC', 'ILC', 'MAIT', 'NK', 'NK_CD56bright', 'NK_Proliferating', 'Plasmablast', 'Platelet', 'Treg', 'cDC1', 'cDC2', 'dnT', 'gdT', 'pDC']

# %%
for celltype in cts:
    ### Read again for each iteration
    # adata = sc.read_h5ad('Hao_PBMC_GSE_level1_rep_cells.h5ad')
    adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/GSE164378/Hao_PBMC_GSE164378.h5ad')
    print('Original adata:', adata.shape)  # (32349, 20568)
    
    # CPM (do not do CPM yet)
    # Total-count normalize the data matrix X to 10,000 reads per cell
    # sc.pp.normalize_total(adata, target_sum=1e6)
    # Log
    sc.pp.log1p(adata)

    ### Remove the genes whose expression is zero in all B cells
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

    ### Start lambda tuning
    st = time.time()
    # Set output path
    server_path = '/home/jovyan/work/GitHub/EvanPys/Progress/PBMC_Hao_GSE/'

    # set learning rate alpha to 0.01
    pipeline_ad2(adata, celltype, label, output_path=server_path)
    et = time.time()
    print('{} Time elapsed: {} minutes.'.format(celltype, (et-st)/60))

print('***** Finished lambda tuning')
print('====================')


# %%
