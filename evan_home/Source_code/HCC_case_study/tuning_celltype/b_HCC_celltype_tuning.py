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
# import scvelo as scv
import sklearn
from scipy.sparse import csr_matrix
from sklearn.metrics.cluster import adjusted_rand_score
import copy
import json
import time


# %%
def pipeline_ad2(X_norm, celltype, label, X_raw_count=None, output_path='', tuning_result=''):
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
        # Set range to 1e-5 to 1, 31 lambda values
        # log_lmbd_range = np.linspace(np.log(1e-5), np.log(1), 31)
        print(log_lmbd_range)
        lmbd_range = np.exp(log_lmbd_range)
        print(lmbd_range)

        # Lambda tuning
        result_dict, loss_history_dict, loss_diff_history_dict = ad.lambda_tuning_para_cpu(X_norm, celltype_label, lmbd_range, device='cpu', loss_tol=1e-6, n_jobs=25)
        # result_dict, loss_history_dict, loss_diff_history_dict = ad.lambda_tuning_para_ttsplit(X_norm, celltype_label, lmbd_range, device='cpu', loss_tol=1e-6, n_jobs=25)

        # Export dataframe
        os.chdir(output_path)
        # print('Exporting resultDF')
        # result_DF.to_csv('{}_result_DF.csv'.format(celltype))

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
            Fig = ad.lambda_tuning_viz(result_dict, 'Other_prevalence', savepath='{}_Other_prevalence.png'.format(celltype))
        except:
            print('***** Error in plotting lambda tuning results')

    return


# %%
# adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/HCC_Lu/HCC_Lu_preprocessed_noscale.h5ad')
adata = sc.read_h5ad(dataset_dir / 'HCC_Lu/HCC_Lu_preprocessed_noscale.h5ad')
print('Shape:', adata.shape, type(adata.X))
adata.obs['celltype'] = adata.obs['celltype'].replace('T/NK', 'T_NK')
label = adata.obs['celltype'].tolist()
types = np.unique(label).tolist()
print('Celltypes:', types)
print('====================')

print(adata.obs.head())
print(adata.obs['celltype'].value_counts())
print('======')

del adata

# print('Force quit')
# sys.exit()


# %%
print('***** Starting tuning')
for celltype in types:
    ### Read again for each iteration
    # adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/HCC_Lu/HCC_Lu_preprocessed_noscale.h5ad')
    adata = sc.read_h5ad(dataset_dir / 'HCC_Lu/HCC_Lu_preprocessed_noscale.h5ad')
    adata.obs['celltype'] = adata.obs['celltype'].replace('T/NK', 'T_NK')
    print('Shape:', adata.shape)
    print(adata.obs['celltype'].unique())

    ad_X_sparse = csr_matrix(adata.X)
    print('TYPE', type(ad_X_sparse))
    
    ### Start lambda tuning
    st = time.time()
    # Set output path
    # server_path = '/home/jovyan/work/GitHub/EvanPys/Progress/HCC_case_study/tuning_celltype'
    server_path = source_code_dir / 'HCC_case_study/tuning_celltype'
    # print(server_path)
    print('Valid output path?', os.path.exists(server_path))

    # set learning rate alpha to 0.01
    pipeline_ad2(ad_X_sparse, celltype, label, output_path=server_path)
    et = time.time()
    print('{} Time elapsed: {} minutes.'.format(celltype, (et-st)/60))
    del adata


print('***** Finished lambda tuning')
print('====================')

