import os
import sys
# sys.path.append('/home/evanlee/PBMC_Hao')
sys.path.append('/home/jovyan/work/GitHub/EvanPys/Progress')
from ADlasso2 import AD2_w_utils_loss as ad

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
    opt_res = ad.ADlasso2(lmbd=opt_lmbd, tol=1e-5, echo=True, device='cuda')  # cuda
    opt_res.fit(data.X, celltype_label, pvl, loss_threshold=1e-2)
    
    # Export selection results
    os.chdir(output_path)
    opt_res.writeList(outpath=output_path+f'/{celltype}_features.txt', featureNameList=data.var_names)
    print(f'{celltype} feature list exported.')

    et = time.time()
    elapsed = (et-st)/60
    print(f'Elapsed time for {celltype}: {elapsed} minutes')

    # Ouput description
    description = f'''Optimal lambda: {opt_lmbd}
    median of selected prevalence: {np.median([pvl[i]  for i, w in enumerate(opt_res.feature_set) if w != 0])}
    total selected feature: {np.sum(opt_res.feature_set)}
    Time elapsed: {elapsed}\n'''
    print('---Selection result for {}'.format(celltype))
    print(description)

    with open(f'{celltype}_description.txt', 'w') as f:
        f.write(description)


# %% Multi-processing to run feature selection

# Define a function to execute pipeline_feature_selection for a single cell type
def run_pipeline_feature_selection(celltype):
    # st = time.time()
    # Read adata
    adata = sc.read('/home/jovyan/work/Research_datasets/Hao_PBMC_level2_rep_cells.h5ad')
    print('Original adata:', adata.shape)  # (32349, 20568)

    ### Remove the genes whose expression is zero in all cells of this celltype
    adata_celltype = adata[adata.obs['celltype.l2'] == celltype]
    print('adata celltype shape:', adata_celltype.shape)

    # Convert the anndata object to a Pandas DataFrame
    df = pd.DataFrame(adata_celltype.X.toarray(), columns=adata_celltype.var_names, index=adata_celltype.obs_names)
    # Find the columns that are all zeros
    all_zeros = np.where(~df.any(axis=0))[0]

    # Remove the columns that are all zeros from the anndata object
    adata = adata[:, ~df.columns.isin(df.columns[all_zeros])]
    print('adata shape after removing all zero columns for celltype cells:', adata.shape)
    del adata_celltype, df, all_zeros
    # L1 celltype as labels
    label = adata.obs['celltype.l2'].tolist()


    ### Read optimal lambda dictionary from json (v3 lambda_decision_new)
    with open('/home/jovyan/work/GitHub/EvanPys/Progress/PBMC_Hao/Level2_pvl0_ttsplit/lambda_decision_ignore_section/L2_optimal_lambda.json', 'r') as f:
        opt_lambda_dict = json.load(f)
    opt_lmbd = opt_lambda_dict[celltype]
    print('optimal lambda:', opt_lmbd)

    # return

    pipeline_feature_selection(adata, celltype, label, opt_lmbd, output_path='/home/jovyan/work/GitHub/EvanPys/Progress/PBMC_Hao/Level2_pvl0_ttsplit/lambda_decision_ignore_section')

    # et = time.time()
    # print(f'Elapsed time for {celltype}: {(et-st)/60:.2f} minutes')


# %% Main code
# types = ['B', 'CD4_T', 'CD8_T', 'DC', 'Mono', 'NK', 'other', 'other_T']
adata = sc.read('/home/jovyan/work/Research_datasets/Hao_PBMC_level2_rep_cells.h5ad')
label = adata.obs['celltype.l2'].tolist()
types = np.unique(label).tolist()
print('all cell types:', types)
done_types = ['CD4_Naive', 'CD4_TCM', 'CD8_Naive', 'CD8_TEM', 'NK']

run_pipeline_feature_selection('CD16_Mono')

# for celltype in types:
#     if celltype in done_types:
#         print('====================')
#         print('Skipping {}...'.format(celltype))
#         continue
#     run_pipeline_feature_selection(celltype)

# Create a pool of worker processes
# with mp.Pool(processes=len(types)) as pool:
#     # Map the run_pipeline_feature_selection function to each cell type using the pool of workers
#     pool.map(run_pipeline_feature_selection, types)


    