import os
import sys
# sys.path.append('/Users/evanli/Documents/EvanPys/Progress')
sys.path.append('/home/jovyan/work/GitHub/EvanPys/Progress')
from ADlasso2 import AD2_w_utils_lossdiff_noZ as ad

import numpy as np
import pandas as pd
from pathlib import Path
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score
import copy
import json
import time
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_property(adata, Y, AD_object):
    # Get feature property
    prop = ad.featureProperty(adata.X, Y, AD_object)
    prop['featureID'] = adata.var_names
    print(prop.head())

    # Plot feature property
    # Filter the data
    positive_data = prop[prop['select'] == 'PreLect_positive']
    negative_data = prop[prop['select'] == 'PreLect_negative']
    other_data = prop[prop['select'] == 'No selected']

    # Plot the other dots with grey color and alpha=0.5
    sns.scatterplot(x="prevalence_1", y="prevalence_0", color='#BCBCBC', alpha=0.5, data=other_data, label='Others')
    # Plot the positive dots with red color and alpha=1
    sns.scatterplot(x="prevalence_1", y="prevalence_0", color='r', alpha=1, data=positive_data, label='PreLect_positive')
    # Plot the negative dots with blue color and alpha=1
    sns.scatterplot(x="prevalence_1", y="prevalence_0", color='b', alpha=1, data=negative_data, label='PreLect_negative')

    # Get the current axes
    ax = plt.gca()

    # Get the handles and labels from the scatterplot
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 2, 0]
    # Set the legend
    plt.legend(handles=[handles[idx] for idx in order], labels=[labels[idx] for idx in order], loc='upper left', fontsize='small')
    plt.xlabel('Target prevalence')
    plt.ylabel('Other prevalence')
    plt.title(f'{celltype}: selection profile')
    plt.show()
    plt.savefig(f'{celltype}_selection_profile.png', dpi=300)
    plt.close('all')
    del adata



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
    
    opt_res = ad.ADlasso2(lmbd=opt_lmbd, loss_tol=1e-6, echo=True, device='cuda')  # cuda
    opt_res.fit(data.X, celltype_label, pvl)
    
    # Export selection results
    os.chdir(output_path)
    opt_res.writeList(outpath=output_path+f'/{celltype}_features.txt', featureNameList=data.var_names)
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

    with open(f'{celltype}_description.txt', 'w') as f:
        f.write(description)
    
    # Export loss_history as json
    with open(f'{celltype}_loss_history.json', 'w') as f:
        json.dump(opt_res.loss_history, f)
    
    # Export loss_diff_history as json
    with open(f'{celltype}_loss_diff_history.json', 'w') as f:
        json.dump(opt_res.loss_diff_history, f)
    
    ### Feature property
    plot_feature_property(data, celltype_label, opt_res)


# %% 

# Define a function to execute pipeline_feature_selection for a single cell type
def run_pipeline_feature_selection(celltype):
    st = time.time()
    # Read adata
    adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/PBMC_Hao/GSE164378_Hao/Harmony_noZ/Hao_L1_repcells_loginv_Harmony_noZ.h5ad')
    print('Original adata:', adata.shape)
    
    # L1 celltype as labels
    label = adata.obs['celltype.l1'].tolist()

    ### Read optimal lambda dictionary from json
    path_opt_lambda = '/home/jovyan/work/GitHub/EvanPys/Progress/PBMC_Hao_batch_noZ/Level1/L1c_k3_opt_lmbd.json'
    with open(path_opt_lambda, 'r') as f:
        opt_lambda_dict = json.load(f)
    opt_lmbd = opt_lambda_dict[celltype]
    print('optimal lambda:', opt_lmbd)

    server_fractal_path = '/home/jovyan/work/GitHub/EvanPys/Progress/PBMC_Hao_batch_noZ/Level1/feature_selection_k3'
    local_path = ''
    pipeline_feature_selection(adata, celltype, label, opt_lmbd, output_path=server_fractal_path)

    et = time.time()
    print(f'Elapsed time for {celltype}: {(et-st)/60:.2f} minutes')
    del adata


# %% Main code
# types = ['B', 'CD4_T', 'CD8_T', 'DC', 'Mono', 'NK', 'other', 'other_T']
adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/PBMC_Hao/GSE164378_Hao/Harmony_noZ/Hao_L1_repcells_loginv_Harmony_noZ.h5ad')
print('Original adata:', adata.shape)
label = adata.obs['celltype.l1'].tolist()
types = np.unique(label).tolist()
print('all cell types:', types)
print('====================')
del adata

queue = ['B']
print('Queue', queue)
# print('force stop')
# sys.exit()

for celltype in queue:
    run_pipeline_feature_selection(celltype)
