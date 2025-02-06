# this code is designed to run on server
# %%
from ADlasso2 import AD2_w_utils as ad
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import scipy
import sklearn
import copy
import importlib
import json

import scanpy as sc
import scvelo as scv
print('cwd:', os.getcwd())

# data = scv.datasets.pancreas(file_path='/Users/evanli/Documents/Research_datasets/endocrinogenesis_day15.h5ad')
data = scv.datasets.pancreas(file_path='/home/evanlee/Pancreas_AD2/endocrinogenesis_day15.h5ad')
label = data.obs['clusters'].tolist()
print(data.shape)

# Count each cell types
from collections import Counter
print(Counter(label))

# log1p: log(x+1)
sc.pp.log1p(data)


# %% 
from ADlasso2 import AD2_w_utils as ad
importlib.reload(ad)

# %% Pipeline of a celltype for ADlasso2
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
        result_dict = ad.lambda_tuning_evan(data.X, celltype_label, lmbd_range, device='cpu')

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

    # Find marker genes with optimal lambda
    pvl = ad.get_prevalence(data.X, celltype_indices)
    opt_res = ad.ADlasso2(lmbd=opt_lmbd, echo=True, device='cpu')
    opt_res.fit(data.X, celltype_label, pvl)

    # print('median of selected prevalence :',np.median([pvl[i]  for i, w in enumerate(opt_res.feature_set) if w != 0]))
    # print('total selected feature :',np.sum(opt_res.feature_set))

    # Export selection results
    # os.chdir(output_path)
    # opt_res.writeList(outpath=f'{celltype}_features.txt', featureNameList=data.var_names)
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


    return


# %%
# query = ['Alpha', 'Ngn3 high EP', 'Delta', 'Epsilon', 'Beta', 'Ductal', 'Pre-endocrine', 'Ngn3 low EP']
# query = np.unique(label)
# print(query)

# for celltype in query:
#     st = time.time()
#     pipeline_ad2(data, celltype, label, output_path='/home/evanlee/Pancreas_AD2/Pancreas_result/result_v2', tuning_result='/home/evanlee/Pancreas_AD2/Pancreas_result')
#     et = time.time()
#     print('Time elapsed: {} minutes.'.format((et-st)/60))

# print('===== End of code')


# %% Multi-processing
query = ['Beta', 'Delta', 'Ductal', 'Epsilon', 'Ngn3 high EP', 'Ngn3 low EP', 'Pre-endocrine']
print(query)

import multiprocessing as mp
import time

def process_celltype(celltype):
    st = time.time()
    pipeline_ad2(data, celltype, label, output_path='/home/evanlee/Pancreas_AD2/Pancreas_result/result_v3', tuning_result='/home/evanlee/Pancreas_AD2/Pancreas_result')
    et = time.time()
    print('Time elapsed for {}: {} minutes.'.format(celltype, (et-st)/60))

if __name__ == '__main__':
    with mp.Pool(processes=len(query)) as pool:
        pool.map(process_celltype, query)

print('===== End of code')