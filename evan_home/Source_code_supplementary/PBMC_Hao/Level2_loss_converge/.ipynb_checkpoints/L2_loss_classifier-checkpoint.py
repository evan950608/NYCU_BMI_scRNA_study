# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import scipy
import sklearn
import copy
import importlib


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score, cross_validate, KFold


# %%
import scanpy as sc
# import scvelo as scv

# data = sc.read('/home/evanlee/PBMC_Hao/Hao_PBMC_level1_rep_cells.h5ad')
# data = sc.read('/Users/evanli/Documents/Research_datasets/PBMC_Hao/Hao_PBMC.h5ad')
data = sc.read_h5ad('/home/jovyan/work/Research_datasets/Hao_PBMC.h5ad')
# data = sc.read_h5ad(r"C:\Users\evanlee\Documents\Research_datasets\Hao_PBMC.h5ad")

# %%
# CPM
sc.pp.normalize_total(data, target_sum=1e6)

# log1p
sc.pp.log1p(data)

# %%
data.obs['celltype.l2'] = data.obs['celltype.l2'].str.replace(' ', '_')

label = data.obs['celltype.l2'].tolist()
types = np.unique(label).tolist()
# types = [s.replace(' ', '_') for s in types]
print(types)


# %%
# Level 2 ADlasso features by loss convergence (level2_loss_converge/loss_feature_selection)
# selected at optimal lambda tuned at Level2_pvl0_ttsplit
import os
# os.chdir(r"C:\Users\evanlee\Documents\GitHub\EvanPys\Progress\PBMC_Hao\Level2_loss_converge\loss_feature_selection")
os.chdir("/home/jovyan/work/GitHub/EvanPys/Progress/PBMC_Hao/Level2_loss_converge/loss_feature_selection")

features_dict = {}
# Read features for each celltype
for celltype in types:
    try:
        feature_df = pd.read_csv(f'{celltype}_features.txt', names=['Gene', 'Weight', 'Tendency'], sep='\t')
        features_dict[celltype] = feature_df
    except:
        print('skipping:', celltype)
        continue
    print(celltype, 'Feature count:', feature_df.shape[0])
    print(celltype, 'Positive feature count:', feature_df[feature_df['Tendency'] == 1].shape[0])
    print('------------------')

# %%
count_df = pd.DataFrame(columns=['Feature_count', 'Positive_feature_count'])
for celltype in features_dict.keys():
    feature_df = features_dict[celltype]
    feature_count = feature_df.shape[0]
    positive_count = feature_df[feature_df['Tendency'] == 1].shape[0]
    count_df.loc[celltype] = [feature_count, positive_count]
print(count_df)

# %%
import pickle

def LR_kfold(data, all_features_dict, celltype, k=5):
    # subset data to celltype features
    X = data[:, all_features_dict[celltype]['Gene'].tolist()].X
    # Binary label
    y = [1 if i==celltype else 0 for i in data.obs['celltype.l2'].tolist()]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    clf = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000)
    clf.fit(X_train, y_train)

    # Kfold cross validation
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'f1_score': 'f1',
        'roc_auc': 'roc_auc'
    }
    cv_results = cross_validate(clf, X, y, cv=5, scoring=scoring)

    mean_accuracy = np.mean(cv_results['test_accuracy'])
    mean_precision = np.mean(cv_results['test_precision'])
    mean_f1 = np.mean(cv_results['test_f1_score'])
    mean_auc = np.mean(cv_results['test_roc_auc'])
    mean_metrics = [mean_accuracy, mean_precision, mean_f1, mean_auc]

    return clf, mean_metrics

# %%
# os.chdir(r"C:\Users\evanlee\Documents\GitHub\EvanPys\Progress\PBMC_Hao\Level2_loss_converge\classifier_all")
os.chdir("/home/jovyan/work/GitHub/EvanPys/Progress/PBMC_Hao/Level2_loss_converge/classifier_all")

all_metrics = pd.DataFrame(columns=['Accuracy', 'Precision', 'F1-score', 'ROC-AUC'])

go = False
for celltype in types:
    if celltype == 'CD4_Proliferating':
        go = True
    if not go:
        print('skipping:', celltype)
        continue

    print('====================')
    print('K-fold CV for:', celltype)
    clf, metrics = LR_kfold(data, features_dict, celltype, k=5)  # metrics is a list
    print(metrics)
    all_metrics = pd.concat([all_metrics, pd.DataFrame([metrics], columns=['Accuracy', 'Precision', 'F1-score', 'ROC-AUC'])], axis=0)
    
    # output LR model as pickle
    filename = f'LR_{celltype}_loss_l2.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)


# %%



