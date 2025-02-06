#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, f1_score, accuracy_score

import os
import scanpy as sc

#%% Load data
data = sc.read_h5ad('/home/jovyan/work/Research_datasets/Hao_PBMC.h5ad')
print('Original data shape:', data.shape)

# CPM
sc.pp.normalize_total(data, target_sum=1e6)
# log1p
sc.pp.log1p(data)

data.obs['celltype.l2'] = data.obs['celltype.l2'].str.replace(' ', '_')
label = data.obs['celltype.l2'].tolist()
types = np.unique(label).tolist()
# types = [s.replace(' ', '_') for s in types]
print('All cell types:', types)


#%% Read feature dict
os.chdir('/home/jovyan/work/GitHub/EvanPys/Progress/PBMC_Hao/Level2_pvl0_ttsplit/lambda_decision_ignore_section/L2_feature_selection')

features_dict = {}
# Read features for each celltype
for celltype in types:
    try:
        print('==================')
        print('Reading features:', celltype)
        feature_df = pd.read_csv(f'{celltype}_features.txt', names=['Gene', 'Weight', 'Tendency'], sep='\t')
        features_dict[celltype] = feature_df
    except:
        print('skipping:', celltype)
        continue
    # print(celltype, 'Feature count:', feature_df.shape[0])
    # print(celltype, 'Positive feature count:', feature_df[feature_df['Tendency'] == 1].shape[0])
    # print('------------------')


#%% Regular K-fold CV for all cell types (only positive features)
import pickle

def LR_kfold_positive(data, all_features_dict, celltype, k=5, classifier=None):
    # subset data to celltype features
    positive = all_features_dict[celltype][all_features_dict[celltype]['Tendency'] == 1]['Gene'].tolist()
    X = data[:, positive].X
    if X.shape[1] == 0:
        print(celltype, 'has no positive features')
        return None, [0, 0, 0, 0]
    # Binary label
    y = [1 if i==celltype else 0 for i in data.obs['celltype.l2'].tolist()]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    if classifier is None:
        print('Building classifier...')
        classifier = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000)
        classifier.fit(X_train, y_train)

    # Kfold cross validation
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'f1_score': 'f1',
        'roc_auc': 'roc_auc'
    }
    cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring)

    mean_accuracy = np.mean(cv_results['test_accuracy'])
    mean_precision = np.mean(cv_results['test_precision'])
    mean_f1 = np.mean(cv_results['test_f1_score'])
    mean_auc = np.mean(cv_results['test_roc_auc'])
    mean_metrics = [mean_accuracy, mean_precision, mean_f1, mean_auc]

    return classifier, mean_metrics


#%% Stratified K-fold CV for all cell types (only positive features)
def LR_kfold_stratified_positive(data, all_features_dict, celltype, k=5, classifier=None):
    # subset data to celltype features
    positive = all_features_dict[celltype][all_features_dict[celltype]['Tendency'] == 1]['Gene'].tolist()
    X = data[:, positive].X
    if X.shape[1] == 0:
        print(celltype, 'has no positive features')
        return None, [0, 0, 0, 0]
    # Binary label
    y = [1 if i==celltype else 0 for i in data.obs['celltype.l2'].tolist()]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    if classifier is None:
        print('Building classifier...')
        classifier = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000)
        classifier.fit(X_train, y_train)

    # Stratified Kfold cross validation
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'f1_score': 'f1',
        'roc_auc': 'roc_auc'
    }
    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    cv_results = cross_validate(classifier, X, y, cv=skf, scoring=scoring)

    mean_accuracy = np.mean(cv_results['test_accuracy'])
    mean_precision = np.mean(cv_results['test_precision'])
    mean_f1 = np.mean(cv_results['test_f1_score'])
    mean_auc = np.mean(cv_results['test_roc_auc'])
    mean_metrics = [mean_accuracy, mean_precision, mean_f1, mean_auc]

    return classifier, mean_metrics


# %% Run LR_kfold for all cell types
os.chdir('/home/jovyan/work/GitHub/EvanPys/Progress/PBMC_Hao/Level2_pvl0_ttsplit/lambda_decision_ignore_section/classifiers_positive')

regular_all_metrics = pd.DataFrame(columns=['Accuracy', 'Precision', 'F1-score', 'ROC-AUC'])
stratified_all_metrics = pd.DataFrame(columns=['Accuracy', 'Precision', 'F1-score', 'ROC-AUC'])

# cts = ['CD8_TCM', 'Treg']
for celltype in types:
    print('==================')
    print('Running K-fold CV for:', celltype)
    # Regular K-fold CV
    clf_r, r_metrics = LR_kfold_positive(data, features_dict, celltype)
    if clf_r is None:
        print('skipping:', celltype)
        continue
    regular_all_metrics.loc[celltype] = r_metrics
    # Stratified K-fold CV
    clf_s, s_metrics = LR_kfold_stratified_positive(data, features_dict, celltype, classifier=clf_r)
    stratified_all_metrics.loc[celltype] = s_metrics

    # Save classifier
    filename = f'LR_positive_{celltype}_l2.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(clf_r, f)


# %% Output metrics
regular_all_metrics.to_csv('L2_regular_Kfold_metrics.csv')
stratified_all_metrics.to_csv('L2_stratified_Kfold_metrics.csv')
print('==================')
print('Metrics')
print(regular_all_metrics.head())
print(stratified_all_metrics.head())

