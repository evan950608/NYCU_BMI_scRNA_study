import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

# read the full GSE dataset
# data = sc.read_h5ad('/Users/evanli/Documents/Research_datasets/PBMC_Hao/GSE164378_Hao/Hao_PBMC_GSE164378.h5ad')
# data = sc.read_h5ad(r"C:\Users\evanlee\Documents\Research_datasets\GSE164378\Hao_PBMC_GSE164378.h5ad")
data = sc.read_h5ad('/ws/Research_datasets/GSE164378/Hao_PBMC_GSE164378.h5ad')
print(data.shape)

### Raw counts were not normalizaed
# CPM
sc.pp.normalize_total(data, target_sum=1e6)

# log1p
sc.pp.log1p(data)

data.obs['celltype.l2'] = data.obs['celltype.l2'].str.replace(' ', '_')
label = data.obs['celltype.l2'].tolist()
types = np.unique(label).tolist()
# types = [s.replace(' ', '_') for s in types]
print(types)


### Read ADlasso features
# Level 2 ADlasso features by loss difference convergence
import os
# os.chdir('/Users/evanli/Documents/EvanPys/Progress/PBMC_Hao_GSE/L1_by_lossdiff/feature_selection')
# os.chdir(r"C:\Users\evanlee\Documents\GitHub\EvanPys\Progress\PBMC_Hao_GSE\L1_by_lossdiff\feature_selection")
os.chdir('/ws/GitHub/EvanPys/Progress/PBMC_Hao_GSE/L2_by_lossdiff/feature_selection')

features_dict = {}
# Read features for each celltype
for celltype in types:
    try:
        feature_df = pd.read_csv(f'{celltype}_features.txt', names=['Gene', 'Weight', 'Tendency'], sep='\t')
        features_dict[celltype] = feature_df
    except:
        print('skipping:', celltype)
        continue

count_df = pd.DataFrame(columns=['Feature_count', 'Positive_feature_count'])
for celltype in features_dict.keys():
    feature_df = features_dict[celltype]
    feature_count = feature_df.shape[0]
    positive_count = feature_df[feature_df['Tendency'] == 1].shape[0]
    count_df.loc[celltype] = [feature_count, positive_count]


### Stratified K-fold cross validation
import pickle
import cupy as cp
import cudf
from cuml import train_test_split, SVC
from cuml.model_selection import StratifiedKFold
from cuml.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import average_precision_score, matthews_corrcoef

def SVM_strat_KFold_cuml(data, all_features_dict, celltype, k=5):
    # X = cp.sparse.csc_matrix(data[:, all_features_dict[celltype]['Gene'].tolist()].X)
    X = cudf.DataFrame(data[:, all_features_dict[celltype]['Gene'].tolist()].X.todense())
    y = cudf.Series([1 if i==celltype else 0 for i in data.obs['celltype.l2'].tolist()])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    print('Fitting SVM model...')
    svm_clf = SVC(kernel='rbf', max_iter=5000)
    svm_clf.fit(X_train, y_train)

    print('Cross validation...')
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'pr_auc': [], 'mcc': []}
    
    kf_count = 1
    # for train_index, test_index in cv.split(cp.asnumpy(X_train), cp.asnumpy(y_train)):
    for train_index, test_index in cv.split(X_train, y_train):
        print('KF:', kf_count)
        # X_train_fold, X_test_fold = cp.asarray(X_train[train_index]), cp.asarray(X_train[test_index])
        # y_train_fold, y_test_fold = cp.asarray(y_train[train_index]), cp.asarray(y_train[test_index])
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        svm_clf.fit(X_train_fold, y_train_fold)
        y_pred = svm_clf.predict(X_test_fold)
        
        # Move y_test_fold and y_pred to CPU memory for sklearn metrics
        y_test_fold_cpu = cp.asnumpy(y_test_fold)
        y_pred_cpu = cp.asnumpy(y_pred)
        metrics['accuracy'].append(accuracy_score(y_test_fold_cpu, y_pred_cpu))
        metrics['precision'].append(precision_score(y_test_fold_cpu, y_pred_cpu))
        metrics['recall'].append(recall_score(y_test_fold_cpu, y_pred_cpu))
        metrics['f1'].append(f1_score(y_test_fold_cpu, y_pred_cpu))
        metrics['roc_auc'].append(roc_auc_score(y_test_fold_cpu, y_pred_cpu))
        metrics['pr_auc'].append(average_precision_score(y_test_fold_cpu, y_pred_cpu))
        metrics['mcc'].append(matthews_corrcoef(y_test_fold_cpu, y_pred_cpu))
        
        kf_count += 1

    mean_metrics = {metric: np.mean(values) for metric, values in metrics.items()}

    # delete used variables
    del X, y, X_train, X_test, y_train, y_test
    return svm_clf, mean_metrics

# run stratified K-fold cross validation
# os.chdir('/Users/evanli/Documents/EvanPys/Progress/PBMC_Hao_GSE/L1_by_lossdiff/SVM_classifiers_all')
# os.chdir(r"C:\Users\evanlee\Documents\GitHub\EvanPys\Progress\PBMC_Hao_GSE\L1_by_lossdiff\SVM_classifiers_all")
os.chdir('/ws/GitHub/EvanPys/Progress/PBMC_Hao_GSE/L2_by_lossdiff/SVM_classifiers_rbf_cuda')

all_metrics = []
for celltype in types:
    print('====================')
    print('K-fold CV for:', celltype)
    clf, metrics_dict = SVM_strat_KFold_cuml(data, features_dict, celltype, k=5)
    print(metrics_dict)
    
    # Append metrics to all_metrics
    all_metrics.append(metrics_dict)

    # output SVM model as pickle
    filename = f'SVM_{celltype}_loss_diff_l2.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)

all_metrics_df = pd.DataFrame(all_metrics)
all_metrics_df.index = types

# save metrics to csv
all_metrics_df.to_csv('SVM_metrics_loss_diff_l2_AD_nostdscale_rbf_cuda.csv')

# Plot metrics for each celltype
cols = ['precision', 'f1', 'roc_auc', 'pr_auc', 'mcc']
ax = all_metrics_df[cols].plot.bar(rot=0, figsize=(15,6), title='One vs. Rest SVM using ADlasso features (kernel=rbf)')
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.savefig('SVM_AD_L2_rbf.png', bbox_inches='tight')