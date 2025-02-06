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

data.obs['celltype.l1'] = data.obs['celltype.l1'].str.replace(' ', '_')

label = data.obs['celltype.l1'].tolist()
types = np.unique(label).tolist()
# types = [s.replace(' ', '_') for s in types]
print(types)


### Read ADlasso features
import os
# os.chdir('/Users/evanli/Documents/EvanPys/Progress/PBMC_Hao_GSE/L1_by_lossdiff/feature_selection')
# os.chdir(r"C:\Users\evanlee\Documents\GitHub\EvanPys\Progress\PBMC_Hao_GSE\L1_by_lossdiff\feature_selection")
os.chdir('/ws/GitHub/EvanPys/Progress/PBMC_Hao_GSE/L1_by_lossdiff/feature_selection')

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


### Read DEG table
# os.chdir("/Users/evanli/Documents/EvanPys/Progress/PBMC_Hao_GSE/L1_by_lossdiff/L1_DEG/L1_DEG_table")
# os.chdir(r"C:\Users\evanlee\Documents\GitHub\EvanPys\Progress\PBMC_Hao_GSE\L1_by_lossdiff\L1_DEG\L1_DEG_table")
os.chdir('/ws/GitHub/EvanPys/Progress/PBMC_Hao_GSE/L1_by_lossdiff/L1_DEG/L1_DEG_table')

DEG_table = pd.read_csv(celltype + '_DEG2000.csv', index_col=0)


### Train SVM classifier
import pickle
import cupy as cp
import cudf
from cuml import train_test_split, SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import average_precision_score, matthews_corrcoef

def SVM_classifier_DEG_cuml(data, DEGn, celltype):
    # subset data to celltype features
    X = cudf.DataFrame(data[:, DEGn].X.todense())
    # Binary label
    y = cudf.Series([1 if i==celltype else 0 for i in data.obs['celltype.l1'].tolist()])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    print('Fitting SVM...')
    svm_clf = SVC(kernel='rbf', max_iter=5000)
    svm_clf.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = svm_clf.predict(X_test)
    y_test_cpu = cp.asnumpy(y_test)
    y_pred_cpu = cp.asnumpy(y_pred)

    # Calculate metrics
    accuracy = accuracy_score(y_test_cpu, y_pred_cpu)
    precision = precision_score(y_test_cpu, y_pred_cpu)
    recall = recall_score(y_test_cpu, y_pred_cpu)
    f1 = f1_score(y_test_cpu, y_pred_cpu)
    roc_auc = roc_auc_score(y_test_cpu, y_pred_cpu)
    pr_auc = average_precision_score(y_test_cpu, y_pred_cpu)
    mcc = matthews_corrcoef(y_test_cpu, y_pred_cpu)

    metrics = [accuracy, precision, recall, f1, roc_auc, pr_auc, mcc]
    print(metrics)
    return svm_clf, metrics


# Run SVM classifier for each celltype
cols = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'PR-AUC', 'MCC']
all_metrics_DEG = pd.DataFrame(columns=cols)

for celltype in types:
    print('==================')
    print('Constructing classifier for:', celltype)
    n_features = count_df.loc[celltype, 'Feature_count']
    print('n:', n_features)
    DEG_table = pd.read_csv(celltype + '_DEG2000.csv', index_col=0)
    DEGn = DEG_table['names'][:n_features].tolist() 
    clf, celltype_metrics = SVM_classifier_DEG_cuml(data, DEGn, celltype)
    all_metrics_DEG.loc[celltype] = celltype_metrics

# Save metrics
os.chdir('/ws/GitHub/EvanPys/Progress/PBMC_Hao_GSE/L1_by_lossdiff/L1_DEG')
all_metrics_DEG.to_csv('L1_loss_diff_DEGn_SVM_rbf_metrics.csv')

# Plot metrics for each celltype
cols = ['Precision', 'F1-score', 'ROC-AUC', 'PR-AUC', 'MCC']
ax = all_metrics_DEG[cols].plot.bar(rot=0, figsize=(8,6), title='One vs. Rest SVM using DEGn (kernel=rbf)')
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.savefig('SVM_DEGn_L1_rbf_cuda.png', bbox_inches='tight')