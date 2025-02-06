import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scanpy as sc


# In[57]:
### ARI and NMI
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score

label = adata.obs['celltype.l1'].tolist()

DEG_performance = pd.DataFrame(columns=['ARI', 'NMI'])
all_features_performance = pd.DataFrame(columns=['ARI', 'NMI'])  # from ADlasso
positive_features_performance = pd.DataFrame(columns=['ARI', 'NMI'])  # from ADlasso

for celltype in types:
    # DEG17
    deg_true = [1 if x == celltype else 0 for x in label]
    deg_pred = adata_DEGn.obs['leiden'].tolist()
    ari = adjusted_rand_score(deg_true, deg_pred)
    nmi = normalized_mutual_info_score(deg_true, deg_pred)
    new_row = pd.DataFrame({'ARI': ari, 'NMI': nmi}, index=[celltype])
    DEG_performance = pd.concat([DEG_performance, new_row])

    # 17 ADlasso features
    all_features_true = [1 if x == celltype else 0 for x in label]
    all_features_pred = adata_type_features.obs['leiden'].tolist()
    ari = adjusted_rand_score(all_features_true, all_features_pred)
    nmi = normalized_mutual_info_score(all_features_true, all_features_pred)
    new_row = pd.DataFrame({'ARI': ari, 'NMI': nmi}, index=[celltype])
    all_features_performance = pd.concat([all_features_performance, new_row])
    

# In[69]:
# Plot ARI comparison of DEG17 and B cell features

DEG_ari = DEG_performance['ARI']
all_features_ari = all_features_performance['ARI']

# set up the figure and axis
fig, ax = plt.subplots(figsize=(4, 4))

# set up the bar widths and positions
x = 0
width = 0.35
offset = width / 2

# plot the bars
rects1 = ax.bar(x - offset, DEG_ari['B'], width, label='DEG17')
rects2 = ax.bar(x + offset, all_features_ari['B'], width, label='B_cell Features')

# add labels and legend
ax.set_xticks((x,))
ax.set_xticklabels('B', ha='right')
ax.set_ylabel('ARI')
ax.set_title('ARI Comparison of DEG17 and B_cell Features')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# show the plot
plt.show()


# In[70]:
# Plot NMI comparison of DEG17 and B cell features

DEG_nmi = DEG_performance['NMI']
all_features_nmi = all_features_performance['NMI']

# set up the figure and axis
fig, ax = plt.subplots(figsize=(4, 4))

# set up the bar widths and positions
x = 0
width = 0.35
offset = width / 2

# plot the bars
rects1 = ax.bar(x - offset, DEG_nmi['B'], width, label='DEG17')
rects2 = ax.bar(x + offset, all_features_nmi['B'], width, label='B_cell Features')

# add labels and legend
ax.set_xticks((x,))
ax.set_xticklabels('B', ha='right')
ax.set_ylabel('NMI')
ax.set_title('NMI Comparison of DEG17 and B_cell Features')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# show the plot
plt.show()


# In[72]:
### Separate B cells cluster

# DEG17
# adata_DEGn 中，B cells 真實label的那群當作true；B cells 對應的 Leiden group 當作 pred

# binary label
deg_true = adata_DEGn[adata_DEGn.obs['celltype.l1']=='B'].obs['celltype.l1'].tolist()
deg_true = [1 if i == 'B' else 0 for i in deg_true]
deg_pred = adata_DEGn[adata_DEGn.obs['celltype.l1']=='B'].obs['leiden'].tolist()

ari = adjusted_rand_score(deg_true, deg_pred)
nmi = normalized_mutual_info_score(deg_true, deg_pred)
