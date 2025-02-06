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

import sys
sys.path.append(str(source_code_dir))
from evan_library import config_utils

# In[] Datasets
### Hao_Harmony_test_no_scale
adata = sc.read_h5ad(dataset_dir / 'PBMC_Hao/GSE164378_Hao/Harmony_noZ/Hao_Harmony_test_no_scale.h5ad')

### Hao_L1_repcells_loginv_Harmony_noZ
adata = sc.read_h5ad(dataset_dir / 'PBMC_Hao/GSE164378_Hao/Harmony_noZ/Hao_L1_repcells_loginv_Harmony_noZ.h5ad')

### Hao_L2_repcells_loginv_Harmony_noZ
adata = sc.read_h5ad(dataset_dir / 'PBMC_Hao/GSE164378_Hao/Harmony_noZ/Hao_L2_repcells_loginv_Harmony_noZ.h5ad')

### HCC raw
adata = sc.read_h5ad(dataset_dir / 'HCC_Lu/HCC_Lu_GSE149614_raw.h5ad')

### HCC_Lu_preprocessed_noscale
adata = sc.read_h5ad(dataset_dir / 'HCC_Lu/HCC_Lu_preprocessed_noscale.h5ad')

### Stuart BM
bmcite = sc.read_h5ad(dataset_dir / 'Stuart_bm/Stuart_bmcite_RNAassay_original.h5ad')

# In[] Paths
### PBMC_Hao_batch_noZ/Level1
server_path = source_code_dir / 'PBMC_Hao_batch_noZ/Level1'

# In[] Testing
print(server_path.exists())
break