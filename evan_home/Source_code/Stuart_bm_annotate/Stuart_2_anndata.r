library(Seurat)
library(SeuratData)
library(cowplot)
library(dplyr)
library(SeuratDisk)
# library(rliger)
library(scCustomize)
# library(qs)

options(timeout = 360)
InstallData("bmcite")
bm <- LoadData(ds = "bmcite")

# Metadata
metadata = bm@meta.data  # 30672 cells
# write.csv(metadata, 'Stuart_metadata_bmcite_RNAassay_original.csv')

# Get var_names (genes)
DefaultAssay(bm) <- 'RNA'
assay.data = GetAssayData(bm)
data.var_names = assay.data@Dimnames[[1]]
data.obs_names = assay.data@Dimnames[[2]]
bm@assays[["RNA"]]@layers[["data"]]@Dimnames[[1]] = data.var_names
bm@assays[["RNA"]]@layers[["data"]]@Dimnames[[2]] = data.obs_names

rownames(bm@assays[["RNA"]]@meta.data) = data.var_names
# SaveH5Seurat(bm, filename = "Stuart_bmcite_RNAassay_original.h5Seurat")

### Convert Seurat_obj to AnnData
library(reticulate)
# py_install("anndata")
as.anndata(x = bm, file_path = "./", assay = 'RNA', file_name = "Stuart_bmcite_RNAassay_original_v2.h5ad")


# ADT assay
DefaultAssay(bm) <- 'ADT'


