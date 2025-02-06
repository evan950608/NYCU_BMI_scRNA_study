library(Seurat)
library(ggplot2)
library(patchwork)


# load PBMC Hao dataset
data.2 <- readRDS("C:/Users/evanlee/Documents/Research_datasets/PBMC_Hao/from_cellxgene/Hao_PBMC_seu.rds")

# metadata
meta = data.2@meta.data

# clear data.2 reduction list
data.2@reductions = list()


# get RNA assay
rna_assay = data.2@assays[["RNA"]]

# SCT
data.2 = SCTransform(data.2)
