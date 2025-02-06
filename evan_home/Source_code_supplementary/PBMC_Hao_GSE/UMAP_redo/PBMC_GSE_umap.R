library(Seurat)
library(ggplot2)
library(patchwork)

options(SeuratData.repo.use = "http://seurat.nygenome.org")

# load PBMC Hao dataset
data <- readRDS("C:/Users/evanlee/Documents/Research_datasets/PBMC_Hao/from_cellxgene/Hao_PBMC_seu.rds")

# metadata
meta = data@meta.data

# Plot WNN UMAP
DimPlot(object = data, reduction = "wnn.umap", group.by = "celltype.l2", label = TRUE, label.size = 3, repel = TRUE, raster=FALSE) + NoLegend()

# Plot RNA UMAP
DimPlot(object = data, reduction = "umap", group.by = "celltype.l2", label = TRUE, label.size = 3, repel = TRUE, raster=FALSE) + NoLegend()

# Plot ADT (protein) UMAP
DimPlot(object = data, reduction = "aumap", group.by = "celltype.l2", label = TRUE, label.size = 3, repel = TRUE, raster=FALSE) + NoLegend()
