library(Seurat)
library(SeuratData)
library(cowplot)
library(dplyr)
library(SeuratDisk)

### CITE-seq dataset from Stuart (2019)
# consists of 30,672 scRNA-seq profiles measured alongside a panel of 25 antibodies from bone marrow.
# two assays, RNA & ADT
options(timeout = 240)
InstallData("bmcite")
bm <- LoadData(ds = "bmcite")

# preprocessing
DefaultAssay(bm) <- 'RNA'
# NormalizeData(): Normalize_total and log1p
# ScaleData():: scales and centers features
bm <- NormalizeData(bm) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()

DefaultAssay(bm) <- 'ADT'
# we will use all ADT features for dimensional reduction
# we set a dimensional reduction name to avoid overwriting the 
VariableFeatures(bm) <- rownames(bm[["ADT"]])
bm <- NormalizeData(bm, normalization.method = 'CLR', margin = 2) %>% 
    ScaleData() %>% RunPCA(reduction.name = 'apca')


### Construct WNN graph
# Identify multimodal neighbors. These will be stored in the neighbors slot, 
# and can be accessed using bm[['weighted.nn']]
# The WNN graph can be accessed at bm[["wknn"]], 
# and the SNN graph used for clustering at bm[["wsnn"]]
# Cell-specific modality weights can be accessed at bm$RNA.weight
bm <- FindMultiModalNeighbors(
    bm, reduction.list = list("pca", "apca"), 
    dims.list = list(1:30, 1:18), modality.weight.name = "RNA.weight"
)


### UMAP based on weighted combination of RNA and protein data
bm <- RunUMAP(bm, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
# graph based clustering 
bm <- FindClusters(bm, graph.name = "wsnn", algorithm = 3, resolution = 2, verbose = FALSE)

p1 <- DimPlot(bm, reduction = 'wnn.umap', label = TRUE, repel = TRUE, label.size = 2.5) + NoLegend()
p2 <- DimPlot(bm, reduction = 'wnn.umap', group.by = 'celltype.l1', label = TRUE, repel = TRUE, label.size = 2.5) + NoLegend()
p3 <- DimPlot(bm, reduction = 'wnn.umap', group.by = 'celltype.l2', label = TRUE, repel = TRUE, label.size = 2.5) + NoLegend()
p1 + p2 + p3


### UMAP visualization based on only the RNA and protein data
bm <- RunUMAP(bm, reduction = 'pca', dims = 1:30, assay = 'RNA', 
              reduction.name = 'rna.umap', reduction.key = 'rnaUMAP_')  # RNA better in identifying progenitor states
bm <- RunUMAP(bm, reduction = 'apca', dims = 1:18, assay = 'ADT', 
              reduction.name = 'adt.umap', reduction.key = 'adtUMAP_')  # ADT better in identifying T cells states

p4 <- DimPlot(bm, reduction = 'rna.umap', group.by = 'celltype.l2', label = TRUE, 
              repel = TRUE, label.size = 2.5) + NoLegend()
p5 <- DimPlot(bm, reduction = 'adt.umap', group.by = 'celltype.l2', label = TRUE, 
              repel = TRUE, label.size = 2.5) + NoLegend()
p4 + p5


### visualize the expression of canonical marker genes
p6 <- FeaturePlot(bm, features = c("adt_CD45RA","adt_CD16","adt_CD161"),
                  reduction = 'wnn.umap', max.cutoff = 2, 
                  cols = c("lightgrey","darkgreen"), ncol = 3)
p7 <- FeaturePlot(bm, features = c("rna_TRDC","rna_MPO","rna_AVP"), 
                  reduction = 'wnn.umap', max.cutoff = 3, ncol = 3)
p6 / p7


# visualize the modality weights
VlnPlot(bm, features = "RNA.weight", group.by = 'celltype.l2', sort = TRUE, pt.size = 0.1) +
    NoLegend()


### Export
# write.csv(bm@meta.data, file = "Stuart_metadata_bmcite_RNAassay.csv")


# RNA assay
DefaultAssay(bm) <- 'RNA'
# SaveH5Seurat(bm, filename = "Stuart_bmcite_RNAassay.h5Seurat")

### Convert Seurat_obj to AnnData
library(reticulate)
library(scCustomize)
# py_install("anndata")
as.anndata(x = bm, file_path = "./", file_name = 'Stuart_bmcite_RNAassay.h5ad')


# ADT assay
DefaultAssay(bm) <- 'ADT'










