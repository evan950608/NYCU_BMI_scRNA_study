### Reference dataset: Monaco immune data (covers typical PBMC)
library(celldex)
ref <- MonacoImmuneData()
head(ref)

# Main celltypes
table(ref@colData@listData[["label.main"]])
# Fine celltypes
table(ref@colData@listData[["label.fine"]])

# reference dataset celltype hierarchy
ref_celltype = data.frame(label.main = ref@colData@listData[["label.main"]], 
                          label.fine = ref@colData@listData[["label.fine"]])
ref_celltype = ref_celltype[order(ref_celltype$label.main), ]  # sort by label.main
ref_celltype_unique = unique(ref_celltype)
write.csv(ref_celltype_unique, 'Monaco_ref_celltypes.csv')

### Load in Hao PBMC
library(SingleR)
library(Seurat)
library(SeuratDisk)
# library(SeuratData)
setwd(r"(C:\Research_datasets_main\GSE164378_Hao\GSE164378_RAW\GSM5008737_RNA_3P)")

expression_matrix <- ReadMtx(
    mtx = "GSM5008737_RNA_3P-matrix.mtx.gz", 
    features = "GSM5008737_RNA_3P-features.tsv.gz",
    cells = "GSM5008737_RNA_3P-barcodes.tsv.gz"
)
seurat_object <- CreateSeuratObject(counts = expression_matrix)

# Normalization
seurat_object = NormalizeData(seurat_object, normalization.method = "LogNormalize", scale.factor = 10000)

# save as H5Seurat
# SaveH5Seurat(seurat_object, 'PBMC_Hao2')

# Info about genes
assay.data = GetAssayData(seurat_object)
data.var_names = assay.data@Dimnames[[1]]
data.obs_names = assay.data@Dimnames[[2]]


### Start SingleR
# pred.Hao = SingleR(test = seurat_object,
#                    ref = ref,
#                    labels = ref$label.main)
pred.Hao = SingleR(test = assay.data, 
                   ref = ref, 
                   labels = ref$label.main, 
                   de.method = 'classic')
colnames(pred.Hao)
sort(table(pred.Hao$labels), decreasing=TRUE)

# save seurat_object and pred.Hao
# save(seurat_object, pred.Hao, file = 'PBMC_Hao_Seuobj.RData')
# save(pred.Hao, file = 'SingleR_pred_label_main.RData')

write.csv(pred.Hao, 'Pred_Hao.csv')


### SingleR label.fine
pred.Hao.fine = SingleR(test = assay.data, 
                        ref = ref, 
                        labels = ref$label.fine, 
                        de.method = 'classic')
colnames(pred.Hao.fine)
sort(table(pred.Hao.fine$labels), decreasing=TRUE)
write.csv(pred.Hao.fine, 'Pred_Hao_fine.csv')
# save(pred.Hao.fine, file = 'SingleR_pred_label_fine.RData')
