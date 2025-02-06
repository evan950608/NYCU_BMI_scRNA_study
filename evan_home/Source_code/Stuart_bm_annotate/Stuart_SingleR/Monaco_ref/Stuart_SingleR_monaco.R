### Reference dataset: Monaco immune data (covers typical PBMC)
library(celldex)

ref <- fetchReference("monaco_immune", "2024-02-26")
head(ref)
# ref2 <- MonacoImmuneData()
# head(ref2)


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


### Load in Stuart BM
library(SingleR)
library(Seurat)
library(SeuratDisk)
library(SeuratData)
InstallData("bmcite")
bm <- LoadData(ds = "bmcite")

### preprocessing
DefaultAssay(bm) <- 'RNA'
# NormalizeData(): Normalize_total and log1p
bm <- NormalizeData(bm, normalization.method = "LogNormalize", scale.factor = 10000)
# ScaleData():: scales and centers features
# bm = ScaleData(bm)

# Info about genes
assay.data = GetAssayData(bm)
data.var_names = assay.data@Dimnames[[1]]
data.obs_names = assay.data@Dimnames[[2]]

### Start SingleR
pred.BM.main = SingleR(test = assay.data, 
                   ref = ref, 
                   labels = ref$label.main, 
                   de.method = 'classic')
colnames(pred.BM.main)
# View predicted label counts
sort(table(pred.BM.main$labels), decreasing=TRUE)
# View predicted pruned_label counts (delta needs to be large so assigned labels are meaningful)
sort(table(pred.BM.main$pruned.labels), decreasing=TRUE)

write.csv(pred.BM.main, 'SingleR_pred_Stuart_main.csv')

to.remove = is.na(pred.BM.main$pruned.labels)
table(Label=pred.BM.main$labels, Removed=to.remove)
# to.remove = pruneScores(pred.BM.main, min.diff.med = 0.1)  # if score < 0.1, return True
# table(Label=pred.BM.main$labels, Removed=to.remove)

# Inspect quality of the predictions
plotScoreHeatmap(pred.BM.main)  # heatmap of prediction scores
plotDeltaDistribution(pred.BM.main, ncol = 4, dots.on.top = FALSE)

# marker genes
all.markers = metadata(pred.BM.main)$de.genes
B.markers = unique(unlist(all.markers$`B cells`))
bm$labels = pred.BM.main$labels
library(scater)
plotHeatmap(bm, order_columns_by="labels", features=B.markers)  # error?


### SingleR label.fine
pred.BM.fine = SingleR(test = assay.data, 
                       ref = ref, 
                       labels = ref$label.fine, 
                       de.method = 'classic')
colnames(pred.BM.fine)
# View predicted label counts
sort(table(pred.BM.fine$labels), decreasing=TRUE)
# View predicted pruned_label counts (delta needs to be large so assigned labels are meaningful)
sort(table(pred.BM.fine$pruned.labels), decreasing=TRUE)

write.csv(pred.BM.fine, 'SingleR_pred_Stuart_fine.csv')

to.remove = is.na(pred.BM.fine$pruned.labels)
table(Label=pred.BM.fine$labels, Removed=to.remove)

# Inspect quality of the predictions
plotScoreHeatmap(pred.BM.fine)  # heatmap of prediction scores
plotDeltaDistribution(pred.BM.fine, ncol = 6, dots.on.top = FALSE)





