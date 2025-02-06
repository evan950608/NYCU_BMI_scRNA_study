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
