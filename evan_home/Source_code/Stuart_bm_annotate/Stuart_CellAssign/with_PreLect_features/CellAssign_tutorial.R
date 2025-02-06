library(SingleCellExperiment)
library(cellassign)

data(example_sce)
print(example_sce)

print(head(example_sce$Group))
#> [1] "Group1" "Group2" "Group2" "Group1" "Group1" "Group1"

data(example_marker_mat)
print(example_marker_mat)
#>        Group1 Group2
#> Gene1       1      0
#> Gene2       0      1
#> Gene3       1      0
#> Gene4       1      0
#> Gene5       1      0
#> Gene6       0      1
#> Gene7       0      1
#> Gene8       0      1
#> Gene9       0      1
#> Gene10      1      0

s <- sizeFactors(example_sce)

### Fitting cellassign
# It is critical that gene expression data containing only marker genes is used as input to cellassign.
fit <- cellassign(exprs_obj = example_sce[rownames(example_marker_mat),], 
                  marker_gene_info = example_marker_mat, 
                  s = s, 
                  learning_rate = 1e-2, 
                  shrinkage = TRUE,
                  verbose = FALSE)
