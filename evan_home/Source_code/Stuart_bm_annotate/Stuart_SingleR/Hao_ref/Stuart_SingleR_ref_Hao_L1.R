library(fs)

# Get the current working directory
current_path = dirname(rstudioapi::getActiveDocumentContext()$path)
home_folder <- "evan_home"

# Function to traverse up the directory tree to find the target folder
find_home_path <- function(current_path, home_folder) {
    parents <- path_split(current_path)[[1]]
    
    for (i in seq_along(parents)) {
        parent_path <- path_join(parents[1:i])
        if (path_file(parent_path) == home_folder) {
            return(parent_path)
        }
    }
    stop(paste("Folder '", home_folder, "' not found in the current working directory.", sep = ""))
}

# Find the home path
home_path <- find_home_path(current_path, home_folder)

cat("Home Path:", home_path, "\n")

# Define source code and dataset directories
source_code_dir <- path(home_path, "Source_code")
dataset_dir <- path(home_path, "Dataset")

### Load Hao_PBMC as reference
library(SingleR)
library(Seurat)
library(SeuratDisk)
library(SeuratData)
# hao = LoadH5Seurat("C:/Users/evanlee/Documents/Research_datasets/PBMC_Hao/from_atlas_fredhutch/Hao_PBMC.h5seurat")
hao = LoadH5Seurat(file.path(dataset_dir, "PBMC_Hao/from_atlas_fredhutch/Hao_PBMC.h5seurat"))

### Load Stuart BMcite as query
bm <- LoadData(ds = "bmcite")
DefaultAssay(bm) <- 'RNA'
# NormalizeData(): Normalize_total and log1p
bm <- NormalizeData(bm, normalization.method = "LogNormalize", scale.factor = 10000)

# Info about genes
assay.data = GetAssayData(bm)
data.var_names = assay.data@Dimnames[[1]]
data.obs_names = assay.data@Dimnames[[2]]


### Start SingleR: Level 1
pred.Hao.L1 = SingleR(test = assay.data,  # Stuart
                       ref = GetAssayData(hao), 
                       labels = hao$celltype.l1,
                      # labels = hao@meta.data[["celltype.l1"]], 
                       de.method = 'classic')

colnames(pred.Hao.L1)
# View predicted label counts
sort(table(pred.Hao.L1$labels), decreasing=TRUE)
# View predicted pruned_label counts (delta needs to be large so assigned labels are meaningful)
sort(table(pred.Hao.L1$pruned.labels), decreasing=TRUE)

write.csv(pred.Hao.L1, 'SingleR_pred_Stuart_ref_Hao_L1.csv')

# Inspect quality of the predictions
plotScoreHeatmap(pred.Hao.L1)
plotDeltaDistribution(pred.Hao.L1, ncol = 4, dots.on.top = FALSE)
