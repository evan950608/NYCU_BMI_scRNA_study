library(scRNAseq)
sce <- GrunHSCData(ensembl=TRUE)
sce

# immgen: reference dataset
library(celldex)
immgen <- ImmGenData(ensembl=TRUE)
immgen
head(immgen$label.fine)

library(SingleR)
# See 'Choices of assay data' for 'assay.type.test=' explanation.
pred <- SingleR(test = sce, ref = immgen, 
                labels = immgen$label.fine, assay.type.test=1)
colnames(pred)

head(sort(table(pred$labels), decreasing=TRUE))

actual.hsc <- pred$labels[sce$protocol=="sorted hematopoietic stem cells" & sce$sample!="JC4"]
head(sort(table(actual.hsc), decreasing=TRUE))


### NestorowaHSCData
sce.nest <- NestorowaHSCData()

# Getting the exonic gene lengths.
library(AnnotationHub)
mm.db <- AnnotationHub()[["AH73905"]]
mm.exons <- exonsBy(mm.db, by="gene")
mm.exons <- reduce(mm.exons)
mm.len <- sum(width(mm.exons))

# Computing the TPMs with a simple scaling by gene length.
library(scater)
keep <- intersect(names(mm.len), rownames(sce.nest))
tpm.nest <- calculateTPM(sce.nest[keep,], lengths=mm.len[keep])

# Performing the assignment.
pred <- SingleR(test = tpm.nest, ref = immgen, labels = immgen$label.fine)
head(sort(table(pred$labels), decreasing=TRUE), 10)


### Chapter 3. Using single-cell references
# Aim: use one pre-labelled dataset to annotate the other unlabelled dataset.

# Reference: Muraro 
library(scRNAseq)
sceM <- MuraroPancreasData()

# Removing unlabelled cells or cells without a clear label.
sceM <- sceM[,!is.na(sceM$label) & sceM$label!="unclear"] 

library(scater)
sceM <- logNormCounts(sceM)
sceM

# Seeing the available labels in this dataset.
table(sceM$label)

# Test dataset: Grun
sceG <- GrunPancreasData()

sceG <- addPerCellQC(sceG)
qc <- quickPerCellQC(colData(sceG), 
                     percent_subsets="altexps_ERCC_percent",
                     batch=sceG$donor,
                     subset=sceG$donor %in% c("D17", "D7", "D2"))
sceG <- sceG[,!qc$discard]

sceG <- logNormCounts(sceG)
sceG

library(SingleR)
pred.grun <- SingleR(test=sceG, ref=sceM, labels=sceM$label, de.method="wilcox")
table(pred.grun$labels)

library(SingleR)
pred.grun2 <- SingleR(test=sceG, ref=sceM, labels=sceM$label, 
                      de.method="t", de.args=list(lfc=1))
table(pred.grun2$labels)


library(scran)
out <- pairwiseBinom(counts(sceM), sceM$label, direction="up")
markers <- getTopMarkers(out$statistics, out$pairs, n=10)

# Upregulated in acinar compared to alpha:
markers$acinar$alpha

# Upregulated in alpha compared to acinar:
markers$alpha$acinar

# Creating label-specific markers.
label.markers <- lapply(markers, unlist)
label.markers <- lapply(label.markers, unique)
str(label.markers)

pred.grun2c <- SingleR(test=sceG, ref=sceM, labels=sceM$label, genes=label.markers)
table(pred.grun2c$labels)

# Pseudo-bulk aggregation
set.seed(100) # for the k-means step.
pred.grun3 <- SingleR(test=sceG, ref=sceM, labels=sceM$label, 
                      de.method="wilcox", aggr.ref=TRUE)
table(pred.grun3$labels)





















