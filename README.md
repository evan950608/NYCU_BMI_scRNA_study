# Marker Gene Identification Algorithm of Precision Clustering for Single-Cell Sequencing

**Zhe-Yuan Li**  
*National Yang Ming Chiao Tung University (NYCU)*  
*Institute of Biomedical Informatics (BMI)*  
*Published: January 2025*  

---

## 1. Introduction
Single-cell RNA sequencing (scRNA-seq) has revolutionized our ability to characterize cellular heterogeneity at cell-level resolution. A pivotal component of scRNA-seq analysis is the identification of marker genes that facilitate cell type annotation. The process entails the precise assignment of each cell to its corresponding type by analyzing patterns of gene expression. 

To address challenges associated with high-dimensional and sparse scRNA-seq data, this study utilizes an in-house feature selection algorithm termed **"PreLect"**. PreLect uniquely integrates prevalence-based penalties into LASSO regularization, a technique that aids in pinpointing genes indicative of specific, localized cellular traits.

## 2. Material and Methods
The core of the methodology revolves around **PreLect**, which helps accurately capture high-dimensional biological characteristics. 

Our study utilizes several open-sourced scRNA-seq datasets for feature selection, validation, and real-world application:
- **PBMC_Hao (GSE164378)**: Used to identify biologically relevant marker genes.
- **Stuart_bm (GSE128639)**: An independent PBMC dataset used to test generalizability.
- **HCC_Lu (GSE149614)**: A hepatocellular carcinoma dataset to evaluate robustness in a case study.

For details on dataset access, please refer to the [Dataset README](evan_home/README_dataset.md).

Models implementing the PreLect algorithm perform supervised feature selection, hyperparameter (lambda) tuning, and classification mapping using robust machine learning engines such as Logistic Regression (LR), Support Vector Machines (SVM), and XGBoost. The pipeline was rigorously evaluated using multi-class performance metrics alongside cross-validation.

> **Note**: Please refer to the [Source Code README](evan_home/Source_code/README_source_code.md) to better understand the code structure, the execution pipeline, and how to utilize the `evan_library` and `ADlasso2` routines.

## 3. Results and Conclusion
1. **High Classification Accuracy**: Logistic regression and XGBoost classifiers built on PreLect-identified features attained over 90% accuracy in multi-class classification on the PBMC dataset, demonstrating impressive performance even for rare cell types.
2. **Biological Relevance**: Comparative analyses successfully revealed a significant alignment between PreLect-selected functionalities and established cell type marker databases (such as ACT markers), firmly validating the intrinsic biological relevance of the identified genes.
3. **Generalizability**: When evaluated on an independent dataset (Stuart PBMC), PreLect-trained Logistic Regression models outperformed conventional cell annotation framework tools such as SingleR and CellAssign.
4. **Real-world Adaptability**: In a case study on hepatocellular carcinoma scRNA-seq data, starting with unsupervised Leiden clustering, the pipeline successfully enabled the discovery of distinct cellular populations and uncovered critical microenvironmental heterogeneity.

From these findings, the study effectively underscores the broad generalizability, adaptability, and robustness of the PreLect-based pipeline for marker gene identification and functional cell-type annotation across diverse modalities of real-world scRNA-seq data.

## 4. System Requirements
This repository supports execution within both local OS and containerized GPU environments. Testing was natively conducted using:
- **Windows 11** (Python 3.9)
- **Ubuntu 20.04 via Docker Environment** (Python 3.10)

Core package dependencies inherently involve: `anndata`, `scanpy`, `numpy`, `scikit-learn`, `xgboost`, `pytorch`, `leidenalg`, and `scvi-tools` among others.

For highly detailed technical setup instructions, explicitly matching dependency versionings, and how to bootstrap the Docker image, please consult the [System Requirements README](evan_home/README_system_requirements.md).
