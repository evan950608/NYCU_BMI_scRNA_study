# README - Source code
This is a README for the source code in the study "Marker gene identification algorithm of precision clustering for single cell sequencing" by Zhe-Yuan Li. (January 2025).
\
\
The source code have serveral parts:

## ADlasso2
- This part contains the PreLect package by Chen et al. (2025)
- The version history is listed in ```Version_history.txt```
- The final version is ```AD2_w_utils_lossdiff_noZ.py```

## PBMC_Hao_batch_noZ
- This part is about applying PreLect to the PBMC Hao dataset for feature selection.
- We implemented an independent feature selection pipeline for each Level_1 and Level_2 cell types.
- The code files were labeled from "a" to "i" indicating different steps in the pipeline.
- The results of each step are stored in the specified folders:
    - Lambda tuning: ```tuning_result```
    - Optimal lambda decision: ```lambda_decision```
    - Feature selection (final version): ```feature_selection_k3```
    - Feature evaluation with different models: ```LR_likelihood```, ```SVM_model``` and ```XGB_model```
    - Compare model performances with DEGs: ```DEG_```
- The comparison between PreLect features and ACT markers is in the ```ACT_annotation``` folder.
- The examination of how larger lambda values affect feature confidence is in the ```large_lambda_various``` folder.

## Stuart_bm_annotate
- This part is about annotating an independent PBMC dataset with trained LR models.
- We compared three annotation methods:
    - ```Predict_w_Hao_LR```: prediction by LR models.
    - ```Stuart_SingleR```: annotation by SingleR.
    - ```Stuart_CellAssign```: annotation by CellAssign.

## HCC_case_study
- This part is about applying the PreLect pipeline on a hepatocellular carcinoma dataset as a case study.
- We implemented an independent feature selection pipeline for each Leiden clusters.
- The code files were labeled from "a" to "g" indicating different steps in the pipeline.
- The results of each step are stored in the specified folders:
    - Lambda tuning: ```tuning_leiden```
    - Optimal lambda decision: ```lambda_decision_leiden_k3```
    - Feature selection: ```feature_selection_k3```
- The comparison between PreLect features and ACT markers is in the ```ACT_annotation``` folder.
- Discussion about HCC microenvironment is in the ```Microenvironment``` folder.
