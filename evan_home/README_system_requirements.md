# README - System requirements
This is a README for system requirements in the study "Marker gene identification algorithm of precision clustering for single cell sequencing" by Zhe-Yuan Li. (January 2025).\
Two operating systems were used interchangeably: **Windows 11** and **Ubuntu** from a Docker image.

## Operating systems
### Windows
- OS Name: Microsoft Windows 11 教育版
- OS Version: 10.0.22631 N/A Build 22631
- Python Version: 3.9.12
- Code Editor: Visual Studio Code Insiders 1.97.0-insider

### Docker (Ubuntu)
- OS Name: Ubuntu 20.04.6 LTS
- Python Version: 3.10.10
- Start the Docker image with the following command:\
(Change the ```{path}``` to your home directory)
```
docker run --gpus all -d -it -p 8848:8888 -v {path}:/home/jovyan/work -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes --user root cschranz/gpu-jupyter:v1.5_cuda-11.6_ubuntu-20.04_python-only
```

## Dependencies
### Windows
```
pip install anndata==0.10.8 h5py==3.10.0 harmony-pytorch==0.1.8 harmonypy==0.0.9 \
    igraph==0.10.8 ipykernel==6.26.0 ipython==8.17.2 jupyter_client==8.6.0 \
    jupyter_core==5.5.0 leidenalg==0.10.1 louvain==0.8.1 matplotlib==3.8.1 \
    matplotlib-inline==0.1.6 matplotlib-venn==1.1.1 networkx==3.2.1 \
    numpy==1.26.2 pandas==2.1.3 \
    scanpy==1.9.6 scikit-learn==1.3.2 scipy==1.11.3 scvelo==0.3.2 \
    scvi-colab==0.12.0 scvi-tools==1.1.6.post2 seaborn==0.13.2 torch==2.1.1+cu121 \
    tqdm==4.66.1 umap-learn==0.5.4 xgboost==2.1.2
```

### Docker (Ubuntu)
```
pip install anndata==0.10.9 conda==23.3.1 h5py==3.8.0 igraph==0.11.6 \
    ipykernel==6.22.0 ipython==8.12.0 json5==0.9.5 jupyter_client==8.1.0 \
    jupyter_core==5.3.0 jupyter_server==2.4.0 jupyterlab==3.6.3 \
    jupyterlab_server==2.22.0 leidenalg==0.10.2 louvain==0.8.2 matplotlib==3.7.1 \
    matplotlib-inline==0.1.6 networkx==3.0 numpy==1.23.5 nvidia-nccl-cu12==2.23.4 \
    pandas==2.0.0 scanpy==1.10.2 scikit-learn==1.2.2 scipy==1.10.1 seaborn==0.13.2 \
    tensorflow==2.10.1 torch==1.13.1+cu116 tqdm==4.65.0 umap-learn==0.5.6 \
    xgboost==2.1.2
```