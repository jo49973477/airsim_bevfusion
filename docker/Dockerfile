FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -yq \
    wget build-essential g++ gcc git \
    libgl1-mesa-glx libglib2.0-0 \
    openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda config --set auto_update_conda false && \
    conda config --add channels https://repo.anaconda.com/pkgs/main && \
    conda config --add channels https://repo.anaconda.com/pkgs/r

# 🌟 그리고 conda create 하기 전에 아예 채널을 기본으로 세팅해버리자!
RUN conda config --set auto_update_conda false && \
    conda config --add channels https://repo.anaconda.com/pkgs/main && \
    conda config --add channels https://repo.anaconda.com/pkgs/r

RUN conda create -n bevfusion python=3.8 -y && conda clean -ya
ENV PATH /opt/conda/envs/bevfusion/bin:$PATH
ENV CONDA_DEFAULT_ENV bevfusion

# Pytorch & Base Dependencies
RUN conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch && conda clean -ya

# BEVFusion & AirSim Dependencies (AirSim 통신용 라이브러리만 딱 추가!)
RUN pip install Pillow==8.4.0 tqdm torchpack mmcv==1.4.0 mmcv-full==1.4.0 mmdet==2.20.0 nuscenes-devkit mpi4py==3.0.3 numba==0.48.0
RUN pip install --no-cache-dir msgpack-rpc-python
RUN pip install --no-cache-dir airsim
RUN pip install  "numpy==1.20.3"
RUN pip install flash-attn==1.0.4 --no-build-isolation
RUN pip uninstall -y mmcv mmcv-full
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

WORKDIR /workspace