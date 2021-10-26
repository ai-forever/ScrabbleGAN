FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update &&\
    apt-get -y install \
    build-essential yasm nasm cmake \
    unzip git htop nvtop wget curl tmux \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool flex bison \
    python3 python3-pip python3-dev python3-setuptools \
    libglib2.0-0 libgl1-mesa-glx \
    libsm6 libxext6 libxrender1 libssl-dev libx264-dev libsndfile1 libmp3lame-dev &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -sf /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

# Upgrade pip for cv package instalation
RUN pip3 install --upgrade pip==21.0.1

RUN pip3 install --no-cache-dir numpy==1.19.5

# Install PyTorch
RUN pip3 install --no-cache-dir \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111 \
    torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

ENV PYTHONPATH $PYTHONPATH:/workdir/
ENV TORCH_HOME=/workdir/data/.torch
ENV LANG C.UTF-8

WORKDIR /workdir

# Install python ML packages
COPY requirements.txt /workdir
RUN pip3 install --no-cache-dir -r requirements.txt
