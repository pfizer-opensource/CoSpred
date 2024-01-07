FROM nvidia/cuda:9.0-devel-ubuntu16.04

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV TENSORFLOW_VERSION=1.6.0
ENV CUDNN_VERSION=7.0.5.15-1+cuda9.0
ENV NCCL_VERSION=2.1.15-1+cuda9.0

# Python 2.7 or 3.5 is supported by Ubuntu Xenial out of the box
ENV PYTHON_VERSION=2.7

RUN apt-get update && apt-get install -y --allow-unauthenticated --allow-downgrades --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    wget \
    ca-certificates \
    libcudnn7=7.0.5.15-1+cuda9.0 \
    libnccl2=2.1.15-1+cuda9.0 \
    libnccl-dev=2.1.15-1+cuda9.0 \
    libjpeg-dev \
    libpng-dev \
    python2.7 \
    python2.7-dev

RUN ln -s /usr/bin/python2.7 /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Upgrade pip
RUN pip install --upgrade pip

# Install TensorFlow and Keras
RUN pip install --no-cache-dir tensorflow-gpu==1.6.0 keras h5py scikit-learn==0.18.0 matplotlib pandas tables argparse nltk==3.2.2
