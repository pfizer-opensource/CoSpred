ARG cudaversion=10.1
ARG cudnnversion=7
ARG build=devel
ARG os=ubuntu
ARG osversion=16.04

FROM nvcr.io/nvidia/cuda:${cudaversion}-cudnn${cudnnversion}-${build}-${os}${osversion}

# Install system packages
RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libxext-dev \
    libxrender-dev \
    wget && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y --no-install-recommends git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Install conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

ARG pyversion=3
ARG cudaversion=10.1

RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda$pyversion-latest-Linux-x86_64.sh && \
    /bin/bash /Miniconda$pyversion-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda$pyversion-latest-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh

# Install Python packages and keras

# Create a new user
ENV NB_USER deepuser
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    chown $NB_USER $CONDA_DIR -R && \
    mkdir -p /workspace && \
    chown $NB_USER /workspace && \
    chown $NB_USER /opt 

USER $NB_USER

ADD py${pyversion}-cuda${cudaversion}.yml /opt/py${pyversion}-cuda${cudaversion}.yml
RUN conda update -n base conda && \
    pip install --upgrade pip && \
    conda env update -n base -f /opt/py${pyversion}-cuda${cudaversion}.yml -v && \
    conda clean -vyt

ENV CMAKE_PREFIX_PATH=/opt/conda/

ADD oe_license.txt /workspace/.oe_license.txt
ENV OE_LICENSE /workspace/.oe_license.txt

WORKDIR /workspace

CMD /bin/bash
