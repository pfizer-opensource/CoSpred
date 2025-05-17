FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y wget bzip2 git graphviz unzip\
    && rm -rf /var/lib/apt/lists/*

# Install Mambaforge
RUN wget -q -O /tmp/mambaforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash /tmp/mambaforge.sh -b -p /opt/conda \
    && rm /tmp/mambaforge.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Copy the Conda environment YAML file into the container
COPY conf/requirements_cospred_cuda12_gpu_py39.yml /environment/

# Create the Conda environment using the YAML file
RUN mamba env create -n cospred_cuda12_gpu_py39 -y -f /environment/requirements_cospred_cuda12_gpu_py39.yml

# Activate the environment, and add it to the path
RUN conda init bash \
    && echo "conda activate cospred_cuda12_gpu_py39" >> ~/.bashrc \
    && /bin/bash -c "source ~/.bashrc" \
    && conda clean -afy

ENV PATH=/opt/conda/envs/cospred_cuda12_gpu_py39/bin:$PATH

# Set the working directory (optional, adjust as needed)
WORKDIR /

# Copy the application files into the container (optional, adjust as needed)
RUN git clone https://github.com/pfizer-opensource/CoSpred.git

WORKDIR /CoSpred

# Default command to run the container (optional, adjust as needed)
CMD ["bash"]