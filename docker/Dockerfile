# Deeplearning Dockerfile
# Updated: 8/28/17
#
# README:
# Creates Docker image containing conda environments for deeplearning on cpu and
# gpu systems. MNIST benchmark scripts for both systems are provided inside image.
#

# Base Images
FROM nvidia/cuda:8.0-cudnn5-devel

# Install MiniConda and Dependecies
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion nano \
    libjpeg-dev zlib1g-dev graphviz tar sudo nasm libpq-dev build-essential

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.3.14-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH

# Create Conda Environments
WORKDIR /tmp
ADD conda/deeplearning-cpu.yml deeplearning-cpu.yml
ADD conda/deeplearning-gpu.yml deeplearning-gpu.yml

## Deeplearning For CPUS
RUN ["conda", "create", "--name", "deeplearning-cpu", "--file", "deeplearning-cpu.yml"]
RUN ["/bin/bash", "-c", "source activate deeplearning-cpu && pip install tensorflow==1.2.0"]

## Deeplearning For GPU
RUN ["conda", "create", "--name", "deeplearning-gpu", "--file", "deeplearning-gpu.yml"]
RUN ["/bin/bash", "-c", "source activate deeplearning-gpu && pip install tensorflow-gpu"]

# Prompt
RUN echo 'export PS1="\[$(tput bold)\]\[\033[38;5;10m\]\u@DL_Docker\[$(tput sgr0)\]\[\033[38;5;15m\]:\[$(tput sgr0)\]\[\033[38;5;12m\]\W\[$(tput sgr0)\]\[\033[38;5;15m\]\\$\[$(tput sgr0)\]"' >> /etc/bash.bashrc
RUN echo 'export PS1="\[$(tput bold)\]\[\033[38;5;10m\]\u@DL_Docker\[$(tput sgr0)\]\[\033[38;5;15m\]:\[$(tput sgr0)\]\[\033[38;5;12m\]\W\[$(tput sgr0)\]\[\033[38;5;15m\]\\$\[$(tput sgr0)\]"' >> ~/.bashrc

# Create Project Folder
WORKDIR /home/
RUN ["mkdir", "benchmarks"]
ADD benchmarks/MNIST_CPU.py benchmarks/MNIST_CPU.py
ADD benchmarks/MNIST_GPU.py benchmarks/MNIST_GPU.py
RUN ["mkdir", "project"]
WORKDIR /home/project
