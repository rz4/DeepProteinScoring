#!/usr/bin/env bash
# nvidia-run.sh
#- Launches Nvidia Docker
# Updated: 09/25/17

# PATHs
PROJECT="$(dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )")"
#DATA=/data/

# Variables
#IMG=rzamora4/hpc-deeplearning:latest
IMG=dl-docker:latest

# Build Protein-Structure-Exploration:GPU
nvidia-docker run -v $PROJECT:/home/project -ti $IMG
#nvidia-docker run -v $DATA:/tmp/ -v $PROJECT:/home/project -ti $IMG
