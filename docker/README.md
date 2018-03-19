# Directory /docker/
Updated: 09/25/17

This directory contains files that are used to build and run a deep learning
docker image. The image contains conda environments for both CPU and GPU systems
with all the Python 3 deep learning libraries needed to run project experiments
on both local and HPC systems.

## Files

- benchmarks/MNIST_CPU.py : MNIST Keras CNN training benchmark for CPU systems.
- benchmarks/MNIST_GPU.py : MNIST Keras CNN training benchmark for GPU systems.
- conda/deeplearning-cpu.yml : Conda enviroment definition for deeplearning on CPU systems.
- conda/deeplearning-gpu.yml : Conda enviroment definition for deeplearning on GPU systems.
- build.sh : Bash file used to build docker image.
- Dockerfile : Docker image definition for Python 3 deep-learning ready docker.
- nvidia-run.sh : Bash file used to run docker image on GPU enabled system.
- run.sh : Bash file used to run docker image on CPU system.
