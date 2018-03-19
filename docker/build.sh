#!/usr/bin/env bash
# build.sh
#- BUILDS Docker
# Updated: 8/28/17

# Build dl-experiments
sudo docker build -t dl-docker .
sudo docker ps -aq --no-trunc | xargs docker rm
sudo docker images -q --filter dangling=true | xargs docker rmi
