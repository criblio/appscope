#!/bin/bash

docker pull moby/buildkit:buildx-stable-1
docker run --privileged --rm tonistiigi/binfmt --install all
if [ $(docker buildx ls | grep cribl-builder  | wc -l) -eq 0 ]; then
    docker buildx create --name cribl-builder
fi
docker buildx use cribl-builder
# docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
docker buildx inspect --bootstrap
docker buildx ls
