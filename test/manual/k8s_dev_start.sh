#! /bin/bash

cd ~/scope-k8s-demo && ./stop.sh
docker rmi -f $(docker images -a -q)
docker volume prune
cd ~/appscope && make all
ver=`~/appscope/bin/linux/scope version --tag`
echo $ver
cd ~/appscope && docker build -t cribl/scope:$ver -f docker/base/Dockerfile .
export SCOPE_VER=$ver
cd ~/scope-k8s-demo && ./start.sh cribl
