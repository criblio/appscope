#! /bin/bash

### Create cluster
kind create cluster

### Load docker image locally (from host into node)
### This step is only requried for local image because
### for official release the image will be available on DockerHub
### and will be pulled.
kind load docker-image cribl/scope:dev
sleep 10

### Run scope k8s to start webhook and k8s server in cluster
docker run -it --rm cribl/scope:dev scope k8s -c tcp://in.main-default-practical-leavitt.cribl.cloud:10091 | kubectl apply -f -
sleep 10
kubectl label namespace default scope=enabled
sleep 10

kubectl get pods
sleep 10

### Example start a pod
### kubectl run redis --image=redis:alpine

### Example: Destroy cluster
### kind delete cluster
