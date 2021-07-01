#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd ${DIR}/.. && sudo make && sudo make local
${DIR}/../../docker/build.sh
minikube delete
minikube start
minikube ssh -- sudo ip link set docker0 promisc on
scope k8s --metricdest tcp://logstream-internal:8125 --metricformat statsd --eventdest tcp://logstream-internal:10070 | kubectl apply -f -
sleep 30
kubectl label namespace default scope=enabled
helm repo add cribl https://criblio.github.io/helm-charts/
helm install -f ${DIR}/../../cribl-values.yml logstream-master cribl/logstream-master
