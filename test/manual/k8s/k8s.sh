#! /bin/bash


if [ $# -ne 1 ]; then
    echo "Please provide an IP:port argument for the Edge AppScope source."
    exit
fi

echo -e "\n--- Starting a k8s cluster connecting to Edge at $1  ---\n"

echo -e "\n--- Build AppScope and image ---\n"
cd ~/appscope && make all && make image

cd "${0%/*}"

echo -e "\n--- Create a new cluster ---\n"
kind create cluster

echo -e "\n--- Load docker images locally ---\n"
kind load docker-image cribl/scope:dev
sleep 2

echo -e "\n--- Start Edge in the cluster ---\n"
echo "### NOTE: the leader setting needs to be updated for a specific cloud instance ###"
helm install --repo "https://criblio.github.io/helm-charts/" --version "^4.1.1" --create-namespace -n "cribl" --set "cribl.leader=tls://3Qlq8wAbKH7njR4LQHuZzqlt4bdCY2yl@main-practical-leavitt.cribl.cloud?group=default_fleet" --set "cribl.readinessProbe=null" --set "cribl.livenessProbe=null" "cribl-edge" edge

echo -e "\n--- Run scope k8s to start webhook and k8s server in cluster ---\n"
# arg 1 example: 10.244.0.6:10092
docker run -it --rm cribl/scope:dev scope k8s --metricformat prometheus -m tcp://scope-prom-export:9109 -e tcp://$1 | kubectl apply -f -
sleep 10
kubectl label namespace default scope=enabled
sleep 10

echo -e "\n--- Start an app as source of events & metrics ---\n"
kubectl run redis --image=redis:alpine
sleep 10

echo -e "\n--- Here's a list of pods in the cluster ---\n"
kubectl get pods


