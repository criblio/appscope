#! /bin/bash


if [ $# -ne 2 ]; then
    echo "Please provide IP:port arguments for 1) the prom exporter and 2) the Edge AppScope source."
    exit
fi

echo "--- Starting a k8s cluster connecting to Edge at $2 and a Prometheus exporter at $1 ---"
echo ""

echo "--- Build AppScope and image ---"
cd ~/appscope && make all && make image

cd "${0%/*}"

echo "--- Create a prom exporter container ---"
docker build -t cribl/prom-exporter -f Dockerfile.exporter .

echo "--- Create a new cluster ---"
kind create cluster

echo "--- Load docker images locally ---"
kind load docker-image cribl/scope:dev
kind load docker-image cribl/prom-exporter:latest
sleep 2

echo "--- Start a prom exporter pod ---"
kubectl apply -f exporter.yml #exp.yml
#kubectl run prom --image=cribl/prom-exporter:latest
sleep 2

echo "--- Copy a current scope exec to the prom exporter pod ---"
#prom=`kubectl get pods -o=name | grep prom`
prom=`kubectl get pods --template '{{range .items}}{{.metadata.name}}{{"\n"}}{{end}}' | grep prom`
kubectl cp ./bin/linux/aarch64/scope default/$prom:/usr/local/bin/.

echo "--- Start a prom exporter ---"
kubectl exec $prom -- scope prom &

echo "--- Start Edge in the cluster ---"
echo "### NOTE: the leader setting needs to be updated for a specific cloud instance ###"
helm install --repo "https://criblio.github.io/helm-charts/" --version "^4.1.1" --create-namespace -n "cribl" --set "cribl.leader=tls://3Qlq8wAbKH7njR4LQHuZzqlt4bdCY2yl@main-practical-leavitt.cribl.cloud?group=default_fleet" --set "cribl.readinessProbe=null" --set "cribl.livenessProbe=null" "cribl-edge" edge

echo "--- Run scope k8s to start webhook and k8s server in cluster ---"
echo "#### using prom $1 and Edge $2 ###"
docker run -it --rm cribl/scope:dev scope k8s --metricformat prometheus -m tcp://$1  -e tcp://$2 | kubectl apply -f -
sleep 2
kubectl label namespace default scope=enabled
sleep 10

echo "--- Start an app as source of events & metrics ---"
kubectl run redis --image=redis:alpine

echo "--- Here's a list of pods in the cluster ---"
kubectl get pods


