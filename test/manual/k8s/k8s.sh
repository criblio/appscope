#! /bin/bash


if [ $# -ne 1 ]; then
    echo "Please provide an IP:port argument for the Edge AppScope source."
    exit
fi

echo -e "\n--- Starting a k8s cluster connecting to Edge at $1  ---\n"

echo -e "\n--- Build AppScope and image ---\n"
cd ~/appscope && make all && make image

cd "${0%/*}"

# A handy way to see the state of everything, if run in another shell
# watch kubectl get all --all-namespaces -o wide

echo -e "\n--- Create a new cluster ---\n"
kind create cluster

echo -e "\n--- Load docker images locally ---\n"
kind load docker-image cribl/scope:dev
sleep 2

echo -e "\n--- Start Edge in the cluster ---\n"
echo "### NOTE: the leader setting needs to be updated for a specific cloud instance ###"
helm install --repo "https://criblio.github.io/helm-charts/" --version "^4.1.2" --create-namespace -n "cribl" --set "cribl.leader=tls://<value provided by the Add/Update Edge Node Kubernetes Script>" --set "cribl.readinessProbe=null" --set "cribl.livenessProbe=null" --set "env.CRIBL_K8S_TLS_REJECT_UNAUTHORIZED=0" "cribl-edge" edge


# To enable Edge to discover our prometheus exporter, configure a
# prometheus edge scraper with Discovery Type of "Kubernetes Pods".
# The k8s command provides annotations for port and filter rule to support this.
#


echo -e "\n--- Run scope k8s to start webhook and prometheus exporter in cluster ---\n"
# arg 1 example: 10.244.0.6:10092
docker run -it --rm cribl/scope:dev scope k8s --metricformat statsd --metricprefix appscope -m tcp://scope-stats-exporter:9109 -e tcp://$1 | kubectl apply -f -
sleep 10
kubectl label namespace default scope=enabled
sleep 10

echo -e "\n--- Start an app as source of events & metrics ---\n"
kubectl run redis --image=redis:alpine
sleep 10

echo -e "\n--- Here's a list of pods in the cluster ---\n"
kubectl get pods


