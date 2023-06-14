This README provides instructions for testing the scope k8s functionality with containers that do not have a shell (distroless).
This directory contains the example distroless Docker image (`Dockerfile`) with k8s deployment definition (`deploy.yaml`).

### Building the example Docker Image

```
docker build -t distrolesstest:latest .
```

### Loading the example Image into the Cluster (kind)

```
kind load docker-image distrolesstest:latest
```

### Deploy the example Image into the Cluster

```
kubectl apply -f deploy.yaml
```
