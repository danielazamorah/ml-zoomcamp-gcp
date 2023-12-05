# cd Desktop/DANIELA/GitHub/ml-zoomcamp-gcp/kubernetes/homework 

MODEL_IMAGE=zoomcamp-model
MODEL_TAG=kubernetes

kind-create-cluster:
	@kind create cluster

get-cluster-info:
	@kubectl cluster-info
	@kubectl get service
	@kubectl get pod
	@kubectl get deployment

kind-model-apply:
	@kind load docker-image ${MODEL_IMAGE}:${MODEL_TAG}
	@kubectl apply -f $(shell pwd)/deployment.yaml
	@kubectl apply -f $(shell pwd)/service.yaml

port-forward:
	@kubectl port-forward service/credit 8080:80

hpa:
	@kubectl autoscale deployment credit --name credit-hpa --cpu-percent=20 --min=1 --max=3
	@kubectl get hpa

# To activate metric-server: https://gist.github.com/sanketsudake/a089e691286bf2189bfedf295222bd43
watch-hpa:
	@kubectl get hpa credit-hpa --watch