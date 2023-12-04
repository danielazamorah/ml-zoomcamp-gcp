MODEL_IMAGE=zoomcamp-model
MODEL_TAG=xception-v4-001
GATEWAY_IMAGE=zoomcamp-gateway
GATEWAY_TAG=002

get-path:
	@echo $(shell pwd)

run-model-container:
	@docker run -it --rm -p 8500:8500 -v $(shell pwd)/clothing-model:/models/clothing-model/1 -e MODEL_NAME="clothing-model" emacski/tensorflow-serving:latest-linux_arm64

build-image-model:
	@docker build -t ${MODEL_IMAGE}:${MODEL_TAG} -f image-model.dockerfile .

run-model-in-container:
	@docker run -it --rm -p 8500:8500 ${MODEL_IMAGE}:${MODEL_TAG}

build-image-gateway:
	@docker build -t ${GATEWAY_IMAGE}:${GATEWAY_TAG} -f image-gateway.dockerfile .

run-gateway-container:
	@docker run -it --rm -p 9696:9696 ${GATEWAY_IMAGE}:${GATEWAY_TAG}

run-compose:
	@docker-compose up -d

down-compose:
	@docker-compose down

kind-create-cluster:
	@kind create cluster

get-cluster-info:
	@kubectl get service
	@kubectl get pod
	@kubectl get deployment

kind-model-apply:
	@kind load docker-image ${MODEL_IMAGE}:${MODEL_TAG}
	@kubectl apply -f $(shell pwd)/kube-config/model-deployment.yaml
	@kubectl apply -f $(shell pwd)/kube-config/model-service.yaml

port-forward:
	# @kubectl port-forward tf-serving-clothing-model-5dc4888564-ttrd5 8500:8500
	# @kubectl port-forward service/tf-serving-clothing-model 8500:8500
	@kubectl port-forward service/gateway 8080:80

kind-gateway-apply:
	@kind load docker-image ${GATEWAY_IMAGE}:${GATEWAY_TAG}
	@kubectl apply -f $(shell pwd)/kube-config/gateway-deployment.yaml
	@kubectl apply -f $(shell pwd)/kube-config/gateway-service.yaml