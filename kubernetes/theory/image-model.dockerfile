FROM emacski/tensorflow-serving:latest-linux_arm64

COPY clothing-model /models/clothing-model/1
ENV MODEL_NAME="clothing-model"