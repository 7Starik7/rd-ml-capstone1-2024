FROM emacski/tensorflow-serving:2.5.1

COPY traffic-sign-recognition-model /models/traffic-sign-recognition-model/1
ENV MODEL_NAME="traffic-sign-recognition-model"