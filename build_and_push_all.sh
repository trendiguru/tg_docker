#!/usr/bin/env bash

#base
nvidia-docker build -t tg:base -f tg_base.Dockerfile .
docker tag tg:base eu.gcr.io/test-paper-doll/tg/base:1
./docker_login_json.sh
gcloud docker push eu.gcr.io/test-paper-doll/tg/base:1

#caffe
nvidia-docker build -t tg/caffe:1 -f cuda_tg_caffe.Dockerfile .
docker tag tg/caffe:1 eu.gcr.io/test-paper-doll/tg/caffe:1
./docker_login_json.sh
gcloud docker push eu.gcr.io/test-paper-doll/tg/caffe:1

#theano
nvidia-docker build -t tg/theano:1 -f cuda_tg_theano.Dockerfile .
docker tag tg/theano:1 eu.gcr.io/test-paper-doll/tg/theano:1
./docker_login_json.sh
gcloud docker push eu.gcr.io/test-paper-doll/tg/theano:1

#keras
nvidia-docker build -t tg/keras:1 -f cuda_tg_keras.Dockerfile .
docker tag tg/keras:1 eu.gcr.io/test-paper-doll/tg/keras:1
./docker_login_json.sh
gcloud docker push eu.gcr.io/test-paper-doll/tg/keras:1

#tensorflow
cd tensorflow
nvidia-docker build -t tg/tf_keras:1 -f cuda_tg_keras-tensorflow.Dockerfile .
docker tag tg/tf_keras:1 eu.gcr.io/test-paper-doll/tg/tf_keras:1
./docker_login_json.sh
gcloud docker push eu.gcr.io/test-paper-doll/tg/tf_keras:1

