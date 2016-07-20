#!/usr/bin/env bash

email='braini-dockers@test-paper-doll.iam.gserviceaccount.com'
keyfile='test-paper-doll-ec1813a50ec0.p12'
project_id='test-paper-doll'
gcr_url='eu.gcr.io'

echo "docker pull google/cloud-sdk"
docker pull google/cloud-sdk

#this creates container with authenticated gcloud sdk
echo "​docker run -v $(pwd):/tmp -t -i --name gcloud-config google/cloud-sdk gcloud auth activate-service-account $email --key-file /tmp/$keyfile --project $project_id"
docker run -v $(pwd):/tmp -t -i --name gcloud-config google/cloud-sdk gcloud auth activate-service-account $email --key-file /tmp/$keyfile --project $project_id

echo "token=(docker run --rm -ti --volumes-from gcloud-config google/cloud-sdk gcloud auth print-access-token)"
token=$(docker run --rm -ti --volumes-from gcloud-config google/cloud-sdk gcloud auth print-access-token)

echo "docker login -e $email -u _token -p $token https://$gcr_url"
docker login -e $email -u _token -p $token https://$gcr_url



 #