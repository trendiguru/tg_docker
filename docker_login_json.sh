#!/usr/bin/env bash

project_id='test-paper-doll'
gcr_url='eu.gcr.io'
json_file='test-paper-doll-0e7488b90747.json'
docker login -u _json_key -p "$(cat $json_file)" https://$gcr_url

#docker pull eu.gcr.io/test-paper-doll/tg/base:1
