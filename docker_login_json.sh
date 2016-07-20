#!/usr/bin/env bash

docker login -u _json_key -p "$(cat test-paper-doll-0e7488b90747.json)" https://$gcr_url

#docker pull eu.gcr.io/test-paper-doll/tg/base:1
