#!/bin/bash
set -e

git -C /usr/lib/python2.7/dist-packages/trendi pull
exec "$@"

