#!/bin/bash

export LOCAL_UID=$(id -u $USER)
export LOCAL_GID=$(id -g $USER)
export WORKING_DIR=$(cd ${PWD}/.. && pwd)
export ARTIFACT_DIR=$(cd ../artifact && pwd)
#docker-compose config
docker-compose up -d

#------------------test--------------------
# docker run -it --rm \
#        -e LOCAL_UID=$(id -u $USER) \
#        -e LOCAL_GID=$(id -g $USER) \
#        -e WORKING_DIR=$(cd ${PWD}/.. && pwd) \
#        -e ARTIFACT_DIR=$(cd ../artifact && pwd) \
#        -p 5000:5000 \
#        -v ${PWD}/..:${PWD}/.. \
#        mlflow_test bash
