#!/bin/bash

USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}

echo "Starting with UID : $USER_ID, GID: $GROUP_ID"
useradd -u $USER_ID -o -m user
groupmod -g $GROUP_ID user
passwd root <<EOF
root
root
EOF

chmod 777 -R ${ARTIFACT_DIR}
#exec /usr/sbin/gosu user "$@"
exec /usr/sbin/gosu user mlflow server --backend-store-uri ${DB_URI} --default-artifact-root ${ARTIFACT_DIR}/mlruns --host 0.0.0.0 --port 5000
