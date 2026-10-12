#!/bin/bash
# AVAIL-AI web interface entrypoint (Hugging Face Space / any Docker host).
set -eu

if [ -z "${AVAIL_WEB_PASSWORD:-}" ]; then
    AVAIL_WEB_PASSWORD="tmp-$(head -c6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    echo "WARNING: AVAIL_WEB_PASSWORD secret not set; temporary password: ${AVAIL_WEB_PASSWORD}" >&2
    export AVAIL_WEB_PASSWORD
fi

mkdir -p /app/uploads /app/data_demos

exec waitress-serve --port="${AVAIL_WEB_PORT:-8090}" --threads=8 \
     --call webapp.app:create_app
