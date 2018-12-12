#!/usr/bin/bash


[ -z "$GIT_REF" ] && GIT_REF="master"
[ -n "$LOCAL" ] && GIT_REF=

function _install() {
    f=$1
    if [ -z "${LOCAL}" ]; then
        f="https://raw.githubusercontent.com/vpavlin/odh-tensorflow-jobs/${GIT_REF}/openshift/$1"
    fi
    echo "==> ${f}"
    oc apply -f ${f}
}

_install "jupyter/jupyter-tensorflow.json"
_install "odh-config.yaml"
_install "odh-tensorflow-serving.bc.yaml"
_install "serving.yaml"
_install "training.job.yaml"
