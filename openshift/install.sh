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
_install "odh-tensorflow-training.bc.yaml"
_install "serving.yaml"
_install "training.job.yaml"

if [ "${GIT_REF}" == "knative" ]; then

    #istio needs this:
    oc adm policy add-scc-to-user anyuid -z default
    oc adm policy add-scc-to-user privileged -z default

    #Deploy Ceph
    _install "ceph/ceph-nano.sa.yaml"
    _install "ceph/ceph-nano-rgw.secret.yaml"
    _install "ceph/ceph-nano.svc.yaml"
    _install "ceph/ceph-nano.statefulset.yaml"

    oc adm policy add-scc-to-user anyuid -z ceph

    #Install serving knative template
    _install "serving.knative.yaml"
fi