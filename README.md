# Open Data Hub Tensorflo Jobs demo

This repository contains artifacts for demoing a machine learning pipeline with Jupyter and Tensorflow on top of OpenShift.

# Content

## OpenShift Templates

You can find all the necessary OpenShift templates in `openshift` directory. Namely those are:

* Jupyter Notebook workspace
* Tensorflow training job
* Tensorflow serving container
* OpenShift secret for object storage configuration and credentials

## Training

You can find a script which runs the training and its dependencies in the `training` directory.

The training dataset was generated using (https://github.com/CermakM/char-generator) and can be found in the [`training` directory as well](training/num-dataset.tar.xz)

## Serving

The `serving` directory contains `s2i` build and run scripts which are used for TF serving container

# How to run

You can use `install.sh` script in `openshift` directory to deploy all templates into your OpenShift cluster:

```
cd openshift
bash install.sh
```

You can specify `GIT_REF` environment variable to pick a git refernce (branch or commmit) to use - `master` is used by default. You can also use `LOCAL` environment variable (with any value) to deploy from local clone of the repo.

Once the templates are imported you can either deploy them from OpenShift Catalog - you can filter for provider **Open Data Hub**; or directly from command line using OpenShift clients.

## Controlling from Jupyter Notebook

There is an example notebook ready in this repository - [odh-tensorflow-jobs.ipynb](https://github.com/vpavlin/odh-tensorflow-jobs/blob/master/odh-tensorflow-jobs.ipynb). To be able to run it successfully you will first need to deploy Jupyter Notebook server - simply head over to OpenShift Catalog in your namespace and click the Jupyter Workspace template. Follow the wizard and you should get a URL to your Jupyter Notebook server displayed in your namespace overview.

Login to your Jupyter server and upload the notebook from this repository. Follow the notebook to get a model trained and served.   